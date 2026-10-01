"""Command-line interface: `lexrag --help`."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lexrag import __version__
from lexrag.chunking import chunk_sections
from lexrag.config import ChunkingSettings, list_domains, load_domain, load_settings
from lexrag.eval.dataset import load_questions, questions_path
from lexrag.eval.retrieval import (
    METRICS,
    evaluate_retrieval,
    save_run,
    sha256_of,
    summarize,
)
from lexrag.index import BM25Retriever
from lexrag.ingest import ingest as run_ingest
from lexrag.ingest import load_sections
from lexrag.ingest.pipeline import sections_path
from lexrag.models import Retrieved

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console()

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="settings.yaml path")]
DomainArg = Annotated[str, typer.Argument(help="Domain name, i.e. a folder under domains/")]


@app.command()
def domains(config: ConfigOpt = None) -> None:
    """List available domains."""
    settings = load_settings(config)
    table = Table("name", "title", "sources", "description")
    for name in list_domains(settings):
        d = load_domain(name, settings)
        table.add_row(d.name, d.title, str(len(d.sources)), d.description)
    console.print(table)


@app.command()
def ingest(
    domain: DomainArg,
    refresh: Annotated[bool, typer.Option(help="Re-download even if cached")] = False,
    config: ConfigOpt = None,
) -> None:
    """Download (or reuse cached) sources for DOMAIN and parse them into sections."""
    settings = load_settings(config)
    reports = run_ingest(load_domain(domain, settings), settings, refresh=refresh)
    table = Table("source", "sections", "chars", "raw", title=f"Ingested {domain}")
    for r in reports:
        table.add_row(
            r.source_id, str(r.sections), f"{r.chars:,}", "cached" if r.cached else "downloaded"
        )
    table.add_row(
        "[bold]total",
        f"[bold]{sum(r.sections for r in reports)}",
        f"[bold]{sum(r.chars for r in reports):,}",
        "",
    )
    console.print(table)


@app.command()
def sections(
    domain: DomainArg,
    doc: Annotated[str | None, typer.Option(help="Only this source id")] = None,
    grep: Annotated[str | None, typer.Option(help="Regex to match against title or text")] = None,
    config: ConfigOpt = None,
) -> None:
    """List parsed sections, e.g. to find the right key for a gold label."""
    settings = load_settings(config)
    pattern = re.compile(grep, re.IGNORECASE) if grep else None
    table = Table("key", "title", "chars", "extent")
    for s in load_sections(domain, settings):
        if doc and s.doc_id != doc:
            continue
        if pattern and not (pattern.search(s.title) or pattern.search(s.text)):
            continue
        table.add_row(s.key, s.title, str(len(s.text)), s.extent or "")
    console.print(table)


@app.command()
def show(
    domain: DomainArg,
    key: Annotated[str, typer.Argument(help='Section key, e.g. "wtr1998#regulation-12"')],
    notes: Annotated[bool, typer.Option(help="Also print editorial notes")] = False,
    config: ConfigOpt = None,
) -> None:
    """Print one section exactly as it will be indexed."""
    settings = load_settings(config)
    match = next((s for s in load_sections(domain, settings) if s.key == key), None)
    if match is None:
        console.print(f"[red]no section {key!r} in {domain}")
        raise typer.Exit(1)
    console.rule(f"[bold]{match.citation}: {match.title}")
    console.print(f"[dim]{' > '.join(match.path)}\n{match.url}  extent={match.extent}")
    console.print(match.text, markup=False, highlight=False)
    if notes and match.notes:
        console.rule("notes")
        for note in match.notes:
            console.print(f"- {note}", markup=False, highlight=False)


class _Timed:
    """Wraps a retriever to record how long each search takes."""

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever
        self.seconds: list[float] = []

    def search(self, query: str, k: int) -> list[Retrieved]:
        start = time.perf_counter()
        results = self.retriever.search(query, k)
        self.seconds.append(time.perf_counter() - start)
        return results


@app.command("eval-retrieval")
def eval_retrieval(
    domain: DomainArg,
    mode: Annotated[str, typer.Option(help="Retriever: bm25")] = "bm25",
    chunking: Annotated[
        str | None, typer.Option(help="Chunking strategy: structure | fixed (default: settings)")
    ] = None,
    k: Annotated[int, typer.Option(min=1, help="Sections retrieved per question")] = 5,
    save: Annotated[bool, typer.Option(help="Write results/<domain>/retrieval/*.json")] = True,
    config: ConfigOpt = None,
) -> None:
    """Score retrieval on DOMAIN's test set: Recall@k, Hit@k, Complete@k, MRR, nDCG@k."""
    settings = load_settings(config)
    chunk_settings = ChunkingSettings.model_validate(
        settings.chunking.model_dump() | ({"strategy": chunking} if chunking else {})
    )
    if mode != "bm25":
        raise typer.BadParameter(f"unknown mode {mode!r}; available: bm25", param_hint="--mode")

    qpath = questions_path(domain, settings)
    questions = load_questions(qpath)
    if unverified := [q.id for q in questions if not q.verified]:
        console.print(f"[yellow]warning: {len(unverified)} unverified questions: {unverified}")

    chunks = chunk_sections(load_sections(domain, settings), chunk_settings)
    retriever = _Timed(BM25Retriever(chunks))
    results = evaluate_retrieval(retriever, questions, k)
    summary = summarize(results)
    latency_ms = 1000 * sum(retriever.seconds) / len(retriever.seconds)

    title = f"{domain}: {mode}, {chunk_settings.strategy} chunks, k={k}"
    table = Table(
        "category", "n", *(f"{m}@{k}" if m != "mrr" else "MRR" for m in METRICS), title=title
    )
    for name, row in summary.items():
        style = "bold" if name == "all" else ""
        cells = [f"{row[m]:.2f}" for m in METRICS]
        table.add_row(name, str(int(row["n"])), *cells, style=style)
    console.print(table)
    console.print(f"{len(chunks)} chunks; mean search latency {latency_ms:.1f} ms")

    misses = [r for r in results if r.recall < 1]
    if misses:
        console.rule(f"{len(misses)} questions with missing requirements")
        for r in misses:
            ranks = ", ".join("-" if x is None else str(x) for x in r.requirement_ranks)
            console.print(f"[bold]{r.id}[/] ranks [{ranks}]  {r.question}", markup=True)
            console.print(f"   got: {', '.join(r.retrieved[:3])}", markup=False, highlight=False)

    if save:
        run_config = {
            "domain": domain,
            "mode": mode,
            "chunking": chunk_settings.strategy,
            "max_chars": chunk_settings.max_chars,
            "overlap_chars": chunk_settings.overlap_chars,
            "k": k,
            "chunks": len(chunks),
            "mean_latency_ms": round(latency_ms, 2),
            "lexrag_version": __version__,
            "sections_sha256": sha256_of(sections_path(domain, settings)),
            "questions_sha256": sha256_of(qpath),
        }
        out = settings.resolve(settings.results_dir) / domain / "retrieval"
        console.print(f"saved {save_run(out, run_config, results, summary)}")


if __name__ == "__main__":
    app()
