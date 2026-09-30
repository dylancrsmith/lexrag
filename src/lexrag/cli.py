"""Command-line interface: `lexrag --help`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lexrag.config import list_domains, load_domain, load_settings
from lexrag.ingest import ingest as run_ingest
from lexrag.ingest import load_sections

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
        table.add_row(r.source_id, str(r.sections), f"{r.chars:,}", "cached" if r.cached else "downloaded")
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


if __name__ == "__main__":
    app()
