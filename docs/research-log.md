# Research log

Working notes for lexrag / KnowRights: what was built, what was decided and why, what the data
showed, and the results so far. This is raw material for the README, the dissertation and
interviews, not the write-up itself.

> **Provenance.** Built with AI pair-programming (Claude Code). This log was first drafted by
> Claude from the session history and is to be reviewed and rewritten by Dylan. Commit hashes
> refer to this repository; result files are under `results/`.

---

## 1. Timeline

| Date | Phase | Commits |
|---|---|---|
| 2026-09-30 | Restart from an empty folder. Real XML/JSON inspected **before** writing the parser. Ingest layer. | `c5e2977`, `718445c` |
| 2026-10-01 | Test suite, Python 3.12 pin, CI on Linux and Windows. | `9d2fafe` to `574c042` |
| 2026-10-01 | Data-quality fix (repeal placeholders), PyTorch with CUDA, chunking. | `f4cadb8`, `00030e1`, `5e2ed1e` |
| 2026-10-01 | Seed test set: drafted, rewritten in workers' voice, verified by Dylan. | `7df409c`, `c81274b`, `8cf9701` |
| 2026-10-01 | BM25, requirement-group gold labels, metrics, eval runner, standard analyzer. | `7ac12a6` to `6805908` |

State at the end of 2026-10-01: 109 tests, ruff + strict mypy clean, CI green.

---

## 2. Decisions

Each decision: what, why, and the evidence behind it.

### D1. RAG, not long context
- **What:** retrieve a few sections per question rather than put the whole corpus in the prompt.
- **Why:** the hospitality corpus is ~1.15M characters (~290k tokens) and a second domain will
  add more. The local model (qwen3:8b) has a ~32k native context, and 16 GB of VRAM can't hold
  the attention cache for 290k tokens anyway. Evaluation means hundreds of questions times many
  configurations. And RAG makes retrieval measurable on its own (Recall@k), separately from
  answer quality.
- **Not pursued (for now):** a long-context baseline (give the model the whole relevant Act).
  Considered and parked; it could be added later as an ablation row.

### D2. Engineering foundations
- **uv + lockfile** (`uv.lock`) for reproducible installs; CI installs with `--locked`, so it
  fails if the lockfile is stale.
- **ruff** (lint and format), **mypy --strict**, **pytest**. CI runs all four on **Ubuntu and
  Windows** for every push. Windows is included because a path-handling difference has already
  appeared once (`/elsewhere` is not absolute on Windows).
- **Python 3.12 pinned** (`9d2fafe`). An application pins the version it is tested on; a library
  would support a range. numpy 2.5's type stubs need 3.12, which broke type-checking under a 3.11
  target.
- **Project moved out of OneDrive** to `C:\dev\lexrag`: syncing a virtualenv (and later about
  3 GB of PyTorch) file by file is slow and risks file locks. Git and GitHub are the backup.
- **PyTorch from PyTorch's CUDA 13.0 index** (`00030e1`). PyPI's Windows wheels are CPU-only, and
  the RTX 5060 Ti (Blackwell, compute capability 12.0) needs a CUDA 12.8+ build. Verified:
  `torch 2.14.1+cu130`, `sm_120` kernels, matmul on `cuda:0`.

### D3. Sources (domains/hospitality/domain.yaml)
- **Tips Act 2023 not ingested.** It is an amending Act: its provisions sit inside
  `BlockAmendment` elements and are inserted into ERA 1996 as **Part 2B (s.27C–27Y)**. Ingesting
  both would index the same law twice. Tips questions are labelled against `era1996`.
- **NMW Regulations 2015 added.** The Act sets up the minimum wage; the rates (£12.71 / £10.85 /
  £8.00), the accommodation offset (£11.10/day) and the rule that tips don't count (reg 10(m))
  are all in the Regulations.
- **Schedules excluded where they're noise** (an `exclude` list in `domain.yaml`, matched on
  whole-id boundaries so `schedule-3` doesn't catch `schedule-30`):

  | Source | Excluded | Why | Sections removed |
  |---|---|---|---|
  | era1996 | Sch 1–3 | consequential amendments, transitional provisions, repeals table (one paragraph is 37k chars) | 78 |
  | nmwa1998 | Sch 2 | amendments to the Agricultural Wages Acts | 23 |
  | fsa1990 | Sch 2–4 | amendments to other Acts, transitional provisions | 57 |

### D4. Parsing: what a "section" is
- One Section per top-level provision (`P1` with an id): `section-86`, `regulation-12`,
  `schedule-1-paragraph-2`. Citation key `"<doc_id>#<section_id>"`.
- `P1`s inside `BlockAmendment` belong to the enclosing section's text (text being inserted into
  another Act), not to this document.
- Alternative wording for another extent (e.g. ERA s.236 for N.I.) lives in a separate
  `Versions` block; only the main body is parsed. Labels come from ids, because those provisions'
  URIs carry an extent suffix.
- Fully repealed provisions (dotted title, no text) are **dropped**.
- Repealed **parts** of provisions: spaced dotted runs (`. . . .`) become `[no longer in force]`
  (`f4cadb8`). Wording chosen because the dots mark text that was repealed (Acts), revoked
  (Regulations) or omitted, so "repealed" would be wrong for a regulation. Unspaced `...` is a
  real in-sentence omission and is kept.
- Editorial commentaries are removed from the text but kept as `Section.notes` (types F, I, E, C;
  marginal citations M dropped). They are not embedded.
- GOV.UK: guides give one Section per part; single-page formats are split at `<h2>`.

### D5. Chunking (`5e2ed1e`)
- **Structure-aware (default):** a section is one chunk; if longer than **1,800 characters** it
  is split *between lines* (between numbered provisions, never mid-sentence), with up to 200
  characters of whole lines carried over. Each chunk's embedded text starts with a header such as
  `Employment Rights Act 1996 > Part X: Unfair dismissal > ... > section 104I: ...`.
- **Why 1,800 and not the planned 2,500:** bge-small-en-v1.5 reads 512 tokens and silently drops
  the rest. Measured with its tokenizer on this corpus: legislation **4.47 chars/token**, guidance
  4.59; header median 26 tokens, p95 41. The budget is 512 − 2 − 41 ≈ 469 tokens, about
  2,100 chars on average, so 1,800 leaves margin for dense text. A 2,500 limit would have
  silently truncated a large share of the law.
- **Fixed-window baseline** for the ablation: per-document windows of 1,800 chars, 200 overlap,
  no header. Made fair on purpose: windows start and end on word boundaries (a test caught them
  starting mid-word). 73% of windows straddle a section boundary; each records every section it
  touches.
- **Search returns sections, not chunks:** top-k means k distinct sections, each represented by
  its best-scoring chunk, because gold labels and citations are per section.

### D6. The test set (domains/hospitality/eval/questions.jsonl)
- 21 seed questions: 10 factual, 4 multi-hop, 3 false-premise, 2 unanswerable, 2 out-of-scope.
- **Written in workers' own voice** (lower case, context, vague), e.g. *"finished at 1am after a
  late close and the rota has me back in at 9am. is that even allowed??"* Textbook phrasing would
  flatter any retriever.
- **`verified` flag = a person checked the row against the source text.** Claude drafted and
  checked the rows; Dylan reviewed and verified them (`8cf9701`). The flag was not set by the AI.
- **Gold labels are requirement groups** (`85d970d`): a list of requirements, each met by *any*
  of its sections. `[["era1996#section-8", "govuk-payslips#intro"]]` means the Act or the
  guidance answers it; `[["wtr1998#regulation-12"], ["wtr1998#regulation-10"]]` means a
  multi-hop question needs both. Motivated by BM25 ranking `govuk-payslips#intro` (a correct
  source) for a question whose gold was only ERA s.8: single-key gold under-reported every method.
- **Rule for gold:** a section is gold only if it would let the system answer *on its own*.
  Guidance was added as an alternative only where the passage states the answer (quoted during
  review). ERA s.22 (final pay) was left out of hosp-014 because the worker didn't say they were
  leaving; WTR reg 15F (a definition only) was taken out of hosp-021.

### D7. Retrieval metrics (`a350c81`)
- **Recall@k** (share of requirements met in the top k), **Hit@k** (at least one),
  **Complete@k** (all met: what a multi-hop answer needs), **MRR**, **nDCG@k** (binary gain,
  capped at 1 when one section meets two requirements).
- A retrieved key matches gold if it is the same section, a sub-provision of it, or the gold
  names a whole document.
- Refusal questions are excluded from retrieval metrics; refusal accuracy is an answer-level
  metric, measured once generation exists.
- Every run is saved as JSON with its configuration, mean latency, and the **SHA-256 of the
  corpus and test set** used.

### D8. BM25 baseline: the standard analyzer (`6805908`)
- `bm25` = Okapi BM25 with Lucene's default English stop set (33 words, also Elasticsearch's
  `_english_`) plus Snowball stemming. `bm25-raw` = lower-cased tokens only, kept as an ablation
  row.
- **The stop list was chosen before seeing results.** Longer lists would remove "i", "my", "do";
  picking whichever scores best on 21 questions would be tuning the baseline to the test set.
- Motivation: an analyzer-less BM25 is weaker than what any real search engine does, so comparing
  dense search against it would overstate dense search's advantage.

### D9. Dense retrieval (`dbc6807`)
- **Model:** `BAAI/bge-small-en-v1.5` (384-dim, 512-token window, ~130 MB), run locally on the
  RTX 5060 Ti. Chunks are embedded once (header + text); search is cosine similarity
  (normalised vectors, so a dot product), grouped into sections like BM25.
- **Query instruction:** queries (not passages) get bge's retrieval prefix
  `"Represent this sentence for searching relevant passages: "`, as its model card recommends
  for short-query to long-passage search.
- Dense search always returns k results (there is no "no match"), so the BM25 rule "drop
  scores <= 0" does not apply to it.
- Tested without a GPU via a fake embedder (CI has no torch); a real-model test, skipped where
  the `ml` extra is missing, checks that basa/cod finds food-quality s.14 and payslips finds the
  itemised-pay-statement s.8, which BM25 cannot.

### D10. Plan for the rest of the retrieval ablation
`BM25 → dense → hybrid (RRF) → + LLM query rewriting → + reranker`, each a row with Recall@5 and
latency. Once the test set grows towards 150–200 it will be split into a **dev set** (for tuning)
and a **held-out test set** (run once at the end).

---

## 3. Findings: things the real data taught us

**Data**
- legislation.gov.uk marks removed text with spaced dots: ~32 tokens of nothing per line. In WTR
  reg 13 (annual leave) they pushed real content past the 512-token window. Collapsing them
  saved 7,768 characters across 138 sections and cut over-length chunks from 5 to 3. The 3 left
  (ERA s.108, s.209, s.236) are dense cross-reference lists (~3 chars/token) and out of scope.
- ERA 2025 inserted zero-hours rights (ERA 1996 s.27BA onwards) that are only in force "for
  specified purposes". The commencement notes are kept in `Section.notes`, but nothing uses them
  yet (see open issues).
- Workers' language and the law's language barely overlap: "letting me go" vs "terminate the
  contract", "payslip" vs "itemised pay statement", "basa when the menu says cod" vs "not of the
  nature or substance or quality demanded".

**Bugs caught by the process** (each fixed with a regression test)

| Bug | How it was found | Commit |
|---|---|---|
| GOV.UK HTML: text before the first tag was silently dropped; HTML comment text leaked into output | a mypy "unreachable" warning, then a runtime check | `718445c` |
| GOV.UK lists: `<li><p>…</p></li>` items lost their bullet | golden-output diff on the real corpus | `718445c` |
| Manifest out of sync after a refresh that failed halfway (new raw file, stale hash) | writing the pipeline failure tests | `edd09d0` |
| Fixed-window baseline started windows mid-word (an unfair baseline) | chunking test | `5e2ed1e` |

**Practices that paid off**
- Inspecting the real XML before writing the parser (amending Acts, extent versions and repeal
  shells were all unknown beforehand).
- **Golden-output diffs:** before changing a parser, save its output on the real corpus and diff
  afterwards; every difference must be explained.
- **Mutation check:** comment out the code a test protects and confirm the test fails.
- **Test-first bug fixes:** reproduce, watch it fail, fix, commit both together.

---

## 4. Results so far

Retrieval, hospitality seed set (17 answerable questions), structure chunks, k = 5.

| Retriever | Recall@5 | Hit@5 | Complete@5 | MRR | nDCG@5 | Latency | Run |
|---|---|---|---|---|---|---|---|
| BM25, raw tokens | 0.35 | 0.35 | 0.35 | 0.25 | 0.28 | 2.0 ms | `20261001T204333Z_bm25-raw_structure.json` |
| BM25, standard analyzer | 0.35 | 0.35 | 0.35 | 0.31 | 0.32 | 1.6 ms | `20261001T204342Z_bm25_structure.json` |
| Dense, bge-small-en-v1.5 | **0.74** | 0.76 | **0.71** | **0.48** | **0.55** | 7.6 ms (+15.8 s index build) | `20261002T000318Z_dense_structure.json` |

**Reading:**
- The analyzer found nothing new (the same 6 hits and 11 misses) but **ranked the hits higher**
  (MRR +0.06) and made the misses more on-topic. For example, "keep half the tips" now returns
  ERA s.27C/27E/27X from the tips Part, just missing 27D, rather than holiday pages.
- Raw BM25's misses were dominated by two **magnet sections** (`govuk-holiday#calculate-leave-entitlement`,
  `govuk-breaks#taking-breaks`): long plain-English pages that match the filler words in casual
  questions. The analyzer reduced this.
- The remaining misses are **vocabulary mismatch**, which no keyword method can fix. That is the
  target for dense retrieval.
- **Dense more than doubled recall (0.35 → 0.74) and lost nothing:** every question BM25 found,
  dense also found, and misses fell from 11 to 5. All false-premise questions are now retrieved,
  and multi-hop Complete@5 went from 0.25 to 0.50. *Prediction recorded beforehand and wrong:* that
  dense would lose some exact-term questions BM25 got right.
- **Dense misses are near-misses:** the right neighbourhood, the wrong neighbour. ERA s.27C
  instead of 27D (tips); WTR reg 11 (weekly rest) instead of reg 10 (daily rest) for "finished
  at 1am, back at 9am"; ERA s.25 from the right Part (*Protection of wages*) instead of s.13/17/18
  for the till shortage; the employee's-notice guidance instead of the employer's notice in
  s.86.
- **Implication for the next rows:** BM25 also missed all 5, so hybrid fusion has little to add
  at the top (still to be measured). A reranker (which reads query and passage together) targets
  wrong-neighbour errors; query rewriting targets the ones that need inference (1am→9am means
  daily rest).
- **Caution:** with 17 questions, one question changing rank moves MRR by about 0.06. Report
  these as directions, not precise gains, until the test set is larger.

---

## 5. Open issues

- [ ] Evaluate the fixed-window baseline (`--chunking fixed`), not yet run.
- [ ] Partly-in-force provisions (ERA 2025 zero-hours rights): show commencement notes to the
      generator or flag them, and add test questions for this.
- [ ] Grow the test set from 21 towards 150–200; then split into dev and held-out test.
- [ ] CLI commands have no tests (typer `CliRunner` smoke tests).
- [ ] 3 chunks still exceed 512 tokens (ERA s.108, s.209, s.236): accepted, out of scope.
- [ ] Check MMU's rules on AI assistance for the dissertation; declare AI use as required.
