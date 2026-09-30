"""End-to-end ingest against a fake legislation.gov.uk / GOV.UK: no network needed."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from lexrag.config import Domain, Settings
from lexrag.ingest import ingest, load_sections

FIXTURES = Path(__file__).parent / "fixtures"
LEGISLATION_URL = "https://www.legislation.gov.uk/ukpga/1996/18/data.xml"
GOVUK_URL = "https://www.gov.uk/api/content/rest-breaks-work"
RESPONSES = {
    LEGISLATION_URL: (FIXTURES / "clml_sample.xml").read_bytes(),
    GOVUK_URL: (FIXTURES / "govuk_guide.json").read_bytes(),
}

DOMAIN = Domain(
    name="demo",
    title="Demo",
    description="A test domain.",
    jurisdiction="England",
    audience="Testers",
    in_scope=["testing"],
    out_of_scope=["everything else"],
    disclaimer="Not legal advice.",
    sources=[
        {"id": "era1996", "type": "legislation", "ref": "ukpga/1996/18"},  # type: ignore[list-item]
        {"id": "govuk-breaks", "type": "govuk", "path": "/rest-breaks-work"},  # type: ignore[list-item]
    ],
)


class FakeServer:
    """Serves the fixtures; `fail` maps a URL to status codes to return before succeeding."""

    def __init__(
        self, fail: dict[str, list[int]] | None = None, responses: dict[str, bytes] | None = None
    ) -> None:
        self.requests: list[str] = []
        self.fail = fail or {}
        self.responses = {**RESPONSES, **(responses or {})}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if self.fail.get(url):
            return httpx.Response(self.fail[url].pop(0))
        if url in self.responses:
            return httpx.Response(200, content=self.responses[url])
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(root=tmp_path)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry back-off instead of actually sleeping."""
    calls: list[float] = []
    sleep: Callable[[float], None] = calls.append
    monkeypatch.setattr("lexrag.ingest.pipeline.time.sleep", sleep)
    return calls


def test_first_run_downloads_parses_and_writes_everything(settings: Settings) -> None:
    server = FakeServer()
    reports = ingest(DOMAIN, settings, client=server.client())

    assert server.requests == [LEGISLATION_URL, GOVUK_URL]
    assert [(r.source_id, r.sections, r.cached) for r in reports] == [
        ("era1996", 4, False),
        ("govuk-breaks", 2, False),
    ]
    raw = settings.data_path("demo", "raw")
    assert (raw / "era1996.xml").read_bytes() == RESPONSES[LEGISLATION_URL]
    assert (raw / "govuk-breaks.json").read_bytes() == RESPONSES[GOVUK_URL]

    keys = [s.key for s in load_sections("demo", settings)]
    assert keys[0] == "era1996#section-86"
    assert keys[-1] == "govuk-breaks#young-workers"
    assert len(keys) == 6


def test_manifest_records_hash_and_url(settings: Settings) -> None:
    ingest(DOMAIN, settings, client=FakeServer().client())
    manifest = json.loads(settings.data_path("demo", "raw", "manifest.json").read_text("utf-8"))

    entry = manifest["era1996"]
    assert entry["url"] == LEGISLATION_URL
    assert entry["file"] == "era1996.xml"
    assert entry["sha256"] == hashlib.sha256(RESPONSES[LEGISLATION_URL]).hexdigest()
    assert entry["retrieved_at"]


def test_second_run_uses_cache(settings: Settings) -> None:
    ingest(DOMAIN, settings, client=FakeServer().client())
    server = FakeServer()
    reports = ingest(DOMAIN, settings, client=server.client())

    assert server.requests == []
    assert all(r.cached for r in reports)
    assert len(load_sections("demo", settings)) == 6


def test_refresh_downloads_again(settings: Settings) -> None:
    ingest(DOMAIN, settings, client=FakeServer().client())
    server = FakeServer()
    ingest(DOMAIN, settings, refresh=True, client=server.client())
    assert server.requests == [LEGISLATION_URL, GOVUK_URL]


def test_server_errors_are_retried_with_backoff(settings: Settings, sleeps: list[float]) -> None:
    server = FakeServer(fail={LEGISLATION_URL: [503, 502]})
    ingest(DOMAIN, settings, client=server.client())

    assert server.requests.count(LEGISLATION_URL) == 3
    assert sleeps == [1, 2]


def test_client_errors_fail_fast_and_keep_previous_output(
    settings: Settings, sleeps: list[float]
) -> None:
    ingest(DOMAIN, settings, client=FakeServer().client())
    before = settings.data_path("demo", "sections.jsonl").read_bytes()

    broken = FakeServer(fail={GOVUK_URL: [404]})
    with pytest.raises(httpx.HTTPStatusError):
        ingest(DOMAIN, settings, refresh=True, client=broken.client())

    assert broken.requests.count(GOVUK_URL) == 1  # a 404 is not worth retrying
    assert sleeps == []
    assert settings.data_path("demo", "sections.jsonl").read_bytes() == before


def test_manifest_matches_raw_files_even_after_a_failed_refresh(
    settings: Settings, sleeps: list[float]
) -> None:
    ingest(DOMAIN, settings, client=FakeServer().client())
    # Upstream amends the Act, then the GOV.UK download fails halfway through the refresh.
    amended = RESPONSES[LEGISLATION_URL].replace(b"statutory instrument", b"order")
    server = FakeServer(fail={GOVUK_URL: [404]}, responses={LEGISLATION_URL: amended})
    with pytest.raises(httpx.HTTPStatusError):
        ingest(DOMAIN, settings, refresh=True, client=server.client())

    raw = settings.data_path("demo", "raw")
    manifest = json.loads((raw / "manifest.json").read_text("utf-8"))
    for source_id, entry in manifest.items():
        actual = hashlib.sha256((raw / entry["file"]).read_bytes()).hexdigest()
        assert entry["sha256"] == actual, f"manifest out of date for {source_id}"


def test_load_sections_before_ingest_explains_what_to_do(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="run `lexrag ingest demo` first"):
        load_sections("demo", settings)
