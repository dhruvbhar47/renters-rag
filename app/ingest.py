"""Fetch gov.uk guidance, chunk it, embed it and store it in Postgres.

Usage:
    python -m app.ingest --dry-run          # fetch + chunk only, print what would be stored
    python -m app.ingest                    # full ingest (needs OPENAI_API_KEY and a database)
    python -m app.ingest --chunk-words 120  # try a different chunk size, then re-run the eval
"""
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .chunking import chunk_text, html_to_text
from .config import settings

log = logging.getLogger("ingest")

GOVUK = "https://www.gov.uk"
HEADERS = {"User-Agent": "renters-rag-portfolio-project/0.1"}


@dataclass
class Document:
    url: str
    title: str
    updated_at: str | None
    text: str


def read_seeds(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def fetch_documents(path: str) -> list[Document]:
    """Use the gov.uk Content API (structured JSON); fall back to scraping the page."""
    resp = requests.get(f"{GOVUK}/api/content{path}", headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        log.warning("Not found, skipping: %s", path)
        return []
    resp.raise_for_status()
    data = resp.json()

    title = data.get("title", path)
    updated = (data.get("public_updated_at") or "")[:10] or None
    details = data.get("details", {})

    docs: list[Document] = []
    for part in details.get("parts") or []:  # multi-page guides
        text = html_to_text(part.get("body", ""))
        if text:
            docs.append(
                Document(f"{GOVUK}{path}/{part['slug']}", f"{title}: {part.get('title', '')}", updated, text)
            )
    if not docs and details.get("body"):  # single-page content
        text = html_to_text(details["body"])
        if text:
            docs.append(Document(f"{GOVUK}{path}", title, updated, text))
    if not docs:  # last resort: scrape the rendered page
        page = requests.get(f"{GOVUK}{path}", headers=HEADERS, timeout=20)
        page.raise_for_status()
        main = BeautifulSoup(page.text, "html.parser").find("main")
        text = html_to_text(str(main)) if main else ""
        if text:
            docs.append(Document(f"{GOVUK}{path}", title, updated, text))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/seed_urls.txt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    parser.add_argument("--overlap-words", type=int, default=settings.overlap_words)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    docs: list[Document] = []
    for path in read_seeds(Path(args.seeds)):
        found = fetch_documents(path)
        log.info("%s -> %d page(s)", path, len(found))
        docs.extend(found)

    if not docs:
        raise SystemExit("No documents fetched. Check data/seed_urls.txt and your connection.")

    store = None
    if not args.dry_run:
        from .openai_client import embed
        from .store import PgStore, connect, init_schema

        conn = connect()
        init_schema(conn)
        store = PgStore(conn)

    total = 0
    for doc in docs:
        chunks = chunk_text(doc.text, args.chunk_words, args.overlap_words)
        total += len(chunks)
        if args.dry_run:
            log.info("%s | %d chunks | updated %s", doc.url, len(chunks), doc.updated_at)
            continue
        # Prepend the title so each chunk carries its context when embedded.
        vectors = embed([f"{doc.title}\n\n{c}" for c in chunks])
        store.replace_document(doc.url, doc.title, doc.updated_at, chunks, vectors)
        log.info("stored %s (%d chunks)", doc.url, len(chunks))

    log.info("Done: %d documents, %d chunks%s", len(docs), total, " (dry run)" if args.dry_run else "")


if __name__ == "__main__":
    main()
