"""Turn gov.uk HTML into clean text, then split it into overlapping chunks."""
import re

from bs4 import BeautifulSoup

_BLOCKS = ["h1", "h2", "h3", "h4", "p", "li"]


def html_to_text(html: str) -> str:
    """Flatten HTML into paragraphs, keeping headings and bullets as light markdown."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(_BLOCKS):
        # A list item that wraps paragraphs or a nested list is captured via its children.
        if el.name == "li" and el.find(["p", "ul", "ol"]):
            continue
        text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            text = "## " + text
        elif el.name == "li":
            text = "- " + text
        lines.append(text)
    return "\n\n".join(lines)


def chunk_text(text: str, max_words: int = 200, overlap_words: int = 30) -> list[str]:
    """Split on paragraph boundaries into chunks of at most `max_words` words.

    Paragraphs longer than the limit are split with a sliding window that overlaps
    by `overlap_words`, so a sentence cut at a boundary still appears whole in one chunk.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            step = max_words - overlap_words
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start : start + max_words]))
                if start + max_words >= len(words):
                    break
            continue

        if current and current_len + len(words) > max_words:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            tail_len = len(tail.split())
            # Carry a short last paragraph into the next chunk as overlap.
            if tail_len <= overlap_words and tail_len + len(words) <= max_words:
                current, current_len = [tail], tail_len
            else:
                current, current_len = [], 0

        current.append(para)
        current_len += len(words)

    if current:
        chunks.append("\n\n".join(current))
    return chunks
