from app.chunking import chunk_text, html_to_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 200, 30) == ["Hello world."]


def test_chunks_respect_max_words():
    text = "\n\n".join(" ".join(["word"] * 50) for _ in range(10))
    chunks = chunk_text(text, max_words=120, overlap_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 120 for c in chunks)


def test_long_paragraph_uses_overlapping_window():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), max_words=200, overlap_words=30)
    assert all(len(c.split()) <= 200 for c in chunks)
    assert chunks[1].split()[:30] == chunks[0].split()[-30:]
    assert chunks[-1].split()[-1] == "w499"  # nothing lost at the end


def test_overlap_must_be_smaller_than_max():
    try:
        chunk_text("a b c", max_words=10, overlap_words=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_html_to_text_keeps_headings_and_bullets():
    html = "<h2>Deposits</h2><p>Your landlord <a href='#'>must</a> protect it.</p><ul><li>One</li><li>Two</li></ul>"
    text = html_to_text(html)
    assert "## Deposits" in text
    assert "Your landlord must protect it." in text
    assert "- One" in text and "- Two" in text
