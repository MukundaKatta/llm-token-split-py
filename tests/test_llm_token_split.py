"""Tests for llm-token-split-py."""
import pytest
from llm_token_split import (
    Chunk, split_by_chars, split_by_tokens, split_by_sentences, iter_chunks
)

TEXT = "Hello world this is a test of chunking text into smaller pieces."


def test_split_basic():
    chunks = split_by_chars("abcdefgh", chunk_size=4)
    assert len(chunks) == 2
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "efgh"
    assert chunks[1].is_last is True


def test_split_with_overlap():
    chunks = split_by_chars("abcdefgh", chunk_size=4, overlap=2)
    # step = 4 - 2 = 2
    # 0..4 = abcd, 2..6 = cdef, 4..8 = efgh
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "cdef"
    assert chunks[2].text == "efgh"


def test_split_exact_size():
    chunks = split_by_chars("abcd", chunk_size=4)
    assert len(chunks) == 1
    assert chunks[0].text == "abcd"
    assert chunks[0].is_last is True


def test_split_empty_text():
    chunks = split_by_chars("", chunk_size=10)
    assert chunks == []


def test_split_chunk_metadata():
    chunks = split_by_chars("abcdefgh", chunk_size=4)
    assert chunks[0].index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 4
    assert chunks[1].index == 1
    assert chunks[1].char_start == 4
    assert chunks[1].char_end == 8


def test_split_is_last():
    chunks = split_by_chars("abcdef", chunk_size=4)
    assert chunks[0].is_last is False
    assert chunks[1].is_last is True


def test_split_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_by_chars("abc", chunk_size=0)


def test_split_invalid_overlap():
    with pytest.raises(ValueError):
        split_by_chars("abc", chunk_size=4, overlap=-1)


def test_split_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError):
        split_by_chars("abc", chunk_size=4, overlap=4)


def test_split_strip_disabled():
    chunks = split_by_chars("  ab  ", chunk_size=6, strip_chunks=False)
    assert chunks[0].text == "  ab  "


def test_split_strip_enabled():
    chunks = split_by_chars("  ab  ", chunk_size=6, strip_chunks=True)
    assert chunks[0].text == "ab"


def test_split_single_char_chunk():
    chunks = split_by_chars("abc", chunk_size=1)
    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["a", "b", "c"]


def test_iter_chunks():
    texts = list(iter_chunks("abcdefgh", chunk_size=4))
    assert texts == ["abcd", "efgh"]


def test_iter_chunks_with_overlap():
    texts = list(iter_chunks("abcdefgh", chunk_size=4, overlap=2))
    assert texts[0] == "abcd"
    assert texts[1] == "cdef"
    assert texts[2] == "efgh"


def test_split_by_tokens_basic():
    chunks = split_by_tokens("a" * 100, chunk_tokens=10, chars_per_token=4.0)
    # 10 tokens * 4 chars = 40 chars per chunk; 100 chars / 40 = 3 chunks
    assert len(chunks) >= 2


def test_split_by_tokens_overlap():
    text = "a" * 120
    chunks = split_by_tokens(text, chunk_tokens=10, overlap_tokens=2, chars_per_token=4.0)
    # chunk_size=40, overlap=8, step=32
    # should produce overlapping chunks
    assert len(chunks) >= 2
    assert chunks[0].char_start == 0
    assert chunks[1].char_start == 32  # step = 40 - 8 = 32


def test_split_by_sentences_basic():
    text = "Hello world. This is a test. Another sentence."
    chunks = split_by_sentences(text, max_chunk_chars=50)
    assert len(chunks) >= 1
    # all text covered
    combined = " ".join(c.text for c in chunks)
    for word in ["Hello", "test", "Another"]:
        assert word in combined


def test_split_by_sentences_overlap():
    text = "First. Second. Third. Fourth."
    chunks = split_by_sentences(text, max_chunk_chars=15, overlap_sentences=1)
    assert len(chunks) >= 2


def test_split_by_sentences_empty():
    chunks = split_by_sentences("", max_chunk_chars=100)
    assert chunks == []


def test_split_by_sentences_last_is_last():
    text = "One. Two. Three."
    chunks = split_by_sentences(text, max_chunk_chars=200)
    assert chunks[-1].is_last is True


def test_chunk_dataclass():
    c = Chunk(text="hello", index=0, char_start=0, char_end=5, is_last=True)
    assert c.text == "hello"
    assert c.index == 0
    assert c.char_start == 0
    assert c.char_end == 5
    assert c.is_last is True
