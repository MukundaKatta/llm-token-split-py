"""Tests for llm-token-split-py.

These tests use only the Python standard library (``unittest``) so they can be
run without any third-party dependencies::

    python -m unittest discover -s tests
"""

import os
import sys
import unittest

# Make the package importable when running from a source checkout (i.e. without
# having installed the package first). ``src`` is added to ``sys.path`` only if
# the package cannot already be imported, so installed-package runs are
# unaffected.
try:  # pragma: no cover - exercised implicitly by the import path chosen
    import llm_token_split  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, _SRC)

from llm_token_split import (  # noqa: E402
    Chunk,
    iter_chunks,
    split_by_chars,
    split_by_sentences,
    split_by_tokens,
)

TEXT = "Hello world this is a test of chunking text into smaller pieces."


class SplitByCharsTests(unittest.TestCase):
    def test_split_basic(self):
        chunks = split_by_chars("abcdefgh", chunk_size=4)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "abcd")
        self.assertEqual(chunks[1].text, "efgh")
        self.assertTrue(chunks[1].is_last)

    def test_split_with_overlap(self):
        chunks = split_by_chars("abcdefgh", chunk_size=4, overlap=2)
        # step = 4 - 2 = 2: 0..4 = abcd, 2..6 = cdef, 4..8 = efgh
        self.assertEqual([c.text for c in chunks], ["abcd", "cdef", "efgh"])

    def test_split_exact_size(self):
        chunks = split_by_chars("abcd", chunk_size=4)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "abcd")
        self.assertTrue(chunks[0].is_last)

    def test_split_empty_text(self):
        self.assertEqual(split_by_chars("", chunk_size=10), [])

    def test_split_chunk_metadata(self):
        chunks = split_by_chars("abcdefgh", chunk_size=4)
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[0].char_end, 4)
        self.assertEqual(chunks[1].index, 1)
        self.assertEqual(chunks[1].char_start, 4)
        self.assertEqual(chunks[1].char_end, 8)

    def test_split_is_last(self):
        chunks = split_by_chars("abcdef", chunk_size=4)
        self.assertFalse(chunks[0].is_last)
        self.assertTrue(chunks[1].is_last)

    def test_exactly_one_chunk_is_last(self):
        # Regression guard: across overlapping output exactly one chunk is flagged.
        chunks = split_by_chars("abcdefghij", chunk_size=4, overlap=1)
        self.assertEqual(sum(c.is_last for c in chunks), 1)
        self.assertTrue(chunks[-1].is_last)

    def test_char_offsets_round_trip(self):
        # char_start/char_end must index back into the original text.
        text = "the quick brown fox jumps over the lazy dog"
        for c in split_by_chars(text, chunk_size=7, overlap=2, strip_chunks=False):
            self.assertEqual(text[c.char_start:c.char_end], c.text)

    def test_split_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            split_by_chars("abc", chunk_size=0)

    def test_split_negative_chunk_size(self):
        with self.assertRaises(ValueError):
            split_by_chars("abc", chunk_size=-3)

    def test_split_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_by_chars("abc", chunk_size=4, overlap=-1)

    def test_split_overlap_gte_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            split_by_chars("abc", chunk_size=4, overlap=4)

    def test_split_strip_disabled(self):
        chunks = split_by_chars("  ab  ", chunk_size=6, strip_chunks=False)
        self.assertEqual(chunks[0].text, "  ab  ")

    def test_split_strip_enabled(self):
        chunks = split_by_chars("  ab  ", chunk_size=6, strip_chunks=True)
        self.assertEqual(chunks[0].text, "ab")

    def test_split_single_char_chunk(self):
        chunks = split_by_chars("abc", chunk_size=1)
        self.assertEqual([c.text for c in chunks], ["a", "b", "c"])

    def test_chunk_size_larger_than_text(self):
        chunks = split_by_chars("abc", chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "abc")
        self.assertTrue(chunks[0].is_last)


class IterChunksTests(unittest.TestCase):
    def test_iter_chunks(self):
        self.assertEqual(list(iter_chunks("abcdefgh", chunk_size=4)), ["abcd", "efgh"])

    def test_iter_chunks_with_overlap(self):
        texts = list(iter_chunks("abcdefgh", chunk_size=4, overlap=2))
        self.assertEqual(texts, ["abcd", "cdef", "efgh"])

    def test_iter_chunks_is_lazy(self):
        # iter_chunks should return an iterator, not a materialized list.
        result = iter_chunks("abcdefgh", chunk_size=4)
        self.assertFalse(isinstance(result, list))
        self.assertEqual(next(iter(result)), "abcd")

    def test_iter_chunks_empty(self):
        self.assertEqual(list(iter_chunks("", chunk_size=4)), [])


class SplitByTokensTests(unittest.TestCase):
    def test_split_by_tokens_basic(self):
        chunks = split_by_tokens("a" * 100, chunk_tokens=10, chars_per_token=4.0)
        # 10 tokens * 4 chars = 40 chars per chunk; 100 chars / 40 -> 3 chunks
        self.assertGreaterEqual(len(chunks), 2)

    def test_split_by_tokens_overlap(self):
        chunks = split_by_tokens(
            "a" * 120, chunk_tokens=10, overlap_tokens=2, chars_per_token=4.0
        )
        # chunk_size=40, overlap=8, step=32
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[1].char_start, 32)

    def test_split_by_tokens_custom_ratio(self):
        # A different chars_per_token changes the effective chunk size.
        chunks = split_by_tokens("a" * 100, chunk_tokens=10, chars_per_token=2.0)
        # 10 tokens * 2 chars = 20 chars per chunk -> 5 chunks
        self.assertEqual(len(chunks), 5)

    def test_split_by_tokens_empty(self):
        self.assertEqual(split_by_tokens("", chunk_tokens=10), [])


class SplitBySentencesTests(unittest.TestCase):
    def test_split_by_sentences_basic(self):
        text = "Hello world. This is a test. Another sentence."
        chunks = split_by_sentences(text, max_chunk_chars=50)
        self.assertGreaterEqual(len(chunks), 1)
        combined = " ".join(c.text for c in chunks)
        for word in ["Hello", "test", "Another"]:
            self.assertIn(word, combined)

    def test_split_by_sentences_overlap(self):
        text = "First. Second. Third. Fourth."
        chunks = split_by_sentences(text, max_chunk_chars=15, overlap_sentences=1)
        self.assertGreaterEqual(len(chunks), 2)

    def test_split_by_sentences_empty(self):
        self.assertEqual(split_by_sentences("", max_chunk_chars=100), [])

    def test_split_by_sentences_whitespace_only(self):
        self.assertEqual(split_by_sentences("   \n  ", max_chunk_chars=100), [])

    def test_split_by_sentences_last_is_last(self):
        chunks = split_by_sentences("One. Two. Three.", max_chunk_chars=200)
        self.assertTrue(chunks[-1].is_last)

    def test_split_by_sentences_overlap_no_duplicate_last(self):
        # With overlap, a chunk that already reaches the end must not spawn a
        # redundant trailing chunk, and exactly one chunk must be flagged is_last.
        text = "First. Second. Third. Fourth."
        chunks = split_by_sentences(text, max_chunk_chars=15, overlap_sentences=1)
        self.assertEqual(sum(c.is_last for c in chunks), 1)
        self.assertTrue(chunks[-1].is_last)
        self.assertEqual(chunks[-1].text, "Third. Fourth.")

    def test_split_by_sentences_overlap_full_coverage(self):
        # Every sentence should appear at least once across overlapping chunks.
        text = "S1. S2. S3. S4. S5."
        chunks = split_by_sentences(text, max_chunk_chars=12, overlap_sentences=1)
        combined = " ".join(c.text for c in chunks)
        for s in ["S1", "S2", "S3", "S4", "S5"]:
            self.assertIn(s, combined)
        self.assertEqual(sum(c.is_last for c in chunks), 1)

    def test_split_by_sentences_oversized_single_sentence(self):
        # A single sentence longer than max_chunk_chars is still emitted whole.
        text = "This single sentence is far too long to fit in the limit."
        chunks = split_by_sentences(text, max_chunk_chars=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("single sentence", chunks[0].text)

    def test_split_by_sentences_custom_separator(self):
        text = "one! two! three!"
        chunks = split_by_sentences(text, max_chunk_chars=200, sentence_sep="!")
        combined = " ".join(c.text for c in chunks)
        for word in ["one", "two", "three"]:
            self.assertIn(word, combined)
        self.assertTrue(chunks[-1].is_last)

    def test_split_by_sentences_indices_sequential(self):
        text = "a. b. c. d. e."
        chunks = split_by_sentences(text, max_chunk_chars=6)
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))


class ChunkDataclassTests(unittest.TestCase):
    def test_chunk_dataclass(self):
        c = Chunk(text="hello", index=0, char_start=0, char_end=5, is_last=True)
        self.assertEqual(c.text, "hello")
        self.assertEqual(c.index, 0)
        self.assertEqual(c.char_start, 0)
        self.assertEqual(c.char_end, 5)
        self.assertTrue(c.is_last)

    def test_chunk_equality(self):
        a = Chunk(text="x", index=1, char_start=0, char_end=1, is_last=False)
        b = Chunk(text="x", index=1, char_start=0, char_end=1, is_last=False)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
