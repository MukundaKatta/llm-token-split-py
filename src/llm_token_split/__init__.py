"""llm-token-split-py — split long documents into overlapping chunks for LLM context windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    """A single text chunk with position metadata."""

    text: str
    index: int  # 0-based chunk index
    char_start: int  # start offset in the original text
    char_end: int  # end offset (exclusive)
    is_last: bool  # True for the final chunk


def split_by_chars(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    strip_chunks: bool = True,
) -> list[Chunk]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text: The text to split.
        chunk_size: Maximum character length per chunk.
        overlap: Number of characters to repeat at the start of each new chunk.
        strip_chunks: Strip leading/trailing whitespace from each chunk.

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If chunk_size < 1 or overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    pos = 0
    index = 0

    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        raw = text[pos:end]
        content = raw.strip() if strip_chunks else raw
        is_last = end >= len(text)
        chunks.append(
            Chunk(
                text=content,
                index=index,
                char_start=pos,
                char_end=end,
                is_last=is_last,
            )
        )
        if is_last:
            break
        pos += step
        index += 1

    return chunks


def split_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int = 0,
    chars_per_token: float = 4.0,
    strip_chunks: bool = True,
) -> list[Chunk]:
    """
    Split text into chunks sized by approximate token count.

    Uses a chars_per_token approximation (default 4.0).

    Args:
        text: The text to split.
        chunk_tokens: Target number of tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.
        chars_per_token: Characters per token approximation.
        strip_chunks: Strip leading/trailing whitespace from each chunk.

    Returns:
        List of Chunk objects.
    """
    chunk_size = max(1, int(chunk_tokens * chars_per_token))
    overlap = int(overlap_tokens * chars_per_token)
    return split_by_chars(
        text, chunk_size=chunk_size, overlap=overlap, strip_chunks=strip_chunks
    )


def split_by_sentences(
    text: str,
    max_chunk_chars: int,
    overlap_sentences: int = 0,
    sentence_sep: str = ".",
) -> list[Chunk]:
    """
    Split text at sentence boundaries, keeping chunks under max_chunk_chars.

    Sentences are detected by the sentence_sep character. Chunks are built
    greedily — sentences are added until the next one would overflow.

    Args:
        text: The text to split.
        max_chunk_chars: Maximum character length per chunk.
        overlap_sentences: Number of sentences to repeat at the start of each chunk.
        sentence_sep: Separator character used to detect sentence boundaries.

    Returns:
        List of Chunk objects.
    """
    # Split into sentences
    raw_sentences = [
        s.strip() + sentence_sep for s in text.split(sentence_sep) if s.strip()
    ]
    if not raw_sentences:
        return []

    chunks: list[Chunk] = []
    index = 0
    i = 0
    char_pos = 0

    while i < len(raw_sentences):
        current: list[str] = []
        current_len = 0
        j = i

        while j < len(raw_sentences):
            s = raw_sentences[j]
            sep = " " if current else ""
            candidate_len = current_len + len(sep) + len(s)
            if current and candidate_len > max_chunk_chars:
                break
            current.append(s)
            current_len = candidate_len
            j += 1

        if not current:
            # Single sentence longer than max — include it anyway
            current = [raw_sentences[i]]
            j = i + 1

        chunk_text = " ".join(current)
        start = char_pos
        end = start + len(chunk_text)
        is_last = j >= len(raw_sentences)
        chunks.append(
            Chunk(
                text=chunk_text,
                index=index,
                char_start=start,
                char_end=end,
                is_last=is_last,
            )
        )
        char_pos = end + 1
        index += 1

        if is_last:
            # This chunk already covers the remaining sentences; emitting
            # another (overlapping) chunk would duplicate content and produce
            # a second chunk flagged is_last.
            break

        # Advance with optional sentence overlap
        i = max(i + 1, j - overlap_sentences)

    return chunks


def iter_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[str]:
    """Iterate over chunk texts without materializing the full list."""
    for chunk in split_by_chars(text, chunk_size=chunk_size, overlap=overlap):
        yield chunk.text


__all__ = [
    "Chunk",
    "split_by_chars",
    "split_by_tokens",
    "split_by_sentences",
    "iter_chunks",
]
