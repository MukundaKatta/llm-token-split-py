# llm-token-split-py

Split long documents into overlapping chunks for LLM context windows. Supports character-based, token-based, and sentence-aware splitting.

- **Zero dependencies** — pure standard library.
- **Typed** — ships a `py.typed` marker, so type checkers see the annotations.
- **Position metadata** — every chunk carries its `char_start` / `char_end` offsets back into the original text.

Requires Python 3.9+.

## Install

```bash
pip install llm-token-split-py
```

## Usage

```python
from llm_token_split import split_by_chars, split_by_tokens, split_by_sentences, iter_chunks

long_text = "..." * 5000

# Character-based splitting
chunks = split_by_chars(long_text, chunk_size=2000, overlap=200)
for chunk in chunks:
    print(chunk.index, chunk.char_start, chunk.char_end, chunk.is_last)
    llm.send(chunk.text)

# Token-approximate splitting (4 chars/token default)
chunks = split_by_tokens(long_text, chunk_tokens=500, overlap_tokens=50)

# Sentence-aware splitting
chunks = split_by_sentences(long_text, max_chunk_chars=2000, overlap_sentences=1)

# Memory-efficient iteration
for text in iter_chunks(long_text, chunk_size=2000, overlap=100):
    process(text)
```

## Chunk fields

`Chunk` is a `@dataclass` with the following fields:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The chunk content (stripped by default) |
| `index` | `int` | 0-based chunk number |
| `char_start` | `int` | Start offset in the original text |
| `char_end` | `int` | End offset (exclusive) |
| `is_last` | `bool` | True for the final chunk |

For `split_by_chars` (and therefore `split_by_tokens` / `iter_chunks`), the
offsets map directly back into the original text — i.e.
`text[chunk.char_start:chunk.char_end]` reproduces the unstripped chunk.

## API

### `split_by_chars(text, chunk_size, overlap=0, strip_chunks=True) -> list[Chunk]`

Fixed-size character chunks with optional overlap. The window advances by
`chunk_size - overlap` characters each step. Raises `ValueError` if
`chunk_size < 1`, `overlap < 0`, or `overlap >= chunk_size`. Returns `[]` for
empty input.

### `split_by_tokens(text, chunk_tokens, overlap_tokens=0, chars_per_token=4.0, strip_chunks=True) -> list[Chunk]`

Convenience wrapper around `split_by_chars` that sizes chunks by an
**approximate** token count using a characters-per-token ratio (default `4.0`,
a common rule of thumb for English text with OpenAI-style tokenizers). This is
a heuristic, not a real tokenizer — if you need exact token counts, size your
chunks with your model's tokenizer and call `split_by_chars` directly.

### `split_by_sentences(text, max_chunk_chars, overlap_sentences=0, sentence_sep=".") -> list[Chunk]`

Greedy sentence-aware splitting: sentences (detected by `sentence_sep`) are
packed into a chunk until the next one would exceed `max_chunk_chars`. A single
sentence longer than the limit is emitted whole rather than being cut. Use
`overlap_sentences` to repeat trailing sentences at the start of the next chunk
for better cross-chunk context.

### `iter_chunks(text, chunk_size, overlap=0) -> Iterator[str]`

Memory-friendly generator that yields only the chunk text (no `Chunk`
objects), for streaming large inputs.

## Development

The test suite uses only the standard library, so no extra dependencies are
required:

```bash
python -m unittest discover -s tests
```

## License

MIT
