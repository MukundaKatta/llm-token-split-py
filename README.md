# llm-token-split-py

Split long documents into overlapping chunks for LLM context windows. Supports character-based, token-based, and sentence-aware splitting.

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

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The chunk content (stripped by default) |
| `index` | `int` | 0-based chunk number |
| `char_start` | `int` | Start offset in the original text |
| `char_end` | `int` | End offset (exclusive) |
| `is_last` | `bool` | True for the final chunk |

## License

MIT
