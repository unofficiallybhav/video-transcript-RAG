# video-transcript-rag

## The corpus

`corpus.json` lists the 83 ingested videos as `{stem: {source, course}}`, and `cache/`
holds one Whisper transcript per stem. Together they are the corpus: `load_corpus()`,
`load_chunks()` and `video_names()` read them, so every offline stage works without the
media files. The videos themselves are gone and are not required.

Rebuilding the chunks from `cache/` reproduces the 406 chunks and the chunk ids already
in qdrant, because `corpus.json` is ordered by course then filename — the order the
original walk of the media directory produced. Keep that ordering if you edit it.

To ingest new media, put it somewhere and pass that path: `ingest_all(root)`.

## Tests

`uv run pytest` runs the whole suite, from `tests/`. It needs no network access and no
API credentials, so it still runs once the daily token budget is gone. pytest is a dev
dependency; add dependencies with `uv add`, never `pip`.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
