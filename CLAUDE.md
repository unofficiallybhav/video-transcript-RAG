# video-transcript-rag

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
