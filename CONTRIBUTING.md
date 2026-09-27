# Contributing

Thanks for helping. This is a small educational tool. Keep changes small, tested and focused.

## Set up

1. Fork the repo and clone your fork.
2. Install the dependencies, including the dev tools (pytest, ruff):
   ```bash
   uv sync --group dev
   ```
   Without uv, run `bash setup.sh` and then install `pytest` and `ruff` into `.venv`.

## Run the checks

```bash
uv run ruff check .
uv run pytest
```

- Unit tests must not use the network. Build price data in memory with `tests/conftest.py:make_ohlcv`.
- `python main.py run` calls Yahoo Finance. Use it for manual checks only, never in tests.

## Make a change

1. Create a branch from `main`, for example `fix/sector-cap` or `feat/new-watchlist`.
2. Write a failing test first when you change scoring, ranking or target logic.
3. Keep each PR to one purpose.
4. Add an entry under `[Unreleased]` in `CHANGELOG.md` for any change a user would notice.
5. Open a pull request against `main`. CI must pass.

## Rules

- **Code style:** ruff is the linter, and its config is in `pyproject.toml`. Match the surrounding code. The codebase aligns `=` in blocks and uses rich tables.
- **Type hints:** add them to every new function signature.
- **Table columns are locked.** Don't add, remove, rename or reorder columns in the stock tables or their CSVs without discussing it in an issue first. Put new data in the JSON output instead.
- **Option direction:** a bullish pick gets a CALL, and a bearish pick gets a PUT. Never mix them.
- **No personal data in commits:** `data/`, `output/`, `logs/` and `.env` are git-ignored. Keep them that way.
- **No financial-advice language:** output describes setups. It never tells anyone to buy or sell.
