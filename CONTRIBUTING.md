# Contributing

I try to reply to issues within a few days.

## Setup

```bash
git clone https://github.com/jayhemnani9910/contextbox.git
cd contextbox
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ocr]"
```

For `contextbox capture` to read text you also need Tesseract and a screenshot tool (`sudo apt install tesseract-ocr scrot` on Debian or Ubuntu).

## Tests and lint

```bash
pytest
ruff check --select E9,F63,F7,F82 contextbox tests
```

CI runs both on Python 3.9 and 3.12, and builds the docs site in `frontend/` (`npm ci && npm run build`).

The full ruff config in `pyproject.toml` still reports a lot of old style issues. Please don't add new ones in the lines you touch, and keep style-only cleanups in their own PR.

## Pull requests

- Branch off `master`, one change per PR.
- Add or update a test in `tests/` when you change behaviour.
- Don't run captures against a screen with private data when you make demo output or fixtures. Use sample text.

## Bug reports

Open an issue with the command you ran, what you expected, what happened, your OS, and the output of `contextbox --version`. Add `--log-level debug` output if you can.
