# Changelog

All notable changes to this project are written down here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
No version has been tagged yet.

## [Unreleased]

### Added

- MIT license, smoke tests, and a CI workflow (tests, lint, docs site build).
- Contributing guide, issue and PR templates, `.env.example`.
- Demo GIF and a rewritten README.

### Fixed (2026-10-07)

- `contextbox --help` no longer crashes on a clean install (watchdog was imported eagerly).
- `list`, `search` and `stats` read the database instead of printing mock rows. `export --all-contexts` exports stored contexts.
- `ask` and `summarize` say they are not implemented instead of printing canned answers.
- `python -m contextbox` calls the real entry point. `config --set` saves.
- Export and import: manifest, artifact ids remapped on import, backups in the managed folder.
- Database connections are closed and `LIKE` wildcards are escaped.
- Web page, Wikipedia and YouTube extractor crashes; the Wikipedia extractor matches the current parse API.
- Docs site sidebar works on phones, and the install command points at GitHub.

### Changed (2025-12-20)

- Flattened the package layout to `contextbox/` and moved packaging to `pyproject.toml`.
- Replaced the OpenAI and Ollama backends with a GitHub Models client.
- Added the React and Vite docs site, deployed to GitHub Pages.

### Added (2025-12-12)

- First version: screenshot capture, OCR, URL extraction, SQLite storage and a Click CLI.
