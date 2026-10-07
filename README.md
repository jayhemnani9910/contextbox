<div align="center">

# ContextBox

**Capture and organize your working context from the command line**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages](https://img.shields.io/badge/docs-live-2ea44f)](https://jayhemnani9910.github.io/contextbox/)

[Documentation](https://jayhemnani9910.github.io/contextbox/) • [Installation](#installation) • [Quick Start](#quick-start)

</div>

---

## What is ContextBox?

ContextBox captures, organizes, and queries your working context. Take screenshots, extract web content, and ask AI questions about everything you've captured.

**Key Features:**

| Feature | Description |
|---------|-------------|
| **Screenshot Capture** | Capture screen with automatic OCR text extraction |
| **Web Extraction** | Pull content from web pages, Wikipedia, YouTube |
| **LLM backend** | GitHub Models client for the Python API (`contextbox ask` is not implemented yet) |
| **Search** | Text search across captured contexts |
| **Cross-Platform** | Works on macOS, Linux, Windows |

## Installation

### Using pip

Not on PyPI yet, so install from GitHub:

```bash
pip install git+https://github.com/jayhemnani9910/contextbox.git
```

### From source

```bash
git clone https://github.com/jayhemnani9910/contextbox.git
cd contextbox
pip install -e ".[all]"
```

### Optional dependencies

Run these from the cloned source folder:

```bash
# LLM features (GitHub Models)
pip install -e ".[llm]"

# OCR support
pip install -e ".[ocr]"

# YouTube extraction
pip install -e ".[youtube]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### 1. Capture your screen

```bash
contextbox capture
```

This takes a screenshot, extracts text via OCR, and stores it in the local database.

### 2. List captured contexts

```bash
contextbox list
```

### 3. Ask questions (not implemented yet in the CLI)

```bash
export GITHUB_TOKEN="your_github_token"
contextbox ask "What was I working on?"
```

### 4. Generate summaries (not implemented yet in the CLI)

```bash
contextbox summarize --all-contexts
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `contextbox capture` | Capture screenshot and extract context |
| `contextbox list` | List all stored contexts |
| `contextbox search <query>` | Search through contexts |
| `contextbox ask <question>` | Ask questions about context (AI), not implemented yet in the CLI |
| `contextbox summarize` | Generate context summaries (AI), not implemented yet in the CLI |
| `contextbox stats` | Show database statistics |
| `contextbox export` | Export contexts to file |

## Configuration

### GitHub Token (for AI features)

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

### Supported AI Models

All models are free via [GitHub Models](https://github.com/marketplace/models):

| Model | Best For |
|-------|----------|
| `gpt-4o-mini` | Fast responses (default) |
| `gpt-4o` | Complex reasoning |
| `meta-llama-3.1-70b-instruct` | Open-source alternative |

## Python API

```python
from contextbox import ContextBox

app = ContextBox()

# Capture and store
context_id = app.capture_context({"active_window": {"title": "Terminal"}, "clipboard": "copied text"})

# Search and retrieve
results = app.database.search_contexts("keyword")
context = app.get_context(context_id)
```

## Requirements

| Platform | Screenshot Tool | OCR |
|----------|----------------|-----|
| macOS | Built-in | Tesseract |
| Linux | scrot / flameshot | Tesseract |
| Windows | pyautogui | Tesseract |

## License

MIT License

---

<div align="center">

[Documentation](https://jayhemnani9910.github.io/contextbox/) • [Issues](https://github.com/jayhemnani9910/contextbox/issues) • [Contribute](https://github.com/jayhemnani9910/contextbox)

</div>
