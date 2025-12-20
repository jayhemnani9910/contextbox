# ContextBox

AI-powered context capture and organization tool for developers.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-2ea44f)](https://jayhemnani9910.github.io/contextbox/)

## Features

- **Screenshot Capture** - Capture screenshots with automatic OCR text extraction
- **Web Content Extraction** - Extract content from web pages, Wikipedia, and YouTube
- **LLM-Powered Q&A** - Ask questions about your captured context using GitHub Models (free)
- **Intelligent Summarization** - Generate summaries of captured content
- **Semantic Search** - Search across all your captured contexts
- **Cross-Platform CLI** - Beautiful terminal interface with Rich formatting

## Installation

### Using pip

```bash
pip install contextbox
```

### From source

```bash
git clone https://github.com/jayhemnani9910/contextbox.git
cd contextbox
pip install -e ".[all]"
```

### Optional dependencies

```bash
# LLM features (GitHub Models)
pip install contextbox[llm]

# OCR support
pip install contextbox[ocr]

# YouTube extraction
pip install contextbox[youtube]

# Everything
pip install contextbox[all]
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

### 3. Ask questions (requires GitHub token)

```bash
export GITHUB_TOKEN="your_github_token"
contextbox ask "What was I working on?"
```

### 4. Generate summaries

```bash
contextbox summarize --all-contexts
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `contextbox capture` | Capture screenshot and extract context |
| `contextbox list` | List all stored contexts |
| `contextbox search <query>` | Search through contexts |
| `contextbox ask <question>` | Ask questions about context (AI) |
| `contextbox summarize` | Generate context summaries (AI) |
| `contextbox stats` | Show database statistics |
| `contextbox config` | Manage configuration |
| `contextbox export` | Export contexts to file |

### Capture Options

```bash
contextbox capture --help

Options:
  -o, --output PATH       Output file for results
  -a, --artifact-dir PATH Directory for artifacts (default: artifacts)
  --no-screenshot         Skip taking screenshot
  --extract-text          Extract text content
  --extract-urls          Extract URLs from content
```

## Configuration

### GitHub Token (for AI features)

Set your GitHub token to enable AI-powered features:

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

Or configure via CLI:

```bash
contextbox config --api-key
```

### Supported LLM Models (via GitHub Models)

| Model | Description |
|-------|-------------|
| `gpt-4o` | Most capable GPT-4 model |
| `gpt-4o-mini` | Fast and efficient (default) |
| `meta-llama-3.1-405b-instruct` | Meta's largest Llama |
| `meta-llama-3.1-70b-instruct` | Meta's Llama 70B |
| `mistral-large` | Mistral's flagship model |

## Python API

```python
from contextbox import ContextBox

# Initialize
app = ContextBox()

# Capture context
context = app.capture()
print(f"Captured: {context['context_id']}")

# Store and retrieve
context_id = app.store_context(context)
retrieved = app.get_context(context_id)

# Search
results = app.search("keyword")
```

### Using the LLM Backend

```python
from contextbox.llm import create_github_models_backend, ChatRequest

# Create backend
backend = create_github_models_backend()

async def ask_question():
    async with backend:
        request = ChatRequest.from_text(
            text="Summarize this context",
            model="gpt-4o-mini",
            provider="github_models"
        )
        response = await backend.chat_completion(request)
        print(response.content)
```

## Project Structure

```
contextbox/
├── contextbox/           # Python package
│   ├── cli.py           # CLI interface (Click + Rich)
│   ├── main.py          # Core ContextBox class
│   ├── capture.py       # Screen capture functionality
│   ├── database.py      # SQLite storage
│   ├── config.py        # Configuration management
│   ├── extractors/      # Content extractors
│   │   ├── webpage.py   # Web page extraction
│   │   ├── wikipedia.py # Wikipedia extraction
│   │   └── youtube.py   # YouTube extraction
│   └── llm/             # LLM integration
│       ├── github_models.py  # GitHub Models backend
│       ├── qa.py        # Q&A system
│       └── summarization.py  # Summarization
├── frontend/            # React + Vite documentation site
├── pyproject.toml       # Package configuration
└── README.md
```

## Requirements

- Python 3.9+
- For screenshots: platform-specific tools
  - macOS: Built-in `screencapture`
  - Linux: `scrot`, `gnome-screenshot`, or `flameshot`
  - Windows: `pyautogui`
- For OCR: Tesseract OCR engine

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/jayhemnani9910/contextbox.git
cd contextbox
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black contextbox/
ruff check contextbox/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://jayhemnani9910.github.io/contextbox)
- [GitHub Repository](https://github.com/jayhemnani9910/contextbox)
- [Issue Tracker](https://github.com/jayhemnani9910/contextbox/issues)
