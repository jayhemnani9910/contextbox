# ContextBox

ContextBox is a powerful tool for capturing and organizing digital context information from your computer activities. It helps you understand your digital behavior patterns and provides insights into your workflow.

## Features

- **Context Capture**: Automatically capture digital context from various sources
- **Text Extraction**: Extract and analyze text content from clipboard, active windows, and files
- **System Monitoring**: Track system activities, applications, and resource usage
- **Network Analysis**: Monitor network connections and online activities
- **Privacy Focused**: Local storage and privacy-first design
- **Extensible**: Modular architecture for custom extractors
- **CLI Interface**: Command-line tool for automation and scripting

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/contextbox/contextbox.git
cd contextbox
```

2. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

### Using pip

```bash
pip install contextbox
```

## Quick Start

### Command Line Interface

Start capturing context:
```bash
contextbox start
```

Stop capturing:
```bash
contextbox stop
```

Extract context from a data file:
```bash
contextbox extract data.json
```

Query stored context:
```bash
contextbox query <context_id>
```

### Python API

```python
from contextbox import ContextBox

# Initialize ContextBox
app = ContextBox()

# Start capturing
app.start_capture()

# Extract context from data
context_data = {"clipboard": "Hello World"}
extracted = app.extract_context(context_data)

# Store in database
context_id = app.store_context(extracted)
```

## Configuration

ContextBox can be configured using JSON or YAML configuration files.

### Example Configuration

```json
{
  "log_level": "INFO",
  "capture": {
    "interval": 1.0,
    "max_captures": 0,
    "enabled_sources": ["clipboard", "active_window", "recent_files"]
  },
  "database": {
    "path": "contextbox.db",
    "backup_enabled": true,
    "backup_interval": 3600
  },
  "extractors": {
    "enabled_extractors": ["text", "system", "network"],
    "confidence_threshold": 0.5
  }
}
```

### Loading Configuration

```bash
contextbox --config config.yml start
```

## Architecture

ContextBox is built with a modular architecture:

- **Capture Module**: Collects context data from various sources
- **Extraction Module**: Processes and analyzes captured data
- **Database Module**: Stores and retrieves context information
- **CLI Module**: Provides command-line interface
- **Utils Module**: Utility functions and helpers

## Data Sources

ContextBox can capture information from:

- **Clipboard**: Text content from system clipboard
- **Active Window**: Currently active application and window title
- **File Access**: Recently accessed files and file system activities
- **Network**: Network connections and online activities
- **System**: System information and resource usage

## Privacy and Security

- **Local Storage**: All data is stored locally on your device
- **No Cloud Sync**: No data is transmitted to external servers
- **Configurable Retention**: Set data retention policies
- **Encryption**: Optional encryption for stored data
- **Privacy Controls**: Granular control over what data is captured

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black contextbox/

# Type checking
mypy contextbox/
```

### Project Structure

```
contextbox/
├── contextbox/          # Main package
│   ├── __init__.py      # Package initialization
│   ├── main.py          # Main application class
│   ├── cli.py           # Command-line interface
│   ├── capture.py       # Context capture module
│   ├── database.py      # Database operations
│   ├── extractors.py    # Data extraction modules
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
├── README.md           # Documentation
└── install.sh          # Installation script
```

## Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://contextbox.readthedocs.io/](https://contextbox.readthedocs.io/)
- **Issues**: [https://github.com/contextbox/contextbox/issues](https://github.com/contextbox/contextbox/issues)
- **Discussions**: [https://github.com/contextbox/contextbox/discussions](https://github.com/contextbox/contextbox/discussions)

## Roadmap

- [ ] GUI interface
- [ ] Web dashboard
- [ ] Mobile companion app
- [ ] Advanced analytics
- [ ] Machine learning insights
- [ ] Export/import features
- [ ] Cloud synchronization (opt-in)
- [ ] Plugin system

---

**Note**: ContextBox is currently in alpha development. Features may change or be incomplete.