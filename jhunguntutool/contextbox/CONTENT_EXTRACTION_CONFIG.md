# ContextBox Content Extraction Configuration Guide

This guide provides comprehensive documentation for configuring and using the integrated content extraction capabilities in ContextBox.

## Table of Contents

1. [Overview](#overview)
2. [Installation and Dependencies](#installation-and-dependencies)
3. [Basic Configuration](#basic-configuration)
4. [Advanced Configuration](#advanced-configuration)
5. [Content Extraction Options](#content-extraction-options)
6. [OCR Configuration](#ocr-configuration)
7. [URL Extraction Configuration](#url-extraction-configuration)
8. [Database Configuration](#database-configuration)
9. [CLI Configuration](#cli-configuration)
10. [Error Handling and Logging](#error-handling-and-logging)
11. [Performance Tuning](#performance-tuning)
12. [Best Practices](#best-practices)

## Overview

ContextBox now includes comprehensive content extraction capabilities that can:

- **Automatically extract content** when URLs are found during context capture
- **Manually extract content** via CLI and API interfaces
- **Process images** using OCR technology for text extraction
- **Extract and analyze URLs** with confidence scoring and domain classification
- **Store results** in the database with proper relationships and metadata
- **Handle errors gracefully** with fallback mechanisms

### Key Components

1. **ContentExtractor**: Main orchestrator for all extraction operations
2. **Enhanced Extractors**: OCR, URL extraction, text processing modules
3. **Database Integration**: Storage and retrieval of extraction results
4. **CLI Interface**: Command-line tools for manual extraction
5. **Configuration System**: Flexible configuration management

## Installation and Dependencies

### Required Dependencies

```bash
# Core dependencies (included in requirements.txt)
pip install pillow>=10.2.0
```

### Optional Dependencies

```bash
# OCR functionality
pip install pytesseract
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# or
brew install tesseract              # macOS

# Clipboard access
pip install pyperclip
```

### System Requirements

- **Python**: 3.7+
- **Operating System**: Windows, macOS, Linux
- **Memory**: 512MB+ recommended for OCR operations
- **Storage**: Space for database and temporary files

### Dependency Status

You can check available dependencies programmatically:

```python
from contextbox.extractors import PIL_AVAILABLE, TESSERACT_AVAILABLE, CLIPBOARD_AVAILABLE

print(f"PIL (Image Processing): {PIL_AVAILABLE}")
print(f"Tesseract (OCR): {TESSERACT_AVAILABLE}")
print(f"Clipboard Access: {CLIPBOARD_AVAILABLE}")
```

## Basic Configuration

### Enable Content Extraction

```python
from contextbox import ContextBox

# Enable content extraction with defaults
config = {
    'enable_content_extraction': True,
    'content_extraction': {
        'auto_extract': True,      # Automatic extraction during capture
        'store_in_database': True, # Store results in database
        'output_format': 'json'    # Default output format
    }
}

contextbox = ContextBox(config)
```

### Minimal Configuration

```python
# Minimal working configuration
config = {
    'enable_content_extraction': True
}

contextbox = ContextBox(config)
# Uses all defaults: auto_extract=True, store_in_database=True, output_format='json'
```

### Disable Content Extraction

```python
# Completely disable content extraction
config = {
    'enable_content_extraction': False
}

contextbox = ContextBox(config)
# Falls back to basic extraction only
```

## Advanced Configuration

### Complete Configuration Example

```python
config = {
    'enable_content_extraction': True,
    'content_extraction': {
        # Core settings
        'auto_extract': True,
        'output_format': 'json',  # json, pretty, summary, detailed
        'store_in_database': True,
        
        # Enabled extractors
        'enabled_extractors': [
            'text_extraction',     # Text analysis and keyword extraction
            'system_extraction',   # System information analysis
            'network_extraction'   # Network-related context
        ],
        
        # Extractor-specific configuration
        'extractors': {
            # OCR configuration
            'ocr': {
                'enhance_contrast': True,      # Increase image contrast
                'sharpen': True,               # Apply sharpening filter
                'grayscale': True,             # Convert to grayscale
                'min_image_size': 300,         # Minimum image dimension
                'psm': 6,                      # Page segmentation mode (0-13)
                'oem': 3,                      # OCR engine mode (0-3)
                'languages': 'eng',            # Recognition languages
                'extract_words': True          # Extract word-level data
            },
            
            # URL extraction configuration
            'confidence_threshold': 0.5,      # Minimum confidence for extraction
            'extract_clipboard': True,        # Enable clipboard extraction
            'infer_domains': True            # Enable domain inference
        }
    },
    
    # Database configuration
    'database': {
        'db_path': 'contextbox.db',
        'timeout': 30.0,
        'backup_interval': 3600  # Backup every hour (seconds)
    },
    
    # Logging configuration
    'log_level': 'INFO'  # DEBUG, INFO, WARNING, ERROR
}

contextbox = ContextBox(config)
```

## Content Extraction Options

### Automatic vs Manual Extraction

#### Automatic Extraction

Automatic extraction is triggered when URLs are found during context capture:

```python
config = {
    'enable_content_extraction': True,
    'content_extraction': {
        'auto_extract': True  # Enable automatic extraction
    }
}

# When capture contains URLs, extraction happens automatically
capture_data = {
    'text': 'Visit https://example.com for more info',
    'active_window': {'title': 'Browser Window'}
}

result = contextbox.extract_content_from_capture(capture_data)
# Automatically extracts and processes URLs found in the data
```

#### Manual Extraction

Manual extraction provides full control over the extraction process:

```python
# Manual extraction with specific options
result = contextbox.extract_content_manually(
    input_data={'text': 'Visit https://example.com'},
    extract_urls=True,        # Extract URLs
    extract_text=True,        # Process text content
    extract_images=True,      # Perform OCR on images
    output_format='pretty'    # Output formatting
)
```

### Output Formats

#### JSON Format (Default)

```json
{
    "type": "manual_content_extraction",
    "timestamp": "2023-01-01T12:00:00",
    "extraction_id": "abc123",
    "extracted_content": {
        "urls": {
            "direct_urls": [...],
            "total_found": 5
        },
        "text": {
            "word_count": 150,
            "processed_text": "..."
        }
    }
}
```

#### Pretty Format

```
============================================================
CONTENT EXTRACTION RESULTS
============================================================
Type: manual_content_extraction
Timestamp: 2023-01-01T12:00:00
Extraction ID: abc123

--- SUMMARY ---
urls: 5 URLs found
text: 150 words processed

--- URL ANALYSIS ---
Total URLs: 5
High confidence URLs: 3
  example.com: 2
  github.com: 1
============================================================
```

#### Summary Format

```
Extraction: 5 URLs, 3 successful
```

#### Detailed Format

Same as JSON but with full structure and metadata.

## OCR Configuration

### Basic OCR Settings

```python
'extractors': {
    'ocr': {
        'enhance_contrast': True,     # Improve contrast for better recognition
        'sharpen': True,              # Apply sharpening filter
        'grayscale': True,            # Convert to grayscale (recommended)
        'min_image_size': 300         # Minimum image dimension
    }
}
```

### Advanced OCR Settings

```python
'extractors': {
    'ocr': {
        # Image preprocessing
        'enhance_contrast': True,
        'sharpen': True,
        'grayscale': True,
        'min_image_size': 300,
        
        # Tesseract configuration
        'psm': 6,                     # Page segmentation mode
        'oem': 3,                     # OCR engine mode
        'languages': 'eng+spa',       # Multiple languages
        
        # Output options
        'extract_words': True,        # Extract word-level data
        'extract_bbox': True          # Extract bounding boxes
    }
}
```

### Page Segmentation Modes (PSM)

- **0**: Orientation and script detection (OSD) only
- **1**: Automatic page segmentation with OSD
- **2**: Automatic page segmentation, but no OSD
- **3**: Fully automatic page segmentation, no OSD
- **4**: Assume a single column of text of variable sizes
- **5**: Assume a single uniform block of vertically aligned text
- **6**: Assume a single uniform block of text
- **7**: Treat the image as a single text line
- **8**: Treat the image as a single word
- **9**: Treat the image as a single word in a circle
- **10**: Treat the image as a single character
- **11**: Sparse text
- **12**: Sparse text with OSD
- **13**: Raw line

### OCR Engine Modes (OEM)

- **0**: Legacy engine only
- **1**: Neural nets LSTM engine only
- **2**: Legacy + LSTM engines
- **3**: Default, based on what is available

### Language Support

```python
'languages': 'eng'        # English only
'languages': 'eng+spa'    # English and Spanish
'languages': 'eng+fra+deu' # English, French, and German
```

Available languages depend on installed Tesseract language packs.

## URL Extraction Configuration

### Basic URL Extraction

```python
'extractors': {
    'confidence_threshold': 0.5,     # Minimum confidence for inclusion
    'extract_clipboard': True,       # Check clipboard for URLs
    'infer_domains': True           # Infer domains from context
}
```

### Advanced URL Analysis

```python
'extractors': {
    # Extraction thresholds
    'confidence_threshold': 0.7,     # Higher threshold for better quality
    
    # Clipboard integration
    'extract_clipboard': True,       # Enable clipboard scanning
    
    # Domain inference
    'infer_domains': True,           # Enable domain inference
    'common_tlds_only': False,       # Allow uncommon TLDs
    
    # Pattern matching
    'strict_parsing': False,         # Use strict URL parsing
    'allow_query_params': True       # Include URL query parameters
}
```

### URL Pattern Support

The URL extractor supports:

- **Standard URLs**: `http://`, `https://`, `ftp://`
- **Protocol-relative**: `//example.com`
- **Domain-only**: `example.com`, `sub.domain.com`
- **IP addresses**: IPv4 and IPv6
- **Email addresses**: `user@example.com` (converted to `mailto:`)
- **File URLs**: `file:///path/to/file`
- **Port numbers**: `example.com:8080`

### Domain Classification

URLs are automatically classified into categories:

- **Social Media**: twitter.com, facebook.com, linkedin.com
- **Search Engines**: google.com, bing.com, yahoo.com
- **Developer Tools**: github.com, gitlab.com, stackoverflow.com
- **News Sites**: Sites with 'news', 'times', 'post' in domain
- **E-commerce**: Sites with 'shop', 'store', 'amazon'
- **General**: All other domains

## Database Configuration

### Basic Database Settings

```python
'database': {
    'db_path': 'contextbox.db',      # Database file path
    'timeout': 30.0                  # Connection timeout (seconds)
}
```

### Advanced Database Settings

```python
'database': {
    'db_path': 'contextbox.db',
    'timeout': 30.0,
    'backup_interval': 3600,         # Auto-backup interval (seconds)
    'auto_vacuum': True,             # Auto-vacuum database
    'synchronous': 'NORMAL',         # Synchronization mode
    'cache_size': 1000               # Cache size in pages
}
```

### Database Schema

The database uses the following tables:

#### captures
- `id`: Primary key
- `created_at`: Timestamp
- `source_window`: Source window title
- `screenshot_path`: Screenshot file path
- `clipboard_text`: Clipboard content
- `notes`: Additional notes

#### artifacts
- `id`: Primary key
- `capture_id`: Foreign key to captures
- `kind`: Artifact type (url, text, ocr_text, etc.)
- `url`: URL if applicable
- `title`: Title or description
- `text`: Text content
- `metadata_json`: Metadata as JSON

### Database Operations

```python
# Store extraction result
success = database.store_extraction_result(capture_id, extraction_result)

# Retrieve extraction results
results = database.get_extraction_results(capture_id)

# Search artifacts
search_results = database.search_extraction_artifacts('python', artifact_kind='url')

# Get statistics
stats = database.get_stats()
```

## CLI Configuration

### Extract Content Command

```bash
# Basic usage
contextbox extract-content input.txt

# With output file
contextbox extract-content input.txt --output results.json

# Different input types
contextbox extract-content image.png --type image
contextbox extract-content data.json --type json

# Output formats
contextbox extract-content input.txt --format pretty
contextbox extract-content input.txt --format summary

# Selective extraction
contextbox extract-content input.txt --no-extract-urls
contextbox extract-content input.txt --extract-text --extract-images

# With custom configuration
contextbox extract-content input.txt --config-extraction config.json
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_file` | Path to input file | Required |
| `--type, -t` | Input type (auto, text, image, json) | auto |
| `--output, -o` | Output file path | None |
| `--format, -f` | Output format (json, pretty, summary, detailed) | json |
| `--extract-urls` | Extract URLs | True |
| `--extract-text` | Extract text content | True |
| `--extract-images` | Extract images (OCR) | True |
| `--config-extraction` | Configuration file path | None |

### Configuration File Format

Create `extraction_config.json`:

```json
{
    "auto_extract": true,
    "output_format": "pretty",
    "store_in_database": false,
    "enabled_extractors": ["text_extraction"],
    "extractors": {
        "ocr": {
            "enhance_contrast": true,
            "psm": 6,
            "languages": "eng"
        },
        "confidence_threshold": 0.7
    }
}
```

## Error Handling and Logging

### Error Handling Strategy

1. **Graceful Degradation**: Missing OCR → skip image processing
2. **Fallback Mechanisms**: Content extractor fails → basic extraction
3. **Validation**: Input validation before processing
4. **Recovery**: Retry mechanisms for transient failures

### Logging Configuration

```python
import logging

# Set global log level
config = {
    'log_level': 'INFO'  # DEBUG, INFO, WARNING, ERROR
}

# Configure specific loggers
logging.getLogger('contextbox.extractors').setLevel(logging.DEBUG)
logging.getLogger('contextbox.database').setLevel(logging.WARNING)
```

### Error Types and Handling

#### ContentExtractionError
- Raised for content extraction failures
- Includes detailed error information
- Allows graceful fallback

#### DatabaseError
- Raised for database operation failures
- Includes operation details
- Supports rollback operations

#### ImportError
- Raised when optional dependencies are missing
- Indicates feature availability
- Allows conditional feature usage

### Example Error Handling

```python
try:
    result = contextbox.extract_content_manually(input_data)
except ContentExtractionError as e:
    print(f"Content extraction failed: {e}")
    # Fallback to basic extraction
    result = contextbox.extract_context(input_data)
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and handle gracefully
    logging.error(f"Extraction error: {e}", exc_info=True)
```

## Performance Tuning

### Memory Optimization

```python
# Limit image processing memory usage
'extractors': {
    'ocr': {
        'max_image_size': 1920,      # Maximum image dimension
        'jpeg_quality': 85,          # JPEG compression quality
        'grayscale': True            # Reduce memory usage
    }
}
```

### Processing Optimization

```python
# Optimize for speed vs accuracy
'extractors': {
    'confidence_threshold': 0.8,     # Higher threshold = faster filtering
    'psm': 6,                        # Faster page segmentation
    'max_text_length': 10000        # Limit text processing length
}
```

### Database Optimization

```python
'database': {
    'cache_size': 2000,              # Increase cache for better performance
    'synchronous': 'OFF',            # Faster writes (less safe)
    'auto_vacuum': 'FULL'           # Periodic cleanup
}
```

### Batch Processing

```python
# Process multiple files efficiently
for filename in file_list:
    result = contextbox.extract_content_manually(
        input_data={'file_content': read_file(filename)},
        extract_urls=True,
        extract_text=True,
        extract_images=False  # Skip OCR for speed
    )
```

## Best Practices

### Configuration Best Practices

1. **Start Simple**: Begin with basic configuration and add features gradually
2. **Feature Detection**: Check available features before enabling advanced options
3. **Memory Management**: Set appropriate limits for large-scale processing
4. **Error Handling**: Always implement fallback mechanisms
5. **Testing**: Test configuration changes in development environment

### Example Production Configuration

```python
production_config = {
    'enable_content_extraction': True,
    'content_extraction': {
        'auto_extract': True,
        'output_format': 'json',
        'store_in_database': True,
        'enabled_extractors': [
            'text_extraction',
            'system_extraction'
            # Remove network_extraction if not needed
        ],
        'extractors': {
            'ocr': {
                'enhance_contrast': True,
                'psm': 6,  # Balanced accuracy/speed
                'languages': 'eng',  # Only needed languages
                'extract_words': False  # Disable for speed
            },
            'confidence_threshold': 0.7,
            'max_image_size': 1920
        }
    },
    'database': {
        'db_path': '/var/lib/contextbox/contextbox.db',
        'timeout': 30.0,
        'backup_interval': 3600
    },
    'log_level': 'WARNING'  # Reduce log noise in production
}

contextbox = ContextBox(production_config)
```

### Development Configuration

```python
development_config = {
    'enable_content_extraction': True,
    'content_extraction': {
        'auto_extract': True,
        'output_format': 'pretty',
        'store_in_database': True,
        'extractors': {
            'ocr': {
                'enhance_contrast': True,
                'sharpen': True,
                'extract_words': True,  # Enable for debugging
                'extract_bbox': True    # Enable for debugging
            },
            'confidence_threshold': 0.5  # Lower threshold for more results
        }
    },
    'database': {
        'db_path': 'contextbox_dev.db',
        'timeout': 10.0
    },
    'log_level': 'DEBUG'  # Detailed logging for development
}
```

### Resource Usage Guidelines

- **OCR Processing**: Expect 1-3 seconds per image depending on size and complexity
- **Memory Usage**: 50-200MB for large-scale OCR operations
- **Database Growth**: ~1KB per extraction artifact
- **Disk Space**: Plan for 10x the size of processed images for temporary files

### Security Considerations

1. **Input Validation**: Always validate input data before processing
2. **File Paths**: Sanitize file paths to prevent directory traversal
3. **Database Access**: Use proper file permissions for database files
4. **Logging**: Avoid logging sensitive information
5. **Configuration**: Keep configuration files secure and version-controlled

### Monitoring and Maintenance

1. **Log Monitoring**: Monitor extraction success rates and errors
2. **Database Maintenance**: Regular backups and vacuum operations
3. **Performance Monitoring**: Track processing times and memory usage
4. **Dependency Updates**: Keep OCR and image processing libraries updated
5. **Storage Management**: Monitor disk space usage for temporary files

This configuration guide provides comprehensive documentation for setting up and optimizing the content extraction capabilities in ContextBox. Start with basic configuration and gradually enable advanced features based on your specific requirements and available resources.
