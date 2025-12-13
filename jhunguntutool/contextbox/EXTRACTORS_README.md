# ContextBox Extractors

Enhanced text extraction and URL extraction functionality for the ContextBox application, featuring OCR capabilities and comprehensive URL pattern matching.

## Features

### 🔍 OCR Text Extraction
- **Tesseract OCR Integration**: Extract text from images using Tesseract OCR engine
- **PIL Image Processing**: Advanced image preprocessing for better OCR accuracy
- **Multiple Input Formats**: Support for file paths, bytes, base64 strings, and PIL Images
- **Confidence Scoring**: Word-level confidence scores and bounding box extraction
- **Image Enhancement**: Contrast enhancement, sharpening, and grayscale conversion

### 🌐 Advanced URL Extraction
- **Comprehensive Patterns**: Support for HTTP/HTTPS, FTP, file URLs, email addresses
- **Smart Domain Detection**: Domain inference from context and common TLD recognition
- **URL Normalization**: Automatic protocol addition and format standardization
- **Confidence Scoring**: Intelligent confidence calculation based on URL quality
- **Domain Inference**: Extract potential domains from text context

### 📋 Clipboard Integration
- **Primary Clipboard**: Direct clipboard text extraction when available
- **Fallback Mechanisms**: Environment variable scanning when clipboard is unavailable
- **URL Filtering**: Extract and filter URLs from clipboard content

### 🛠️ Text Processing
- **Text Cleaning**: Remove excessive whitespace and normalize formatting
- **Special Character Handling**: Optional preservation of special characters
- **Multiple Formats**: Support for various text input formats

## Installation Dependencies

### Required Dependencies (included in project)
- `pillow>=10.2.0` - Image processing capabilities
- `pytesseract` - OCR functionality (optional)

### Optional Dependencies
- `pyperclip` - Clipboard access (optional)
- `tesseract-ocr` - System-level OCR engine

```bash
# Install system OCR engine (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install optional Python dependencies
pip install pytesseract pyperclip
```

## Usage Examples

### Basic URL Extraction

```python
from contextbox.extractors import URLExtractor

# Extract URLs from text
text = "Visit https://example.com and check out github.com/user/repo"
urls = URLExtractor.extract_urls(text)

for url in urls:
    print(f"Found: {url.normalized_url} (confidence: {url.confidence:.2f})")
```

### OCR Text Extraction

```python
from contextbox.extractors import OCRExtractor

# Extract text from image
result = OCRExtractor.extract_text_from_image('screenshot.png')

print(f"OCR Text: {result.text}")
print(f"Confidence: {result.confidence:.2f}%")

# Extract with custom configuration
config = {
    'enhance_contrast': True,
    'sharpen': True,
    'grayscale': True,
    'psm': 6,  # Page segmentation mode
    'languages': 'eng+spa'  # Multiple languages
}

result = OCRExtractor.extract_text_from_image('image.png', config)
```

### Full Context Extraction

```python
from contextbox.extractors import EnhancedContextExtractor

# Initialize extractor
config = {
    'ocr': {
        'enhance_contrast': True,
        'sharpen': True,
        'grayscale': True
    },
    'enabled_extractors': ['text', 'system', 'network']
}

extractor = EnhancedContextExtractor(config)

# Extract from various data sources
data = {
    'text': 'Check out https://github.com and visit python.org',
    'image': 'screenshot.png',  # OCR will be performed
    'extract_clipboard_urls': True,  # Extract from clipboard
    'active_window': {
        'title': 'GitHub - Main Page',
        'application': 'Web Browser'
    }
}

result = extractor.extract(data)
print(f"Total URLs found: {result['metadata']['total_urls_found']}")
```

### Convenience Functions

```python
from contextbox.extractors import (
    extract_text_from_image,
    extract_urls_from_text,
    extract_clipboard_urls,
    process_screenshot_and_extract_urls
)

# Quick text extraction from image
text = extract_text_from_image('screenshot.png')

# Quick URL extraction from text
urls = extract_urls_from_text("Visit https://example.com")

# Quick clipboard URL extraction
clipboard_urls = extract_clipboard_urls()

# Combined OCR + URL extraction from screenshot
result = process_screenshot_and_extract_urls('screenshot.png')
print(f"Found {result['total_findings']} URLs in screenshot")
```

## Configuration Options

### OCR Configuration
```python
ocr_config = {
    'enhance_contrast': True,     # Increase image contrast
    'sharpen': True,              # Apply sharpening filter
    'grayscale': True,            # Convert to grayscale
    'min_image_size': 300,        # Minimum image dimension
    'psm': 3,                     # Page segmentation mode (0-13)
    'oem': 3,                     # OCR engine mode (0-3)
    'languages': 'eng',           # Recognition languages
    'extract_words': True         # Extract word-level data
}
```

### URL Extraction Configuration
```python
url_config = {
    'confidence_threshold': 0.5,  # Minimum confidence for extraction
    'extract_clipboard': True,    # Enable clipboard extraction
    'infer_domains': True        # Enable domain inference
}
```

## Data Structures

### ExtractedURL
```python
@dataclass
class ExtractedURL:
    url: str                    # Original URL text
    normalized_url: str         # Normalized URL
    domain: str                # Extracted domain
    is_direct: bool = True     # Direct URL or inferred
    confidence: float = 1.0    # Extraction confidence
    context: str = None        # Surrounding context
    position: tuple = None     # Position in text
```

### OCRResult
```python
@dataclass
class OCRResult:
    text: str                        # Extracted text
    confidence: float               # Average confidence score
    bbox: tuple = None              # Bounding box (x1, y1, x2, y2)
    words: list = None              # Word-level data
```

## Advanced Features

### Smart Domain Detection
- Recognizes common TLDs (.com, .org, .net, country codes, etc.)
- Filters out false positives (localhost, example.com, etc.)
- Infers domains from partial text matches
- Handles international domain names

### URL Pattern Recognition
- Standard HTTP/HTTPS URLs
- FTP servers
- File URLs
- Email addresses (converts to mailto:)
- IP addresses (IPv4 and IPv6)
- Protocol-relative URLs
- Domain-only references

### Text Processing
- Whitespace normalization
- Newline cleanup
- Special character handling
- Stop word filtering for keywords
- Entity extraction

### Error Handling
- Graceful degradation when dependencies are missing
- Comprehensive logging for debugging
- Fallback mechanisms for URL extraction
- Empty result handling

## Error Handling

The module includes comprehensive error handling:

```python
try:
    result = OCRExtractor.extract_text_from_image('image.png')
except ImportError as e:
    print(f"OCR dependencies not available: {e}")
except Exception as e:
    print(f"OCR extraction failed: {e}")

# Check availability
from contextbox.extractors import PIL_AVAILABLE, TESSERACT_AVAILABLE, CLIPBOARD_AVAILABLE
print(f"PIL: {PIL_AVAILABLE}, Tesseract: {TESSERACT_AVAILABLE}, Clipboard: {CLIPBOARD_AVAILABLE}")
```

## Integration with ContextBox

The enhanced extractor integrates seamlessly with the existing ContextBox architecture:

```python
from contextbox import ContextBox

# ContextBox automatically uses the enhanced extractor
contextbox = ContextBox({
    'extractors': {
        'ocr': {'enhance_contrast': True},
        'enabled_extractors': ['text', 'system', 'network']
    }
})

# Extract context (includes OCR and URL extraction)
result = contextbox.extract_context(data)
```

## Testing

Run the test suite to verify functionality:

```bash
cd /workspace/contextbox
python test_extractors.py
```

The test suite covers:
- URL extraction patterns
- Text processing functions
- OCR functionality (when available)
- Clipboard integration
- Full integration testing

## Performance Notes

- OCR processing time depends on image size and complexity
- URL extraction is highly optimized with regex caching
- Memory usage scales with image size for OCR operations
- Batch processing available for multiple images
- Configurable confidence thresholds for filtering

## Dependencies Status

The module gracefully handles missing dependencies:

| Feature | Required Package | Optional |
|---------|------------------|----------|
| Basic URL Extraction | None | - |
| OCR Text Extraction | `pillow`, `pytesseract` | ❌ |
| Clipboard Access | `pyperclip` | ✅ |
| System OCR Engine | `tesseract-ocr` | ❌ |

When optional dependencies are missing, the module provides appropriate warnings and fallback behavior without breaking functionality.

## 🚀 Smart Content Classifier

Intelligent content classification and routing system that automatically determines the best extraction method for any given URL.

### Key Features

#### ✅ URL Pattern Recognition
- Recognizes patterns for different content types (YouTube, Wikipedia, news sites, etc.)
- Supports multiple URL formats and variations
- Normalizes URLs for consistent processing

#### ✅ Domain-based Classification
- **YouTube URLs** → Video content
- **Wikipedia URLs** → Encyclopedia content  
- **News sites** → Article content
- **Documentation sites** → Technical content
- **Social media URLs** → Post content
- **Generic URLs** → Default extraction

#### ✅ Content Type Detection
- Video content
- Encyclopedia/reference content
- News articles
- Technical documentation
- Social media posts
- Generic web pages

#### ✅ Automatic Extractor Selection and Routing
- Routes URLs to appropriate extractors based on classification
- Selects best extraction method automatically
- Provides fallback mechanisms for failed extractions

#### ✅ Confidence Scoring
- Scores classification accuracy from 0.0 to 1.0
- Higher confidence for well-matched patterns
- Transparent confidence reporting

#### ✅ Fallback Strategies
- Multiple extractor fallback chains
- Graceful degradation when primary methods fail
- Comprehensive error handling

#### ✅ Custom Domain Rules
- Support for adding custom classification rules
- Priority-based rule matching
- Configurable confidence boosts

#### ✅ Performance Optimization
- Intelligent caching to avoid redundant requests
- Configurable cache size and TTL
- Batch processing capabilities

#### ✅ Database Integration
- Stores classification results in ContextBox database
- Metadata tracking for analytics
- Integration with existing artifacts system

### Quick Start

#### Basic URL Classification

```python
from contextbox.extractors.classifier import classify_url

# Classify a single URL
result = classify_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
print(f"Content Type: {result.content_type}")  # Output: video
print(f"Extractor: {result.extractor_name}")   # Output: youtube
print(f"Confidence: {result.confidence}")      # Output: 1.0
```

#### Content Extraction

```python
from contextbox.extractors.classifier import extract_url_content

# Classify and extract content in one step
result = extract_url_content("https://developer.mozilla.org/docs/Web/API")
print(f"Success: {result['success']}")
print(f"Content Type: {result['content_type']}")
print(f"Metadata: {result['metadata']}")
```

#### Batch Processing

```python
from contextbox.extractors.classifier import batch_extract_urls

urls = [
    "https://www.youtube.com/watch?v=test1",
    "https://en.wikipedia.org/wiki/Test",
    "https://www.bbc.com/news",
    "https://docs.python.org"
]

results = batch_extract_urls(urls)
for result in results:
    if result['success']:
        print(f"{result['url']} -> {result['content_type']}")
```

### Advanced Usage

#### Custom Classification Rules

```python
from contextbox.extractors.classifier import SmartContentClassifier, ContentRule

classifier = SmartContentClassifier()

# Create custom rule for blog posts
custom_rule = ContentRule(
    name='blog_rule',
    domain_patterns=['myblog.com', 'techblog.net'],
    url_patterns=[r'.*/article/.*', r'.*/post/.*'],
    content_type='blog_post',
    domain_type='blog',
    extractor_name='generic',
    priority=40,
    confidence_boost=0.1
)

classifier.add_custom_rule(custom_rule)
```

#### Performance Configuration

```python
config = {
    'use_database': True,           # Enable database storage
    'cache_size_limit': 1000,       # Maximum cached results
    'cache_ttl': 3600,             # Cache TTL in seconds (1 hour)
    'db_config': {                 # Database configuration
        'db_path': 'contextbox.db',
        'timeout': 30.0
    }
}

classifier = SmartContentClassifier(config)
```

### Classification Results

The classifier returns comprehensive `ClassificationResult` objects:

```python
{
    'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'content_type': 'video',
    'domain_type': 'video_platform',
    'extractor_name': 'youtube',
    'confidence': 1.0,
    'metadata': {
        'platform': 'youtube',
        'video_id': 'dQw4w9WgXcQ',
        'extraction_type': 'video'
    },
    'fallback_extractors': ['generic'],
    'routing_info': {
        'rule_applied': 'youtube_video',
        'extractor_selected': 'youtube',
        'confidence_score': 1.0
    },
    'classification_timestamp': '2023-01-01T12:00:00.000000',
    'processing_time': 0.002345
}
```

### Database Integration

Classification results are automatically stored in ContextBox's database as artifacts:

```python
# Results are stored in the artifacts table
# Capture the results in your ContextBox workflow
capture = contextbox.capture()
for artifact in capture.get('artifacts', []):
    if artifact.get('kind') == 'url_classification':
        classification = json.loads(artifact['text'])
        print(f"Classified: {classification['url']} -> {classification['content_type']}")
```

### Testing

Run the comprehensive test suite:

```bash
python test_classifier.py
```

The test suite covers:
- ✅ Basic URL classification (17+ test URLs)
- ✅ Content extraction
- ✅ Batch processing
- ✅ Custom rules
- ✅ Performance and caching
- ✅ Error handling
- ✅ Database integration

### Dependencies

The Smart Content Classifier has no additional dependencies beyond the core ContextBox requirements:

| Feature | Required Package | Optional |
|---------|------------------|----------|
| Basic URL Classification | None | - |
| OCR Text Extraction | `pillow`, `pytesseract` | ❌ |
| Clipboard Access | `pyperclip` | ✅ |
| System OCR Engine | `tesseract-ocr` | ❌ |
| **Smart Classification** | **None** | **-** |

The classifier gracefully handles missing dependencies and provides appropriate warnings without breaking functionality.