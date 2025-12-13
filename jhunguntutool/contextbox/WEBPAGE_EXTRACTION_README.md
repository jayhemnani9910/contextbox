# ContextBox Web Page Extraction Module

A comprehensive web page content extraction module for the ContextBox framework that provides intelligent content extraction, metadata parsing, and integration with the ContextBox database.

## Features

### Core Capabilities

- **Intelligent Content Extraction**: Uses the readability algorithm to extract main content while filtering out navigation, ads, and other non-essential content
- **Robust HTML Parsing**: Built on BeautifulSoup4 for reliable HTML parsing and structure analysis
- **Content Analysis**: Provides content scoring, readability analysis, and language detection
- **Rate Limiting**: Respectful scraping with configurable rate limiting
- **Error Handling**: Comprehensive error handling with timeouts and retry mechanisms

### Content Extraction

- **Main Content Extraction**: Separates main article content from navigation, sidebar, and footer content
- **Title and Description**: Extracts page titles and meta descriptions
- **Open Graph Tags**: Parses Open Graph metadata for social media integration
- **Twitter Cards**: Extracts Twitter Card metadata
- **Structured Data**: Parses JSON-LD and microdata for enhanced content understanding

### Link and Media Extraction

- **Link Extraction**: Extracts all links with anchor text, metadata, and external/internal classification
- **Image Extraction**: Extracts images with alt text, dimensions, format, and lazy-loading information
- **Media Extraction**: Identifies videos, audio files, iframes, and embeds with provider detection
- **Social Media Links**: Special detection for social media platform links

### Content Quality

- **Text Cleaning**: Removes advertisements, navigation elements, and other noise content
- **Content Scoring**: Calculates content quality scores based on page structure and content ratio
- **Readability Analysis**: Provides basic readability scores using sentence/word complexity
- **Language Detection**: Simple language detection based on common word patterns

## Installation

The module requires the following dependencies (already included in ContextBox):

```bash
pip install requests beautifulsoup4 lxml readability-lxml
```

## Quick Start

### Basic Usage

```python
from contextbox.extractors import extract_webpage_content

# Extract content from a URL
content = extract_webpage_content("https://example.com")

print(f"Title: {content.title}")
print(f"Description: {content.description}")
print(f"Word Count: {content.word_count}")
print(f"Links Found: {len(content.links)}")
print(f"Images Found: {len(content.images)}")
```

### Advanced Usage

```python
from contextbox.extractors import WebPageExtractor

# Configure the extractor
config = {
    'timeout': 30,
    'rate_limit': 1.0,
    'extract_images': True,
    'extract_social_links': True,
    'max_retries': 3,
    'user_agent': 'MyBot/1.0'
}

extractor = WebPageExtractor(config)

# Extract content
content = extractor.extract("https://example.com")

# Access extracted data
print(f"Main Content: {content.cleaned_text[:200]}...")
print(f"Content Score: {content.content_score:.2f}")
print(f"Extraction Method: {content.extraction_method}")

# Iterate through links
for link in content.links:
    print(f"Link: {link.text} -> {link.url}")
    if link.is_external:
        print(f"  (External link)")

# Iterate through images
for image in content.images:
    print(f"Image: {image.src} (Alt: {image.alt})")
    if image.width and image.height:
        print(f"  Dimensions: {image.width}x{image.height}")
```

### Batch Extraction

```python
from contextbox.extractors import extract_multiple_pages

urls = [
    "https://example.com",
    "https://httpbin.org/html",
    "https://www.python.org"
]

contents = extract_multiple_pages(urls)

for content in contents:
    print(f"URL: {content.final_url}")
    print(f"Title: {content.title}")
    print(f"Status: {'Success' if not content.error else 'Error'}")
    print("---")
```

### ContextBox Integration

```python
from contextbox.extractors import extract_and_store_in_contextbox

# Extract and store directly in ContextBox database
capture_id = extract_and_store_in_contextbox(
    url="https://example.com",
    capture_data={'notes': 'Automated extraction'}
)

print(f"Stored in ContextBox with ID: {capture_id}")
```

## API Reference

### WebPageExtractor Class

The main extraction engine with configurable options.

#### Constructor

```python
WebPageExtractor(config: Optional[Dict[str, Any]] = None)
```

**Configuration Options:**

- `timeout`: Request timeout in seconds (default: 30)
- `max_retries`: Maximum retry attempts (default: 3)
- `user_agent`: Custom user agent string (default: ContextBox Web Extractor)
- `rate_limit`: Requests per second (default: 1.0)
- `burst_size`: Maximum burst requests before throttling (default: 3)
- `extract_images`: Whether to extract images (default: True)
- `extract_social_links`: Whether to extract social media links (default: True)
- `respect_robots_txt`: Whether to check robots.txt (default: False)
- `follow_redirects`: Whether to follow HTTP redirects (default: True)

#### Methods

- `extract(url: str) -> WebPageContent`: Extract content from a single URL
- `batch_extract(urls: List[str], max_concurrent: int = 5) -> List[WebPageContent]`: Extract from multiple URLs
- `save_extraction(content: WebPageContent, output_path: str)`: Save extraction to JSON file
- `get_stats() -> Dict[str, Any]`: Get extraction statistics
- `cleanup()`: Clean up resources

### WebPageContent Class

The result object containing all extracted information.

**Attributes:**

- `url`: Original URL
- `final_url`: Final URL after redirects
- `title`: Page title
- `description`: Meta description
- `main_content`: Raw extracted main content
- `cleaned_text`: Cleaned text content
- `content_score`: Content quality score (0.0 - 1.0)
- `extraction_method`: Method used for extraction
- `encoding`: Character encoding
- `meta_tags`: Dictionary of meta tags
- `open_graph`: Open Graph metadata
- `twitter_cards`: Twitter Card metadata
- `structured_data`: JSON-LD and microdata
- `word_count`: Number of words in content
- `character_count`: Number of characters
- `readability_score`: Readability score (0-100)
- `language`: Detected language
- `links`: List of ExtractedLink objects
- `images`: List of ExtractedImage objects
- `media`: List of ExtractedMedia objects
- `social_links`: List of social media links
- `response_time`: Request/response time in seconds
- `status_code`: HTTP status code
- `error`: Error message if extraction failed
- `content_hash`: MD5 hash of content for deduplication
- `extracted_at`: Timestamp of extraction

### Data Classes

#### ExtractedLink

Represents an extracted link with metadata.

```python
@dataclass
class ExtractedLink:
    url: str
    text: str
    title: Optional[str] = None
    rel: Optional[str] = None
    target: Optional[str] = None
    hreflang: Optional[str] = None
    type_attr: Optional[str] = None
    is_external: bool = False
    is_nofollow: bool = False
    is_ugc: bool = False
    confidence: float = 1.0
    position: Optional[Tuple[int, int]] = None
```

#### ExtractedImage

Represents an extracted image with metadata.

```python
@dataclass
class ExtractedImage:
    src: str
    alt: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    srcset: Optional[str] = None
    sizes: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    is_lazy_loaded: bool = False
    position: Optional[Tuple[int, int]] = None
```

#### ExtractedMedia

Represents extracted media elements.

```python
@dataclass
class ExtractedMedia:
    src: str
    type: str  # video, audio, iframe, embed
    title: Optional[str] = None
    poster: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[int] = None
    autoplay: bool = False
    controls: bool = False
    muted: bool = False
    loop: bool = False
    provider: Optional[str] = None
    thumbnail: Optional[str] = None
```

## ContextBox Database Integration

The module integrates seamlessly with the ContextBox database schema:

### Stored Artifacts

1. **webpage_content**: Main page content with full metadata
2. **webpage_link**: Individual links with context
3. **webpage_image**: Images with alt text and metadata

### Database Storage

```python
from contextbox.extractors import integrate_with_contextbox

capture_id = integrate_with_contextbox(extractor, capture_data, url)
```

The integration automatically:
- Creates appropriate database records
- Stores content artifacts with metadata
- Links to the ContextBox capture system
- Handles error cases gracefully

## Content Quality Analysis

### Content Scoring

The module calculates a content score (0.0 - 1.0) based on:

- Presence of advertisements (negative impact)
- Navigation element density (negative impact)
- Content-to-page ratio (positive impact)
- Link density (negative impact if excessive)

### Readability Analysis

Basic readability scoring using:
- Average sentence length
- Average syllables per word
- Flesch-like formula

### Language Detection

Simple language detection based on common word patterns. Can be enhanced with proper language detection libraries.

## Rate Limiting and Ethics

### Respectful Scraping

- Configurable rate limiting (default: 1 request/second)
- Burst handling to prevent overwhelming servers
- User-agent identification
- Timeout handling with exponential backoff

### Configuration for Production

```python
config = {
    'rate_limit': 0.5,  # 1 request every 2 seconds
    'timeout': 30,
    'max_retries': 3,
    'respect_robots_txt': True,
    'user_agent': 'MyApp/1.0 (+contact@example.com)'
}
```

## Error Handling

The module provides comprehensive error handling:

- **Network Errors**: Connection timeouts, DNS failures
- **HTTP Errors**: 404, 500, etc.
- **Content Errors**: Invalid HTML, encoding issues
- **Parsing Errors**: BeautifulSoup parsing failures

All errors are captured in the `error` field of WebPageContent objects.

## Performance Considerations

### Memory Usage

- Streams large content instead of loading entirely into memory
- Limits extracted elements to prevent database bloat
- Efficient BeautifulSoup parsing with lxml

### Database Integration

- Batch processing for multiple URLs
- Configurable limits for stored artifacts
- Automatic cleanup of temporary data

## Testing

Run the test suite:

```bash
python test_webpage_extractor.py
```

The test suite covers:
- Basic extraction functionality
- Batch processing
- Configuration options
- Content cleaning
- Database integration

## Examples

### Blog Post Extraction

```python
from contextbox.extractors import extract_webpage_content

# Extract blog post content
content = extract_webpage_content("https://blog.example.com/post/123")

print(f"Article: {content.title}")
print(f"Summary: {content.description}")
print(f"Content: {content.cleaned_text}")
print(f"Readability: {content.readability_score}")
```

### News Article Analysis

```python
from contextbox.extractors import WebPageExtractor

extractor = WebPageExtractor()
content = extractor.extract("https://news.example.com/article")

# Analyze structure
if content.content_score > 0.7:
    print("High-quality content detected")
else:
    print("Low-quality or navigation-heavy page")

# Check for social media presence
for social_link in content.social_links:
    print(f"Social presence: {social_link.url}")
```

### E-commerce Product Page

```python
# Extract product information
content = extract_webpage_content("https://shop.example.com/product/123")

# Look for images
for image in content.images:
    if image.width and image.height and image.width > 400:
        print(f"Product image: {image.src}")

# Extract structured data
if 'jsonld_0' in content.structured_data:
    product_data = content.structured_data['jsonld_0']
    print(f"Product: {product_data.get('name')}")
    print(f"Price: {product_data.get('offers', {}).get('price')}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Timeout Errors**: Increase timeout configuration
3. **Content Not Found**: Check if the page requires JavaScript
4. **Encoding Issues**: The module automatically detects encoding

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all extraction details will be logged
content = extract_webpage_content("https://example.com")
```

## Contributing

When contributing to the web page extraction module:

1. Add comprehensive tests for new features
2. Follow the existing code style
3. Document all new classes and methods
4. Ensure backward compatibility
5. Test with various website types

## License

This module is part of the ContextBox project and follows the same licensing terms.