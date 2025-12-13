# Wikipedia Extraction Module - Implementation Complete

## Overview

The Wikipedia extraction module has been successfully implemented in `/workspace/contextbox/contextbox/extractors/wikipedia.py`. This module provides comprehensive Wikipedia content extraction using the MediaWiki API.

## Features Implemented

### 1. MediaWiki API Integration ✅
- **Robust API requests** with configurable timeout and retry logic
- **Multi-language support** for 20+ Wikipedia language variants
- **Error handling** for network issues, API rate limits, and invalid responses
- **Session management** with proper user agents and headers

### 2. URL Parsing ✅
Supports multiple Wikipedia URL formats:
- `https://en.wikipedia.org/wiki/Article_Title`
- `https://en.wikipedia.org/w/index.php?title=Article_Title&printable=yes`
- `https://de.wikipedia.org/wiki/Artikel_Titel` (other languages)
- Automatic language detection and title extraction
- URL decoding and title normalization

### 3. Content Cleaning and Text Extraction ✅
- **HTML parsing** with BeautifulSoup for clean text extraction
- **Template removal** (infoboxes, navigation, references)
- **Citation cleanup** (removes [edit], [citation needed], etc.)
- **Lead section extraction** (introductory text)
- **Main content extraction** (full article body)

### 4. Summary Generation ✅
- **Automatic summary creation** from lead section or main text
- **Configurable length limits** (default: 500 characters)
- **Sentence-based summarization** (takes first 3 sentences)
- **Fallback mechanisms** when lead section unavailable

### 5. Link Extraction ✅
- **Internal Wikipedia links** extraction
- **Link count statistics** and limitations
- **Category extraction** from page metadata
- **Section structure** analysis

### 6. Image and Media Information ✅
- **Image metadata extraction** from Wikipedia API
- **Image URLs and descriptions** 
- **Media file references** with wiki URLs
- **Configurable extraction** (can be disabled)

### 7. Reference and Citation Handling ✅
- **Reference count tracking**
- **Reference text extraction** (limited to first 50)
- **Citation metadata** parsing
- **Reference index tracking**

### 8. Multi-language Wikipedia Support ✅
Supports major Wikipedia language variants:
- English (en.wikipedia.org)
- German (de.wikipedia.org) 
- French (fr.wikipedia.org)
- Spanish (es.wikipedia.org)
- Italian (it.wikipedia.org)
- Portuguese (pt.wikipedia.org)
- Russian (ru.wikipedia.org)
- Japanese (ja.wikipedia.org)
- Chinese (zh.wikipedia.org)
- Arabic (ar.wikipedia.org)
- Dutch (nl.wikipedia.org)
- Polish (pl.wikipedia.org)
- Swedish (sv.wikipedia.org)
- Czech (cs.wikipedia.org)
- Finnish (fi.wikipedia.org)
- Hungarian (hu.wikipedia.org)
- Korean (ko.wikipedia.org)
- Norwegian (no.wikipedia.org)
- Turkish (tr.wikipedia.org)
- Thai (th.wikipedia.org)
- Vietnamese (vi.wikipedia.org)

### 9. Error Handling ✅
- **Non-existent pages** detection
- **Redirect handling** support
- **API error responses** management
- **Network timeout** handling with retries
- **Invalid URL** validation
- **Graceful fallbacks** for all failure modes

### 10. ContextBox Database Integration ✅
- **Artifact creation** functions
- **Metadata storage** with rich information
- **Summary artifacts** separate from main content
- **Database schema compatibility** with existing structure
- **Capture linking** support

## Usage Examples

### Basic Usage
```python
from contextbox.extractors.wikipedia import WikipediaExtractor

# Initialize extractor
extractor = WikipediaExtractor({
    'extract_images': True,
    'extract_references': True,
    'max_summary_length': 300
})

# Extract content
result = extractor.extract_from_url("https://en.wikipedia.org/wiki/Artificial_intelligence")

if 'error' not in result:
    print(f"Title: {result['metadata']['title']}")
    print(f"Summary: {result['content']['summary']}")
    print(f"Word count: {result['statistics']['word_count']}")
```

### Database Integration
```python
from contextbox.extractors.wikipedia import extract_and_store_wikipedia
from contextbox.database import ContextDatabase

# Initialize database
db = ContextDatabase()

# Extract and store directly
artifact_id = extract_and_store_wikipedia(
    capture_id=123,
    wikipedia_url="https://en.wikipedia.org/wiki/Quantum_computing",
    database=db
)
```

### Convenience Functions
```python
from contextbox.extractors import extract_wikipedia_content

# Quick extraction
result = extract_wikipedia_content("https://de.wikipedia.org/wiki/Künstliche_Intelligenz")
```

## API Response Structure

The extraction returns a comprehensive dictionary with the following structure:

```json
{
  "type": "wikipedia_extraction",
  "timestamp": "2023-XX-XXTXX:XX:XX",
  "source_url": "https://en.wikipedia.org/wiki/Article_Title",
  "metadata": {
    "title": "Article Title",
    "language": "en",
    "extraction_method": "mediawiki_api"
  },
  "content": {
    "raw_html": "<html>...</html>",
    "main_text": "Full article text...",
    "summary": "Article summary...",
    "lead_section": "Lead section text..."
  },
  "structure": {
    "sections": [{"title": "Section 1", "level": 2}],
    "categories": ["Category1", "Category2"],
    "infobox": {}
  },
  "links": {
    "internal": ["Link1", "Link2"],
    "external": []
  },
  "media": {
    "images": [{"title": "Image.jpg", "wiki_url": "..."}],
    "other_media": []
  },
  "references": {
    "count": 15,
    "list": [{"index": 1, "text": "Reference..."}]
  },
  "statistics": {
    "word_count": 2500,
    "character_count": 15000,
    "extraction_confidence": 0.9
  }
}
```

## Configuration Options

The extractor supports the following configuration options:

```python
config = {
    'timeout': 30,                    # Request timeout in seconds
    'max_retries': 3,                 # Maximum retry attempts
    'api_language': 'en',             # Default language for API
    'extract_images': True,           # Whether to extract image info
    'extract_references': True,       # Whether to extract references
    'max_summary_length': 500,        # Maximum summary length
    'user_agent': 'Custom UA/1.0'     # Custom User-Agent string
}
```

## Integration with ContextBox

The module integrates seamlessly with ContextBox through:

1. **Artifact Creation**: Converts extraction results to ContextBox artifacts
2. **Database Storage**: Direct storage to ContextBox database
3. **Metadata Richness**: Stores confidence scores, word counts, categories
4. **Multiple Artifacts**: Creates both main article and summary artifacts

## Dependencies

The module requires:
- `requests` - for HTTP API calls
- `beautifulsoup4` - for HTML parsing and cleaning
- Standard library modules: `urllib.parse`, `re`, `json`, `logging`

## Testing

Comprehensive testing has been implemented in:
- `/workspace/contextbox/test_wikipedia_extraction.py` - Full test suite
- `/workspace/contextbox/demo_wikipedia.py` - Simple demonstration

Test coverage includes:
- URL parsing for various formats
- Content extraction from multiple languages
- Error handling for invalid URLs
- Database integration testing
- Configuration option testing

## Error Handling

The module handles various error scenarios:
- **Invalid URLs**: Returns structured error result
- **Network timeouts**: Implements retry logic with exponential backoff
- **API errors**: Handles rate limits, invalid pages, redirects
- **Parsing errors**: Fallback to simple HTML stripping
- **Missing dependencies**: Graceful degradation

## Performance

- **Configurable timeouts** prevent hanging requests
- **Rate limiting friendly** with proper User-Agent headers
- **Retry logic** for transient network issues
- **Efficient HTML parsing** with BeautifulSoup
- **Memory efficient** with streaming for large content

## Summary

The Wikipedia extraction module is fully implemented with all requested features:

✅ MediaWiki API integration  
✅ URL parsing and validation  
✅ Content cleaning and extraction  
✅ Summary generation  
✅ Link extraction  
✅ Image and media information  
✅ Reference handling  
✅ Multi-language support  
✅ Comprehensive error handling  
✅ ContextBox database integration  

The implementation is production-ready, well-tested, and integrates seamlessly with the existing ContextBox architecture.