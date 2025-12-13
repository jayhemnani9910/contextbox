# YouTube Transcript Extractor for ContextBox

This document describes the implementation of the YouTube transcript extraction module for ContextBox, which provides comprehensive YouTube video transcript extraction with multiple fallback methods and seamless database integration.

## 🎯 Features

### Core Functionality
- **Multi-method transcript extraction**: Uses YouTubeTranscriptApi with yt-dlp fallback
- **Universal URL support**: Handles all YouTube URL formats (watch, youtu.be, embed, live, shorts)
- **Timestamp preservation**: Maintains exact timing information for searchable transcripts
- **Multiple language support**: Automatically detects and prioritizes languages
- **Auto/manual caption support**: Works with both types of captions
- **Intelligent search**: Build searchable indexes with timestamp-accurate results

### Advanced Features
- **Error handling**: Graceful fallback when transcripts are unavailable
- **Database integration**: Seamless ContextBox artifact format conversion
- **Batch processing**: Extract multiple videos efficiently
- **Text cleaning**: Automatic formatting and cleanup
- **Metadata enrichment**: Video information extraction and storage

## 📁 Implementation Structure

```
contextbox/extractors/youtube.py
├── YouTubeURLProcessor              # URL parsing and video ID extraction
├── YouTubeTranscriptExtractor       # Main extraction engine
├── YouTubeExtractorIntegration     # Database integration layer
├── TranscriptData                  # Data model for transcript results
└── TranscriptSegment               # Individual transcript segment model
```

## 🔧 Dependencies

### Required (Already in requirements.txt)
- `youtube-transcript-api>=0.4.0` - Primary transcript extraction method
- `yt-dlp>=2023.0.0` - Fallback method for caption downloading

### Optional
- `requests` - For subtitle file downloads
- Standard library modules: `re`, `json`, `logging`, `tempfile`, `subprocess`

## 🚀 Usage Examples

### Basic Transcript Extraction

```python
from contextbox.extractors.youtube import extract_youtube_transcript

# Extract transcript from YouTube URL
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
transcript = extract_youtube_transcript(url)

if transcript:
    print(f"Title: {transcript.title}")
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")
    print(f"Text: {transcript.clean_text[:200]}...")
```

### Advanced Configuration

```python
from contextbox.extractors.youtube import YouTubeTranscriptExtractor

config = {
    'languages': ['en', 'es', 'fr'],        # Preferred languages
    'fallback_methods': ['api', 'yt-dlp'],   # Extraction methods in order
    'max_duration': 3600,                    # Max video duration (seconds)
    'clean_text': True,                      # Enable text cleaning
    'build_search_index': True               # Build search index
}

extractor = YouTubeTranscriptExtractor(config)
transcript = extractor.extract_transcript(youtube_url)
```

### URL Processing

```python
from contextbox.extractors.youtube import YouTubeURLProcessor

processor = YouTubeURLProcessor()

# Extract video ID from various URL formats
urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "dQw4w9WgXcQ"  # Just the video ID
]

for url in urls:
    video_id = processor.extract_video_id(url)
    is_valid = processor.is_valid_video_id(video_id)
    normalized = processor.normalize_url(url)
    print(f"URL: {url}")
    print(f"ID: {video_id}, Valid: {is_valid}")
```

### Search Functionality

```python
# Search within transcript
results = extractor.search_transcript(transcript, "python programming")

for result in results:
    print(f"{result['formatted_start']}: {result['text']}")

# Extract keyword timestamps
timestamps = extractor.extract_timestamps(transcript, ["tutorial", "introduction"])

for ts in timestamps:
    print(f"{ts['keyword']}: {ts['formatted_time']} - {ts['context']}")
```

### Database Integration

```python
from contextbox.extractors.youtube import YouTubeExtractorIntegration
from contextbox.contextbox.database import ContextDatabase

# Initialize database and integration
db = ContextDatabase()
integration = YouTubeExtractorIntegration(db)

# Extract and store in ContextBox
result = integration.extract_and_store(youtube_url, capture_id=123)

if result['success']:
    print(f"Stored transcript for video {result['video_id']}")
    print(f"Capture ID: {result['capture_id']}")
    print(f"Artifact ID: {result['artifact_id']}")
```

### Batch Processing

```python
# Extract multiple videos
urls = [
    "https://www.youtube.com/watch?v=video1",
    "https://youtu.be/video2",
    "video3"  # Just video ID
]

results = integration.batch_extract_and_store(urls)
print(f"Processed {results['total_videos']} videos")
print(f"Successful: {results['successful_extractions']}")
print(f"Failed: {results['failed_extractions']}")
```

## 📊 Data Structures

### TranscriptData
```python
@dataclass
class TranscriptData:
    video_id: str                          # YouTube video ID
    title: Optional[str] = None           # Video title
    language: Optional[str] = None        # Transcript language
    is_auto_generated: bool = False       # Auto vs manual captions
    segments: List[TranscriptSegment] = None  # Timestamped segments
    raw_text: str = ""                    # Raw transcript text
    clean_text: str = ""                  # Cleaned text
    search_index: Dict[str, List[Tuple[int, int]]] = None  # Search index
    extraction_method: str = "unknown"    # How transcript was obtained
    metadata: Dict[str, Any] = None       # Additional metadata
```

### TranscriptSegment
```python
@dataclass
class TranscriptSegment:
    start: float                          # Start time in seconds
    duration: float                       # Duration in seconds
    text: str                             # Segment text
    end: float = None                     # End time (computed)
    
    def format_timestamp(self) -> str:    # Format as HH:MM:SS
```

### ContextBox Artifact Format
```python
{
    'kind': 'youtube_transcript',
    'url': 'https://www.youtube.com/watch?v=VIDEO_ID',
    'title': 'Video Title',
    'text': 'Clean transcript text...',
    'metadata': {
        'video_id': 'VIDEO_ID',
        'language': 'en',
        'is_auto_generated': False,
        'extraction_method': 'YouTubeTranscriptApi',
        'timestamp_data': [...],           # Segment details
        'search_index': {...},             # Search index
        'segment_count': 150,
        'extraction_timestamp': '2024-01-01T12:00:00'
    }
}
```

## 🔍 Supported URL Formats

The extractor supports all major YouTube URL formats:

| Format | Example | Description |
|--------|---------|-------------|
| Standard | `https://www.youtube.com/watch?v=VIDEO_ID` | Most common format |
| Short | `https://youtu.be/VIDEO_ID` | YouTube short URLs |
| Embed | `https://www.youtube.com/embed/VIDEO_ID` | Embedded player URLs |
| Live | `https://www.youtube.com/live/VIDEO_ID` | Live stream URLs |
| Shorts | `https://www.youtube.com/shorts/VIDEO_ID` | YouTube Shorts |
| Video ID | `VIDEO_ID` | Just the 11-character ID |

## 🎛️ Configuration Options

### YouTubeTranscriptExtractor Configuration
```python
config = {
    # Language preferences (checked in order)
    'languages': ['en', 'es', 'fr', 'de'],
    
    # Extraction methods (tried in order)
    'fallback_methods': ['api', 'yt-dlp'],
    
    # Maximum video duration to process (seconds)
    'max_duration': 3600,  # 1 hour
    
    # Enable text cleaning and formatting
    'clean_text': True,
    
    # Build search index for fast text search
    'build_search_index': True,
    
    # API-specific options
    'api_timeout': 30,           # Request timeout
    'retry_attempts': 3,         # Retry failed requests
    
    # yt-dlp specific options
    'ydl_quiet': True,           # Suppress yt-dlp output
    'ydl_no_warnings': True      # Suppress warnings
}
```

## 🚨 Error Handling

The extractor provides comprehensive error handling for various scenarios:

### Common Error Types
- **NoTranscriptAvailable**: Video has no captions/transcripts
- **PrivateVideo**: Video is not publicly accessible
- **NetworkError**: Connection or timeout issues
- **InvalidURL**: Unsupported or malformed URL
- **DependencyMissing**: Required libraries not installed

### Graceful Fallbacks
```python
try:
    transcript = extractor.extract_transcript(url)
except Exception as e:
    logger.warning(f"Extraction failed: {e}")
    # Returns None or partial data instead of crashing
```

### Error Recovery
- Multiple extraction methods tried automatically
- Network timeouts and retries
- Partial transcript recovery when possible
- Detailed error logging for debugging

## 📈 Performance Considerations

### Optimization Features
- **Lazy loading**: Transcripts fetched only when needed
- **Caching**: Avoid repeated API calls for same video
- **Batch processing**: Efficient handling of multiple videos
- **Search index**: Fast text search without re-parsing
- **Memory management**: Efficient segment storage

### Scalability
- Supports processing thousands of videos
- Configurable timeouts and retry limits
- Progress tracking for long operations
- Configurable maximum video duration

## 🔗 Integration with ContextBox

### Database Storage
The extractor automatically converts transcript data to ContextBox's artifact format:

```python
# Automatic conversion
artifact_data = transcript.to_contextbox_format()

# Store in database
db.create_artifact(
    capture_id=capture_id,
    kind=artifact_data['kind'],
    url=artifact_data['url'],
    title=artifact_data['title'],
    text=artifact_data['text'],
    metadata=artifact_data['metadata']
)
```

### Capture Integration
```python
# Create capture for YouTube extraction
capture_id = db.create_capture(
    source_window="YouTube Video Analysis",
    notes=f"Transcript extraction for {video_id}"
)

# Extract and store
result = integration.extract_and_store(youtube_url, capture_id)
```

## 🧪 Testing

### Test Coverage
- URL processing for all formats
- Video ID extraction and validation
- Transcript data structure integrity
- Search functionality accuracy
- Database integration
- Error handling scenarios
- Edge cases and invalid inputs

### Running Tests
```bash
# Basic functionality test
python test_youtube_simple.py

# Comprehensive test suite
python test_youtube_extractor.py

# Interactive demo
python demo_youtube_extractor.py
```

## 📝 Implementation Notes

### Design Decisions
1. **Dual extraction methods**: Ensures reliability when one method fails
2. **Timestamp preservation**: Enables accurate video navigation and search
3. **Search index building**: Provides fast text search capabilities
4. **Database integration**: Seamless ContextBox workflow integration
5. **Error resilience**: Graceful handling of network and API issues

### Code Quality
- Comprehensive docstrings for all classes and methods
- Type hints for better IDE support
- Consistent error handling and logging
- Modular design for easy testing and maintenance
- Performance optimizations for large-scale usage

## 🔄 Future Enhancements

Potential improvements for future versions:
- Support for YouTube playlist transcript extraction
- Real-time transcript updates for live streams
- Advanced video metadata extraction (thumbnails, etc.)
- Integration with YouTube Data API for additional features
- Machine learning-based transcript quality assessment
- Multi-language transcript alignment and translation

## 📞 Support

For issues, feature requests, or contributions:
- Check the test files for usage examples
- Review error logs for debugging information
- Ensure all dependencies are properly installed
- Verify video has accessible transcripts

---

**Status**: ✅ Fully implemented and tested
**Version**: 1.0.0
**Last Updated**: 2024-11-05
