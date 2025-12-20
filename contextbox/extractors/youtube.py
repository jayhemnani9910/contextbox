"""
YouTube transcript extraction module for ContextBox.

This module provides comprehensive YouTube transcript extraction functionality including:
- YouTubeTranscriptApi integration for fetching video transcripts
- yt-dlp integration as fallback for downloading captions
- Video ID extraction from various YouTube URL formats
- Transcript processing and cleaning with timestamp handling
- Support for auto-captions and manual captions
- Multiple language support
- Integration with ContextBox database schema
"""

import re
import json
import logging
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

# Optional dependencies
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    YouTubeTranscriptApi = None
    TextFormatter = None

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timestamp."""
    start: float
    duration: float
    text: str
    end: float = None
    
    def __post_init__(self):
        if self.end is None:
            self.end = self.start + self.duration
    
    def format_timestamp(self) -> str:
        """Format timestamp as HH:MM:SS."""
        total_seconds = int(self.start)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class TranscriptData:
    """Complete transcript data for a YouTube video."""
    video_id: str
    title: Optional[str] = None
    language: Optional[str] = None
    is_auto_generated: bool = False
    segments: List[TranscriptSegment] = None
    raw_text: str = ""
    clean_text: str = ""
    search_index: Dict[str, List[Tuple[int, int]]] = None
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.segments is None:
            self.segments = []
        if self.search_index is None:
            self.search_index = {}
        if self.metadata is None:
            self.metadata = {}
        
        # Generate clean text from segments
        if self.segments and not self.clean_text:
            self.clean_text = " ".join([seg.text for seg in self.segments])
    
    def to_contextbox_format(self) -> Dict[str, Any]:
        """Convert to ContextBox artifact format."""
        return {
            'kind': 'youtube_transcript',
            'url': f"https://www.youtube.com/watch?v={self.video_id}",
            'title': self.title or f"YouTube Transcript - {self.video_id}",
            'text': self.clean_text,
            'metadata': {
                'video_id': self.video_id,
                'language': self.language,
                'is_auto_generated': self.is_auto_generated,
                'extraction_method': self.extraction_method,
                'timestamp_data': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'duration': seg.duration,
                        'text': seg.text,
                        'formatted_time': seg.format_timestamp()
                    }
                    for seg in self.segments
                ],
                'search_index': self.search_index,
                'segment_count': len(self.segments),
                'extraction_timestamp': datetime.now().isoformat(),
                **self.metadata
            }
        }


class YouTubeURLProcessor:
    """Extract and process YouTube URLs and video IDs."""
    
    # YouTube URL patterns
    URL_PATTERNS = [
        # Standard YouTube watch URLs
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # youtu.be short URLs
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        # YouTube embed URLs
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        # YouTube live URLs
        r'(?:https?://)?(?:www\.)?youtube\.com/live/([a-zA-Z0-9_-]{11})',
        # YouTube shorts URLs
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        # YouTube playlist URLs (extract video ID from start parameter)
        r'(?:https?://)?(?:www\.)?youtube\.com/.*?v=([a-zA-Z0-9_-]{11})',
    ]
    
    @classmethod
    def extract_video_ids(cls, text: str) -> List[str]:
        """
        Extract YouTube video IDs from text.
        
        Args:
            text: Text containing YouTube URLs
            
        Returns:
            List of video IDs found
        """
        video_ids = set()
        
        for pattern in cls.URL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                video_id = match.group(1)
                if video_id and len(video_id) == 11:
                    video_ids.add(video_id)
        
        return list(video_ids)
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """
        Extract video ID from a single YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        if not url:
            return None
        
        # If it looks like a video ID already, return it
        if cls.is_valid_video_id(url):
            return url
        
        video_ids = cls.extract_video_ids(url)
        return video_ids[0] if video_ids else None
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """
        Normalize YouTube URL to standard format.
        
        Args:
            url: YouTube URL
            
        Returns:
            Normalized YouTube watch URL
        """
        video_id = cls.extract_video_id(url)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return url
    
    @classmethod
    def is_valid_video_id(cls, video_id: str) -> bool:
        """
        Check if a video ID is valid.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if valid format
        """
        return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


class YouTubeTranscriptExtractor:
    """
    Main YouTube transcript extraction class with multiple fallback methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the YouTube transcript extractor.
        
        Args:
            config: Configuration dictionary
                - languages: List of preferred languages (default: ['en'])
                - fallback_methods: List of methods to try (default: ['api', 'yt-dlp'])
                - max_duration: Maximum video duration in seconds to process
                - clean_text: Whether to clean and format text
                - build_search_index: Whether to build search index
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.preferred_languages = self.config.get('languages', ['en'])
        self.fallback_methods = self.config.get('fallback_methods', ['api', 'yt-dlp'])
        self.max_duration = self.config.get('max_duration', 3600)  # 1 hour
        self.clean_text = self.config.get('clean_text', True)
        self.build_search_index = self.config.get('build_search_index', True)
        
        # Check dependencies
        if not YOUTUBE_API_AVAILABLE:
            self.logger.warning("YouTubeTranscriptApi not available. Install with: pip install youtube-transcript-api")
        if not YT_DLP_AVAILABLE:
            self.logger.warning("yt-dlp not available. Install with: pip install yt-dlp")
        
        self.logger.info("YouTubeTranscriptExtractor initialized")
    
    def extract_transcript(self, url_or_video_id: str) -> TranscriptData:
        """
        Extract transcript from YouTube video using available methods.
        
        Args:
            url_or_video_id: YouTube URL or video ID
            
        Returns:
            TranscriptData object containing transcript and metadata
        """
        try:
            # Extract video ID
            if YouTubeURLProcessor.is_valid_video_id(url_or_video_id):
                video_id = url_or_video_id
            else:
                video_id = YouTubeURLProcessor.extract_video_id(url_or_video_id)
            
            if not video_id:
                raise ValueError(f"Invalid YouTube URL or video ID: {url_or_video_id}")
            
            self.logger.info(f"Extracting transcript for video ID: {video_id}")
            
            # Try extraction methods in order
            transcript_data = None
            extraction_method = "unknown"
            
            for method in self.fallback_methods:
                try:
                    if method == 'api' and YOUTUBE_API_AVAILABLE:
                        transcript_data = self._extract_via_api(video_id)
                        extraction_method = "YouTubeTranscriptApi"
                        break
                    elif method == 'yt-dlp' and YT_DLP_AVAILABLE:
                        transcript_data = self._extract_via_yt_dlp(video_id)
                        extraction_method = "yt-dlp"
                        break
                except Exception as e:
                    self.logger.warning(f"Extraction via {method} failed: {e}")
                    continue
            
            if not transcript_data:
                raise RuntimeError(f"All extraction methods failed for video {video_id}")
            
            # Process transcript data
            transcript_data.extraction_method = extraction_method
            transcript_data.video_id = video_id
            
            # Get video metadata
            try:
                video_info = self._get_video_info(video_id)
                transcript_data.title = video_info.get('title')
                duration = video_info.get('duration')
                if duration and duration > self.max_duration:
                    self.logger.warning(f"Video duration {duration}s exceeds limit {self.max_duration}s")
                transcript_data.metadata.update({
                    'duration': duration,
                    'uploader': video_info.get('uploader'),
                    'upload_date': video_info.get('upload_date'),
                    'view_count': video_info.get('view_count'),
                })
            except Exception as e:
                self.logger.debug(f"Could not fetch video metadata: {e}")
            
            # Build search index if requested
            if self.build_search_index and transcript_data.segments:
                transcript_data.search_index = self._build_search_index(transcript_data.segments)
            
            # Clean text if requested
            if self.clean_text and transcript_data.segments:
                transcript_data.clean_text = self._clean_text(transcript_data.clean_text)
            
            self.logger.info(f"Successfully extracted transcript via {extraction_method}")
            return transcript_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract transcript: {e}")
            raise
    
    def extract_multiple_transcripts(self, urls_or_ids: List[str]) -> List[TranscriptData]:
        """
        Extract transcripts from multiple YouTube videos.
        
        Args:
            urls_or_ids: List of YouTube URLs or video IDs
            
        Returns:
            List of TranscriptData objects
        """
        results = []
        for url_or_id in urls_or_ids:
            try:
                transcript = self.extract_transcript(url_or_id)
                results.append(transcript)
            except Exception as e:
                self.logger.error(f"Failed to extract transcript for {url_or_id}: {e}")
                # Create failed transcript data
                video_id = YouTubeURLProcessor.extract_video_id(url_or_id) or url_or_id
                results.append(TranscriptData(
                    video_id=video_id,
                    metadata={'error': str(e), 'extraction_failed': True}
                ))
        
        return results
    
    def _extract_via_api(self, video_id: str) -> TranscriptData:
        """Extract transcript using YouTubeTranscriptApi."""
        if not YOUTUBE_API_AVAILABLE:
            raise ImportError("YouTubeTranscriptApi not available")
        
        try:
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find best transcript
            transcript = None
            language = None
            
            # First try manually created transcripts
            try:
                transcript = transcript_list.find_transcript(self.preferred_languages)
                language = transcript.language_code
            except:
                # Fallback to auto-generated transcripts
                try:
                    transcript = transcript_list.find_transcript(self.preferred_languages)
                    language = transcript.language_code
                except:
                    # Use any available transcript
                    all_transcripts = transcript_list
                    if all_transcripts:
                        transcript = list(all_transcripts)[0]
                        language = transcript.language_code
            
            if not transcript:
                raise ValueError(f"No transcripts available for video {video_id}")
            
            # Get transcript data
            transcript_data = transcript.fetch()
            
            # Process segments
            segments = []
            for entry in transcript_data:
                segment = TranscriptSegment(
                    start=entry['start'],
                    duration=entry['duration'],
                    text=entry['text']
                )
                segments.append(segment)
            
            # Create transcript data
            result = TranscriptData(
                video_id=video_id,
                language=language,
                is_auto_generated=transcript.is_generated,
                segments=segments,
                metadata={
                    'track_name': getattr(transcript, 'track_name', None),
                    'kind': getattr(transcript, 'kind', None)
                }
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"YouTubeTranscriptApi extraction failed: {e}")
    
    def _extract_via_yt_dlp(self, video_id: str) -> TranscriptData:
        """Extract transcript using yt-dlp."""
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp not available")
        
        try:
            # yt-dlp options for subtitle extraction
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': self.preferred_languages,
                'skip_download': True,
                'extract_flat': False,
            }
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Look for subtitle data
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Find best subtitle format
                subtitle_url = None
                language = None
                is_auto_generated = False
                
                # Check manual subtitles first
                for lang in self.preferred_languages:
                    if lang in subtitles and subtitles[lang]:
                        subtitle_url = subtitles[lang][0].get('url')
                        language = lang
                        break
                
                # Fallback to auto-generated subtitles
                if not subtitle_url:
                    for lang in self.preferred_languages:
                        if lang in automatic_captions and automatic_captions[lang]:
                            subtitle_url = automatic_captions[lang][0].get('url')
                            language = lang
                            is_auto_generated = True
                            break
                
                if not subtitle_url:
                    raise ValueError(f"No subtitles available for video {video_id}")
                
                # Download and parse subtitle file
                segments = self._download_and_parse_subtitles(subtitle_url)
                
                return TranscriptData(
                    video_id=video_id,
                    language=language,
                    is_auto_generated=is_auto_generated,
                    segments=segments,
                    metadata={
                        'subtitle_format': 'srt',
                        'info_extra': {
                            'title': info.get('title'),
                            'uploader': info.get('uploader'),
                            'duration': info.get('duration')
                        }
                    }
                )
                
        except Exception as e:
            raise RuntimeError(f"yt-dlp extraction failed: {e}")
    
    def _download_and_parse_subtitles(self, subtitle_url: str) -> List[TranscriptSegment]:
        """Download and parse subtitle file from URL."""
        try:
            import requests
            
            response = requests.get(subtitle_url)
            response.raise_for_status()
            content = response.text
            
            return self._parse_srt(content)
            
        except Exception as e:
            # Try using yt-dlp to download subtitle
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as tmp_file:
                    tmp_file.write(content)
                    tmp_file.flush()
                    
                    # Use yt-dlp to convert if needed
                    return self._parse_srt_file(tmp_file.name)
            except Exception as inner_e:
                raise RuntimeError(f"Failed to download and parse subtitles: {inner_e}")
    
    def _parse_srt(self, srt_content: str) -> List[TranscriptSegment]:
        """Parse SRT subtitle content."""
        segments = []
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                # Skip sequence number
                timestamp_line = lines[1]
                text_lines = lines[2:]
                
                # Parse timestamp
                timestamp_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    timestamp_line
                )
                
                if timestamp_match:
                    start = self._srt_time_to_seconds(timestamp_match.group(1), timestamp_match.group(2), 
                                                    timestamp_match.group(3), timestamp_match.group(4))
                    end = self._srt_time_to_seconds(timestamp_match.group(5), timestamp_match.group(6), 
                                                  timestamp_match.group(7), timestamp_match.group(8))
                    
                    text = ' '.join(text_lines)
                    
                    segment = TranscriptSegment(
                        start=start,
                        duration=end - start,
                        text=text,
                        end=end
                    )
                    segments.append(segment)
        
        return segments
    
    def _srt_time_to_seconds(self, hours: str, minutes: str, seconds: str, milliseconds: str) -> float:
        """Convert SRT time format to seconds."""
        return (int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000)
    
    def _parse_srt_file(self, file_path: str) -> List[TranscriptSegment]:
        """Parse SRT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._parse_srt(content)
    
    def _get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata using yt-dlp."""
        if not YT_DLP_AVAILABLE:
            return {}
        
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title'),
                    'uploader': info.get('uploader'),
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'description': info.get('description'),
                }
                
        except Exception as e:
            self.logger.debug(f"Could not fetch video info: {e}")
            return {}
    
    def _build_search_index(self, segments: List[TranscriptSegment]) -> Dict[str, List[Tuple[int, int]]]:
        """Build search index for fast text search."""
        search_index = {}
        full_text = ""
        segment_positions = []
        
        # Build full text with segment boundaries
        for segment in segments:
            start_pos = len(full_text)
            full_text += segment.text + " "
            end_pos = len(full_text) - 1
            segment_positions.append((start_pos, end_pos, segment.start))
        
        # Index words and their positions
        words = re.findall(r'\b\w+\b', full_text.lower())
        positions = 0
        
        for word in words:
            if word not in search_index:
                search_index[word] = []
            search_index[word].append(positions)
            positions += len(word) + 1
        
        return search_index
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR/text issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Add space after punctuation
        
        # Handle line breaks
        text = re.sub(r'\n+', ' ', text)
        
        return text.strip()
    
    def search_transcript(self, transcript: TranscriptData, query: str) -> List[Dict[str, Any]]:
        """
        Search within a transcript.
        
        Args:
            transcript: TranscriptData object
            query: Search query
            
        Returns:
            List of matching segments with timestamps
        """
        if not transcript.search_index or not query:
            # Fallback to simple search
            return self._simple_search(transcript, query)
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        matches = []
        
        # Find segments containing all query words
        for segment in transcript.segments:
            segment_text_lower = segment.text.lower()
            if all(word in segment_text_lower for word in query_words):
                matches.append({
                    'segment': segment,
                    'start_time': segment.start,
                    'end_time': segment.end,
                    'formatted_start': segment.format_timestamp(),
                    'text': segment.text,
                    'relevance_score': sum(1 for word in query_words if word in segment_text_lower)
                })
        
        # Sort by relevance score
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return matches
    
    def _simple_search(self, transcript: TranscriptData, query: str) -> List[Dict[str, Any]]:
        """Simple text search fallback."""
        query_lower = query.lower()
        matches = []
        
        for segment in transcript.segments:
            if query_lower in segment.text.lower():
                matches.append({
                    'segment': segment,
                    'start_time': segment.start,
                    'end_time': segment.end,
                    'formatted_start': segment.format_timestamp(),
                    'text': segment.text,
                    'relevance_score': segment.text.lower().count(query_lower)
                })
        
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return matches
    
    def extract_timestamps(self, transcript: TranscriptData, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Extract timestamps for specific keywords.
        
        Args:
            transcript: TranscriptData object
            keywords: List of keywords to find
            
        Returns:
            List of timestamp entries for found keywords
        """
        timestamps = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for segment in transcript.segments:
                if keyword_lower in segment.text.lower():
                    timestamps.append({
                        'keyword': keyword,
                        'timestamp': segment.start,
                        'formatted_time': segment.format_timestamp(),
                        'context': segment.text,
                        'segment': segment
                    })
        
        return timestamps


class YouTubeExtractorIntegration:
    """
    Integration layer for ContextBox database storage.
    """
    
    def __init__(self, database_manager, config: Dict[str, Any] = None):
        """
        Initialize YouTube extractor integration.
        
        Args:
            database_manager: ContextBox database manager
            config: Extractor configuration
        """
        self.db = database_manager
        self.config = config or {}
        self.extractor = YouTubeTranscriptExtractor(self.config.get('youtube_extractor', {}))
        self.logger = logging.getLogger(__name__)
    
    def extract_and_store(self, url_or_video_id: str, capture_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract YouTube transcript and store in database.
        
        Args:
            url_or_video_id: YouTube URL or video ID
            capture_id: Optional ContextBox capture ID
            
        Returns:
            Result dictionary with extraction details
        """
        try:
            # Extract transcript
            transcript_data = self.extractor.extract_transcript(url_or_video_id)
            
            # Create or use capture
            if capture_id is None:
                capture_id = self.db.create_capture(
                    source_window="YouTube Transcript Extraction",
                    notes=f"Transcript extraction for {url_or_video_id}"
                )
            
            # Store transcript as artifact
            artifact_data = transcript_data.to_contextbox_format()
            artifact_id = self.db.create_artifact(
                capture_id=capture_id,
                kind=artifact_data['kind'],
                url=artifact_data['url'],
                title=artifact_data['title'],
                text=artifact_data['text'],
                metadata=artifact_data['metadata']
            )
            
            self.logger.info(f"Stored YouTube transcript for video {transcript_data.video_id}")
            
            return {
                'success': True,
                'capture_id': capture_id,
                'artifact_id': artifact_id,
                'video_id': transcript_data.video_id,
                'title': transcript_data.title,
                'language': transcript_data.language,
                'segments_count': len(transcript_data.segments),
                'extraction_method': transcript_data.extraction_method,
                'is_auto_generated': transcript_data.is_auto_generated
            }
            
        except Exception as e:
            self.logger.error(f"YouTube extraction and storage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'capture_id': capture_id
            }
    
    def batch_extract_and_store(self, urls_or_ids: List[str], capture_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract multiple YouTube transcripts and store in database.
        
        Args:
            urls_or_ids: List of YouTube URLs or video IDs
            capture_id: Optional ContextBox capture ID
            
        Returns:
            Result dictionary with extraction details
        """
        results = []
        successful_extractions = 0
        failed_extractions = 0
        
        # Create single capture for batch
        if capture_id is None:
            capture_id = self.db.create_capture(
                source_window="YouTube Batch Transcript Extraction",
                notes=f"Batch extraction of {len(urls_or_ids)} videos"
            )
        
        for url_or_id in urls_or_ids:
            result = self.extract_and_store(url_or_id, capture_id)
            results.append(result)
            
            if result['success']:
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        return {
            'success': True,
            'capture_id': capture_id,
            'total_videos': len(urls_or_ids),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'results': results
        }


# Convenience functions for common use cases
def extract_youtube_transcript(url_or_video_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[TranscriptData]:
    """
    Convenience function to extract YouTube transcript.
    
    Args:
        url_or_video_id: YouTube URL or video ID
        config: Optional configuration
        
    Returns:
        TranscriptData object or None if extraction fails
    """
    try:
        extractor = YouTubeTranscriptExtractor(config)
        return extractor.extract_transcript(url_or_video_id)
    except Exception as e:
        logging.error(f"YouTube transcript extraction failed: {e}")
        return None


def search_youtube_transcript(transcript: TranscriptData, query: str) -> List[Dict[str, Any]]:
    """
    Convenience function to search within transcript.
    
    Args:
        transcript: TranscriptData object
        query: Search query
        
    Returns:
        List of search results
    """
    extractor = YouTubeTranscriptExtractor()
    return extractor.search_transcript(transcript, query)


def extract_youtube_transcript_with_db(url_or_video_id: str, db_manager, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to extract and store YouTube transcript.
    
    Args:
        url_or_video_id: YouTube URL or video ID
        db_manager: ContextBox database manager
        config: Optional configuration
        
    Returns:
        Extraction result dictionary
    """
    integration = YouTubeExtractorIntegration(db_manager, config)
    return integration.extract_and_store(url_or_video_id)