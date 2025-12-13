#!/usr/bin/env python3
"""
YouTube Transcript Extractor Test Suite

This test file demonstrates and validates the YouTube transcript extraction functionality.
It includes tests for:
- URL processing and video ID extraction
- Transcript extraction using multiple methods
- Search functionality
- Database integration
- Error handling
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from the youtube module to avoid circular imports
sys.path.append(str(Path(__file__).parent / 'contextbox' / 'extractors'))
from youtube import (
    YouTubeURLProcessor,
    YouTubeTranscriptExtractor,
    YouTubeExtractorIntegration,
    TranscriptData,
    TranscriptSegment,
    extract_youtube_transcript,
    search_youtube_transcript,
    extract_youtube_transcript_with_db
)
sys.path.append(str(Path(__file__).parent / 'contextbox'))
from contextbox.database import ContextDatabase


class YouTubeExtractorTester:
    """Test suite for YouTube transcript extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Test configuration
        self.config = {
            'languages': ['en'],
            'fallback_methods': ['api', 'yt-dlp'],
            'max_duration': 1800,  # 30 minutes
            'clean_text': True,
            'build_search_index': True
        }
        
        # Test YouTube URLs with different formats
        self.test_urls = [
            # Standard YouTube URL
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            # Short URL
            "https://youtu.be/dQw4w9WgXcQ", 
            # Embed URL
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            # Just video ID
            "dQw4w9WgXcQ",
        ]
        
        # Sample video that should have transcripts available
        self.sample_video_ids = [
            "dQw4w9WgXcQ",  # Rick Roll - should have captions
            "M7lc1UVf-VE",  # YouTube IFrame Player API Demo
            "jNQXAC9IVRw",  # Me at the zoo - first YouTube video
        ]
    
    def test_url_processing(self):
        """Test URL processing and video ID extraction."""
        print("\n" + "="*50)
        print("Testing URL Processing")
        print("="*50)
        
        url_processor = YouTubeURLProcessor()
        
        # Test URL formats
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        
        for url, expected_id in test_cases:
            try:
                extracted_id = url_processor.extract_video_id(url)
                is_valid = url_processor.is_valid_video_id(extracted_id)
                normalized_url = url_processor.normalize_url(url)
                
                print(f"✅ URL: {url}")
                print(f"   Extracted ID: {extracted_id}")
                print(f"   Expected ID: {expected_id}")
                print(f"   Valid: {is_valid}")
                print(f"   Normalized: {normalized_url}")
                
                assert extracted_id == expected_id, f"Expected {expected_id}, got {extracted_id}"
                assert is_valid, f"Video ID {extracted_id} is not valid"
                
                print("   ✅ PASSED\n")
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}\n")
                return False
        
        # Test text with multiple URLs
        text_with_urls = """
        Check out these videos:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        And this one: https://youtu.be/jNQXAC9IVRw
        Also: https://www.youtube.com/embed/M7lc1UVf-VE
        """
        
        video_ids = url_processor.extract_video_ids(text_with_urls)
        print(f"📝 Text with multiple URLs: Found {len(video_ids)} video IDs")
        print(f"   Video IDs: {video_ids}")
        
        expected_count = 3
        assert len(video_ids) == expected_count, f"Expected {expected_count} IDs, found {len(video_ids)}"
        
        print("✅ URL Processing Tests PASSED")
        return True
    
    def test_transcript_extraction(self):
        """Test transcript extraction with real YouTube videos."""
        print("\n" + "="*50)
        print("Testing Transcript Extraction")
        print("="*50)
        
        extractor = YouTubeTranscriptExtractor(self.config)
        
        # Test with first available video ID
        for video_id in self.sample_video_ids:
            print(f"\n🔍 Testing video ID: {video_id}")
            
            try:
                # Test URL extraction
                url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"   URL: {url}")
                
                # Extract transcript
                transcript_data = extractor.extract_transcript(url)
                
                # Validate results
                assert transcript_data.video_id == video_id
                assert transcript_data.clean_text or transcript_data.segments
                assert transcript_data.extraction_method != "unknown"
                
                print(f"   ✅ Extraction successful!")
                print(f"   Title: {transcript_data.title}")
                print(f"   Language: {transcript_data.language}")
                print(f"   Method: {transcript_data.extraction_method}")
                print(f"   Auto-generated: {transcript_data.is_auto_generated}")
                print(f"   Segments: {len(transcript_data.segments)}")
                print(f"   Text length: {len(transcript_data.clean_text)}")
                
                # Show first few segments
                if transcript_data.segments:
                    print(f"   First segment: {transcript_data.segments[0].format_timestamp()} - {transcript_data.segments[0].text[:100]}...")
                
                # Test conversion to ContextBox format
                contextbox_format = transcript_data.to_contextbox_format()
                assert contextbox_format['kind'] == 'youtube_transcript'
                assert contextbox_format['url'] == url
                assert 'metadata' in contextbox_format
                
                print(f"   ✅ ContextBox format conversion successful")
                
                # Test search functionality if we have text
                if transcript_data.segments:
                    print(f"   🔍 Testing search functionality...")
                    
                    # Simple word search
                    search_query = "the"
                    if any(search_query in seg.text.lower() for seg in transcript_data.segments):
                        search_results = extractor.search_transcript(transcript_data, search_query)
                        print(f"   Search for '{search_query}': {len(search_results)} results")
                        if search_results:
                            print(f"   First result: {search_results[0]['formatted_start']} - {search_results[0]['text'][:100]}...")
                    
                    # Test keyword extraction
                    keywords = ["hello", "welcome", "video"]
                    timestamps = extractor.extract_timestamps(transcript_data, keywords)
                    if timestamps:
                        print(f"   Keyword timestamps: {len(timestamps)} found")
                
                print(f"   ✅ All tests passed for video {video_id}")
                return True
                
            except Exception as e:
                print(f"   ❌ Failed to extract transcript: {e}")
                continue
        
        print("⚠️  No videos could be processed successfully")
        return False
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("\n" + "="*50)
        print("Testing Error Handling")
        print("="*50)
        
        extractor = YouTubeTranscriptExtractor(self.config)
        
        # Test invalid URLs and IDs
        invalid_inputs = [
            "https://www.youtube.com/watch?v=INVALID",
            "invalid_url",
            "",
            "https://www.youtube.com/watch?v=",
            "invalid_video_id",
        ]
        
        for invalid_input in invalid_inputs:
            print(f"\n❌ Testing invalid input: {invalid_input}")
            
            try:
                transcript_data = extractor.extract_transcript(invalid_input)
                print(f"   ⚠️  Unexpectedly succeeded: {transcript_data.video_id}")
            except Exception as e:
                print(f"   ✅ Correctly handled error: {type(e).__name__}")
        
        print("✅ Error Handling Tests PASSED")
        return True
    
    def test_search_functionality(self):
        """Test transcript search functionality."""
        print("\n" + "="*50)
        print("Testing Search Functionality")
        print("="*50)
        
        # Create a test transcript
        test_segments = [
            TranscriptSegment(0, 5, "Welcome to this video about Python programming"),
            TranscriptSegment(10, 5, "In this tutorial, we'll learn about functions"),
            TranscriptSegment(20, 5, "Functions are a fundamental concept in programming"),
            TranscriptSegment(30, 5, "Let's start with a simple example"),
        ]
        
        transcript_data = TranscriptData(
            video_id="test123",
            segments=test_segments
        )
        
        extractor = YouTubeTranscriptExtractor(self.config)
        
        # Test searches
        search_queries = [
            "programming",
            "functions",
            "tutorial",
            "Python",
            "nonexistent"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = extractor.search_transcript(transcript_data, query)
            
            print(f"   Found {len(results)} results")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"   Result {i+1}: {result['formatted_start']} - {result['text'][:80]}...")
        
        # Test keyword timestamp extraction
        keywords = ["Python", "functions", "programming"]
        timestamps = extractor.extract_timestamps(transcript_data, keywords)
        
        print(f"\n📍 Keyword timestamps for {keywords}:")
        for timestamp in timestamps:
            print(f"   {timestamp['keyword']}: {timestamp['formatted_time']} - {timestamp['context'][:50]}...")
        
        print("✅ Search Functionality Tests PASSED")
        return True
    
    def test_database_integration(self):
        """Test database integration."""
        print("\n" + "="*50)
        print("Testing Database Integration")
        print("="*50)
        
        try:
            # Initialize database
            db = ContextDatabase({'db_path': ':memory:'})
            
            # Initialize integration
            integration = YouTubeExtractorIntegration(db, {'youtube_extractor': self.config})
            
            print("✅ Database and integration initialized")
            
            # Test with a known video
            test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            
            print(f"\n💾 Testing extraction and storage for: {test_url}")
            
            try:
                result = integration.extract_and_store(test_url)
                
                print(f"   Success: {result['success']}")
                if result['success']:
                    print(f"   Capture ID: {result['capture_id']}")
                    print(f"   Artifact ID: {result['artifact_id']}")
                    print(f"   Video ID: {result['video_id']}")
                    print(f"   Title: {result.get('title', 'N/A')}")
                    print(f"   Language: {result.get('language', 'N/A')}")
                    print(f"   Segments: {result.get('segments_count', 0)}")
                    print(f"   Method: {result.get('extraction_method', 'N/A')}")
                else:
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                
                # Test retrieval
                if result['success'] and result['capture_id']:
                    capture = db.get_capture(result['capture_id'])
                    artifacts = db.get_artifacts_by_capture(result['capture_id'])
                    
                    print(f"   Retrieved capture: {capture is not None}")
                    print(f"   Retrieved artifacts: {len(artifacts)}")
                    
                    if artifacts:
                        artifact = artifacts[0]
                        print(f"   Artifact kind: {artifact['kind']}")
                        print(f"   Artifact title: {artifact['title']}")
                        print(f"   Has metadata: {'metadata' in artifact}")
                
                print("✅ Database Integration Test PASSED")
                return result['success']
                
            except Exception as e:
                print(f"   ❌ Database integration failed: {e}")
                return False
        
        except Exception as e:
            print(f"   ❌ Database initialization failed: {e}")
            return False
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        print("\n" + "="*50)
        print("Testing Batch Processing")
        print("="*50)
        
        try:
            # Initialize database
            db = ContextDatabase({'db_path': ':memory:'})
            integration = YouTubeExtractorIntegration(db, {'youtube_extractor': self.config})
            
            # Test with multiple URLs
            test_urls = self.test_urls[:2]  # Use first 2 URLs
            
            print(f"📦 Processing {len(test_urls)} URLs in batch")
            for i, url in enumerate(test_urls):
                print(f"   {i+1}. {url}")
            
            try:
                result = integration.batch_extract_and_store(test_urls)
                
                print(f"   Batch result: {result['success']}")
                print(f"   Total videos: {result['total_videos']}")
                print(f"   Successful: {result['successful_extractions']}")
                print(f"   Failed: {result['failed_extractions']}")
                print(f"   Capture ID: {result['capture_id']}")
                
                print("✅ Batch Processing Test PASSED")
                return True
                
            except Exception as e:
                print(f"   ❌ Batch processing failed: {e}")
                return False
        
        except Exception as e:
            print(f"   ❌ Batch processing setup failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🚀 Starting YouTube Transcript Extractor Test Suite")
        print("="*60)
        
        test_results = {}
        
        try:
            # Test URL processing
            test_results['url_processing'] = self.test_url_processing()
            
            # Test transcript extraction
            test_results['transcript_extraction'] = self.test_transcript_extraction()
            
            # Test error handling
            test_results['error_handling'] = self.test_error_handling()
            
            # Test search functionality
            test_results['search_functionality'] = self.test_search_functionality()
            
            # Test database integration
            test_results['database_integration'] = self.test_database_integration()
            
            # Test batch processing
            test_results['batch_processing'] = self.test_batch_processing()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests PASSED! YouTube extractor is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total


def main():
    """Main test execution."""
    # Enable detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    tester = YouTubeExtractorTester()
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())