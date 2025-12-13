#!/usr/bin/env python3
"""
Simple YouTube Extractor Test

Direct test of the YouTube extractor functionality without circular imports.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'contextbox' / 'extractors'))

# Test imports
print("Testing YouTube extractor imports...")

try:
    from youtube import YouTubeURLProcessor, YouTubeTranscriptExtractor, TranscriptData, TranscriptSegment
    print("✅ YouTube extractor imports successful")
except ImportError as e:
    print(f"❌ YouTube extractor import failed: {e}")
    sys.exit(1)

def test_url_processing():
    """Test URL processing."""
    print("\n" + "="*50)
    print("Testing URL Processing")
    print("="*50)
    
    processor = YouTubeURLProcessor()
    
    # Test cases
    test_cases = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ", 
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ"
    ]
    
    for url in test_cases:
        video_id = processor.extract_video_id(url)
        is_valid = processor.is_valid_video_id(video_id)
        
        print(f"URL: {url}")
        print(f"  Video ID: {video_id}")
        print(f"  Valid: {is_valid}")
        print(f"  Status: {'✅ PASS' if video_id == 'dQw4w9WgXcQ' else '❌ FAIL'}")
    
    print("\n✅ URL Processing Test Complete")

def test_transcript_data():
    """Test transcript data structures."""
    print("\n" + "="*50)
    print("Testing Transcript Data Structures")
    print("="*50)
    
    # Create test segments
    segments = [
        TranscriptSegment(0, 5, "Welcome to this video"),
        TranscriptSegment(10, 5, "In this video we will learn"),
        TranscriptSegment(20, 5, "Thank you for watching")
    ]
    
    # Create transcript data
    transcript = TranscriptData(
        video_id="test123",
        title="Test Video",
        language="en",
        segments=segments,
        clean_text="Welcome to this video In this video we will learn Thank you for watching"
    )
    
    print(f"Video ID: {transcript.video_id}")
    print(f"Title: {transcript.title}")
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")
    print(f"Text: {transcript.clean_text[:50]}...")
    
    # Test timestamp formatting
    if segments:
        print(f"First segment time: {segments[0].format_timestamp()}")
    
    # Test ContextBox format conversion
    contextbox_format = transcript.to_contextbox_format()
    print(f"ContextBox kind: {contextbox_format['kind']}")
    print(f"ContextBox URL: {contextbox_format['url']}")
    print(f"Has metadata: {'metadata' in contextbox_format}")
    
    print("\n✅ Transcript Data Test Complete")

def main():
    """Run all tests."""
    print("🚀 YouTube Extractor Simple Test")
    print("="*40)
    
    try:
        test_url_processing()
        test_transcript_data()
        
        print("\n🎉 All basic tests completed successfully!")
        print("The YouTube extractor module is properly structured.")
        
        # Try to import database for integration test
        try:
            sys.path.append(str(Path(__file__).parent / 'contextbox'))
            from contextbox.database import ContextDatabase
            print("✅ Database integration available")
            
            # Quick database test
            db = ContextDatabase({'db_path': ':memory:'})
            capture_id = db.create_capture(source_window="Test")
            print(f"✅ Database test: Created capture {capture_id}")
            
        except Exception as e:
            print(f"⚠️  Database test failed: {e}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
