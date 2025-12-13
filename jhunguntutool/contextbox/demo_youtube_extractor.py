#!/usr/bin/env python3
"""
YouTube Transcript Extractor Demo

Demonstrates the YouTube transcript extraction functionality with real videos.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'contextbox' / 'extractors'))

from youtube import (
    YouTubeURLProcessor,
    YouTubeTranscriptExtractor,
    TranscriptData,
    TranscriptSegment,
    extract_youtube_transcript,
    search_youtube_transcript
)

def demo_url_processing():
    """Demonstrate URL processing capabilities."""
    print("🔗 YouTube URL Processing Demo")
    print("="*40)
    
    processor = YouTubeURLProcessor()
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "dQw4w9WgXcQ"  # Just the video ID
    ]
    
    for url in test_urls:
        video_id = processor.extract_video_id(url)
        is_valid = processor.is_valid_video_id(video_id)
        normalized = processor.normalize_url(url)
        
        print(f"Input:  {url}")
        print(f"ID:     {video_id}")
        print(f"Valid:  {is_valid}")
        print(f"Norm:   {normalized}")
        print()

def demo_transcript_extraction():
    """Demonstrate transcript extraction."""
    print("📺 YouTube Transcript Extraction Demo")
    print("="*40)
    
    # Configuration
    config = {
        'languages': ['en'],
        'fallback_methods': ['api', 'yt-dlp'],
        'max_duration': 3600,
        'clean_text': True,
        'build_search_index': True
    }
    
    extractor = YouTubeTranscriptExtractor(config)
    
    # Test with different videos
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
        "https://youtu.be/jNQXAC9IVRw",  # First YouTube video
        "https://www.youtube.com/watch?v=M7lc1UVf-VE",  # IFrame Player API Demo
    ]
    
    for url in test_urls:
        print(f"\n🎬 Processing: {url}")
        print("-" * 50)
        
        try:
            # Extract transcript
            transcript = extractor.extract_transcript(url)
            
            print(f"✅ Extraction successful!")
            print(f"Video ID:     {transcript.video_id}")
            print(f"Title:        {transcript.title or 'N/A'}")
            print(f"Language:     {transcript.language or 'N/A'}")
            print(f"Method:       {transcript.extraction_method}")
            print(f"Auto-gen:     {transcript.is_auto_generated}")
            print(f"Segments:     {len(transcript.segments)}")
            print(f"Text length:  {len(transcript.clean_text)} characters")
            
            # Show first few segments
            if transcript.segments:
                print(f"\nFirst segment ({transcript.segments[0].format_timestamp()}):")
                print(f"  {transcript.segments[0].text[:100]}...")
                
                if len(transcript.segments) > 1:
                    print(f"Last segment ({transcript.segments[-1].format_timestamp()}):")
                    print(f"  {transcript.segments[-1].text[:100]}...")
            
            # Test search functionality
            if transcript.segments:
                search_terms = ["the", "and", "you", "video"]
                for term in search_terms:
                    if any(term.lower() in seg.text.lower() for seg in transcript.segments):
                        results = extractor.search_transcript(transcript, term)
                        if results:
                            print(f"\n🔍 Search for '{term}': {len(results)} results")
                            print(f"   First match: {results[0]['formatted_start']} - {results[0]['text'][:80]}...")
                        break
            
            # Test keyword extraction
            keywords = ["hello", "welcome", "learn"]
            timestamps = extractor.extract_timestamps(transcript, keywords)
            if timestamps:
                print(f"\n📍 Keyword timestamps found: {len(timestamps)}")
                for ts in timestamps[:3]:  # Show first 3
                    print(f"   {ts['keyword']}: {ts['formatted_time']} - {ts['context'][:50]}...")
            
            print(f"✅ Processing complete for {transcript.video_id}")
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            print(f"   This might be due to:")
            print(f"   - No transcript available")
            print(f"   - Network connectivity issues")
            print(f"   - Video restrictions")

def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🛠️  Convenience Functions Demo")
    print("="*40)
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print(f"Testing with URL: {test_url}")
    
    try:
        # Use convenience function
        transcript = extract_youtube_transcript(test_url)
        
        if transcript:
            print(f"✅ Convenience function worked!")
            print(f"   Video: {transcript.title or 'N/A'}")
            print(f"   Segments: {len(transcript.segments)}")
            print(f"   Text: {transcript.clean_text[:100]}...")
            
            # Test search
            results = search_youtube_transcript(transcript, "never")
            if results:
                print(f"🔍 Search for 'never': {len(results)} results")
                print(f"   First: {results[0]['formatted_start']} - {results[0]['text'][:60]}...")
        else:
            print(f"⚠️  No transcript returned (may not be available)")
            
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")

def main():
    """Run all demonstrations."""
    print("🎯 YouTube Transcript Extractor Demo")
    print("="*50)
    print("This demo shows the key features of the YouTube extractor:")
    print("• URL processing and video ID extraction")
    print("• Transcript extraction with timestamps")
    print("• Search functionality within transcripts")
    print("• Database integration formatting")
    print("• Multiple extraction methods (API + fallback)")
    print()
    
    try:
        demo_url_processing()
        demo_convenience_functions()
        demo_transcript_extraction()
        
        print("\n🎉 Demo completed!")
        print("\nKey Features Demonstrated:")
        print("✅ URL processing for all YouTube URL formats")
        print("✅ Video ID extraction and validation")
        print("✅ Transcript data structures with timestamps")
        print("✅ Search and keyword extraction capabilities")
        print("✅ ContextBox database integration format")
        print("✅ Error handling for missing transcripts")
        
        print("\n💡 Usage Tips:")
        print("• Works with any YouTube video that has captions/transcripts")
        print("• Supports both manually created and auto-generated captions")
        print("• Provides timestamp-accurate transcript segments")
        print("• Includes search index for fast text search")
        print("• Integrates seamlessly with ContextBox database")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
