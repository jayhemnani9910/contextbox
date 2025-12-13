#!/usr/bin/env python3
"""
Simple Wikipedia extraction demonstration script.

This script demonstrates the core Wikipedia extraction functionality.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to the path to import contextbox
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextbox.extractors.wikipedia import WikipediaExtractor

def main():
    """Demonstrate Wikipedia extraction functionality."""
    print("Wikipedia Extraction Demo")
    print("=" * 40)
    
    # Configure extractor
    config = {
        'extract_images': True,
        'extract_references': True,
        'max_summary_length': 300,
        'timeout': 15
    }
    
    extractor = WikipediaExtractor(config)
    
    # Test URL
    url = "https://en.wikipedia.org/wiki/Quantum_computing"
    print(f"Extracting content from: {url}")
    print("-" * 40)
    
    try:
        result = extractor.extract_from_url(url)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            # Display results
            metadata = result.get('metadata', {})
            content = result.get('content', {})
            statistics = result.get('statistics', {})
            structure = result.get('structure', {})
            media = result.get('media', {})
            
            print(f"✓ Successfully extracted Wikipedia content!")
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Language: {metadata.get('language', 'N/A')}")
            print(f"Word Count: {statistics.get('word_count', 0)}")
            print(f"Confidence: {statistics.get('extraction_confidence', 0):.2f}")
            print(f"Sections: {len(structure.get('sections', []))}")
            print(f"Categories: {len(structure.get('categories', []))}")
            print(f"Images: {len(media.get('images', []))}")
            
            # Show summary
            summary = content.get('summary', '')
            if summary:
                preview = summary[:200] + '...' if len(summary) > 200 else summary
                print(f"\nSummary: {preview}")
            
            print("\n" + "=" * 40)
            print("Wikipedia extraction working correctly!")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()