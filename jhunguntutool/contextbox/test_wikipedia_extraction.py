#!/usr/bin/env python3
"""
Test script for Wikipedia extraction functionality.

This script tests the WikipediaExtractor class to ensure it can properly
extract content from Wikipedia URLs and integrate with ContextBox.
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path to import contextbox
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from contextbox.extractors.wikipedia import WikipediaExtractor, extract_wikipedia_content
    from contextbox.database import ContextDatabase
    print("✓ Successfully imported Wikipedia extraction modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def test_wikipedia_extraction():
    """Test basic Wikipedia extraction functionality."""
    print("\n=== Testing Wikipedia Extraction ===")
    
    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz",
        "https://en.wikipedia.org/w/index.php?title=Machine_learning"
    ]
    
    # Configure extractor
    config = {
        'extract_images': True,
        'extract_references': True,
        'max_summary_length': 300,
        'timeout': 10
    }
    
    extractor = WikipediaExtractor(config)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\nTest {i}: Extracting from {url}")
        print("-" * 60)
        
        try:
            result = extractor.extract_from_url(url)
            
            if 'error' in result:
                print(f"✗ Error: {result['error']}")
            else:
                # Display key information
                metadata = result.get('metadata', {})
                content = result.get('content', {})
                statistics = result.get('statistics', {})
                
                print(f"✓ Success! Extracted content:")
                print(f"  Title: {metadata.get('title', 'N/A')}")
                print(f"  Language: {metadata.get('language', 'N/A')}")
                print(f"  Word Count: {statistics.get('word_count', 0)}")
                print(f"  Confidence: {statistics.get('extraction_confidence', 0):.2f}")
                
                # Show summary preview
                summary = content.get('summary', '')
                if summary:
                    preview = summary[:200] + '...' if len(summary) > 200 else summary
                    print(f"  Summary: {preview}")
                
                # Show structure info
                structure = result.get('structure', {})
                print(f"  Sections: {len(structure.get('sections', []))}")
                print(f"  Categories: {len(structure.get('categories', []))}")
                
                # Show media info
                media = result.get('media', {})
                print(f"  Images: {len(media.get('images', []))}")
                
                # Show links info
                links = result.get('links', {})
                print(f"  Internal Links: {len(links.get('internal', []))}")
                
                # Save result for inspection
                output_file = f"wikipedia_test_result_{i}.json"
                extractor.save_extraction_result(result, output_file)
                print(f"  Saved detailed result to: {output_file}")
                
        except Exception as e:
            print(f"✗ Exception during extraction: {e}")

def test_url_parsing():
    """Test URL parsing functionality."""
    print("\n=== Testing URL Parsing ===")
    
    test_urls = [
        "https://en.wikipedia.org/wiki/Article_Title",
        "https://en.wikipedia.org/w/index.php?title=Article_Title&printable=yes",
        "https://de.wikipedia.org/wiki/Artikel_Titel",
        "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
        "https://en.wikipedia.org/wiki/Machine_learning#Applications",
        "https://www.wikipedia.org/"  # Should fail
    ]
    
    extractor = WikipediaExtractor()
    
    for url in test_urls:
        print(f"\nParsing: {url}")
        parsed = extractor._parse_wikipedia_url(url)
        if parsed:
            page_title, language, wiki_site = parsed
            print(f"  ✓ Parsed: Title='{page_title}', Language='{language}', Site='{wiki_site}'")
        else:
            print(f"  ✗ Failed to parse")

def test_database_integration():
    """Test integration with ContextBox database."""
    print("\n=== Testing Database Integration ===")
    
    try:
        # Initialize database
        db = ContextDatabase({'db_path': ':memory:'})
        print("✓ Database initialized")
        
        # Force database initialization to ensure tables are created
        import sqlite3
        with db._get_connection() as conn:
            db._create_captures_table(conn)
            db._create_artifacts_table(conn)
            db._create_indexes(conn)
            conn.commit()
        print("✓ Database tables initialized")
        
        # Create a test capture
        capture_id = db.create_capture(
            source_window="Wikipedia Test",
            notes="Test capture for Wikipedia extraction"
        )
        print(f"✓ Created capture with ID: {capture_id}")
        
        # Test with a simple Wikipedia URL
        url = "https://en.wikipedia.org/wiki/Quantum_computing"
        result = extract_wikipedia_content(url)
        
        if 'error' not in result:
            # Create artifact data
            from contextbox.extractors.wikipedia import create_wikipedia_artifact_data, create_summary_artifact_data
            
            main_artifact = create_wikipedia_artifact_data(result, capture_id)
            summary_artifact = create_summary_artifact_data(result, capture_id)
            
            # Store main artifact
            artifact_id = db.create_artifact(
                capture_id=capture_id,
                kind=main_artifact['kind'],
                url=main_artifact['url'],
                title=main_artifact['title'],
                text=main_artifact['text'],
                metadata=main_artifact['metadata']
            )
            print(f"✓ Stored main artifact with ID: {artifact_id}")
            
            # Store summary artifact if available
            if summary_artifact:
                summary_id = db.create_artifact(
                    capture_id=capture_id,
                    kind=summary_artifact['kind'],
                    url=summary_artifact['url'],
                    title=summary_artifact['title'],
                    text=summary_artifact['text'],
                    metadata=summary_artifact['metadata']
                )
                print(f"✓ Stored summary artifact with ID: {summary_id}")
            
            # Verify storage
            artifacts = db.get_artifacts_by_capture(capture_id)
            print(f"✓ Retrieved {len(artifacts)} artifacts from database")
            
            # Show artifact info
            for artifact in artifacts:
                print(f"  - Artifact {artifact['id']}: {artifact['kind']} - {artifact['title'][:50]}...")
                
        else:
            print(f"✗ Wikipedia extraction failed: {result['error']}")
        
        db.cleanup()
        
    except Exception as e:
        print(f"✗ Database integration test failed: {e}")

def test_configuration_options():
    """Test different configuration options."""
    print("\n=== Testing Configuration Options ===")
    
    # Test with minimal config
    print("\nTest 1: Minimal configuration")
    config1 = {}
    extractor1 = WikipediaExtractor(config1)
    print(f"✓ Created extractor with minimal config")
    
    # Test with full config
    print("\nTest 2: Full configuration")
    config2 = {
        'timeout': 15,
        'max_retries': 5,
        'api_language': 'en',
        'extract_images': False,
        'extract_references': False,
        'max_summary_length': 200
    }
    extractor2 = WikipediaExtractor(config2)
    print(f"✓ Created extractor with full config")
    
    # Test feature flags
    print("\nTest 3: Feature availability")
    from contextbox.extractors import WIKIPEDIA_AVAILABLE
    print(f"Wikipedia extraction available: {WIKIPEDIA_AVAILABLE}")

def main():
    """Main test function."""
    print("ContextBox Wikipedia Extraction Test Suite")
    print("=" * 50)
    
    # Check dependencies
    try:
        import requests
        print("✓ requests library available")
    except ImportError:
        print("✗ requests library not available")
        return
    
    try:
        from bs4 import BeautifulSoup
        print("✓ BeautifulSoup4 available")
    except ImportError:
        print("✗ BeautifulSoup4 not available")
        return
    
    # Run tests
    test_url_parsing()
    test_configuration_options()
    test_wikipedia_extraction()
    test_database_integration()
    
    print("\n" + "=" * 50)
    print("Wikipedia extraction test suite completed!")

if __name__ == "__main__":
    main()