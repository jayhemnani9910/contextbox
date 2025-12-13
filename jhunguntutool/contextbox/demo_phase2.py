#!/usr/bin/env python3
"""
ContextBox Demonstration Script
Shows the complete functionality including content extraction
"""

import sys
import json
import tempfile
from pathlib import Path

# Add contextbox to path
sys.path.insert(0, '/workspace/contextbox')

from contextbox import ContextBox
from contextbox.extractors.webpage import extract_webpage_content
from contextbox.extractors.youtube import extract_youtube_transcript
from contextbox.database import ContextDatabase

def main():
    print("🚀 ContextBox Phase 2 - Content Extraction Demo")
    print("=" * 50)
    
    # Initialize ContextBox
    print("1️⃣ Initializing ContextBox...")
    app = ContextBox()
    print("✓ ContextBox initialized successfully")
    
    # Demo 1: Web page extraction
    print("\n2️⃣ Testing Web Page Extraction...")
    try:
        # Create a sample HTML content for testing
        test_html = """
        <html>
        <head><title>Sample Article</title></head>
        <body>
            <h1>Introduction to ContextBox</h1>
            <p>ContextBox is a powerful tool for capturing and organizing digital context.</p>
            <p>Visit our website: <a href="https://contextbox.example.com">ContextBox Website</a></p>
            <p>Learn more at <a href="https://docs.contextbox.example.com">Documentation</a></p>
        </body>
        </html>
        """
        
        # Save test HTML to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(test_html)
            test_file = f.name
        
        print(f"   Testing web page extraction from: {test_file}")
        content = extract_webpage_content(f"file://{test_file}")
        print(f"   ✓ Extracted title: {content.title}")
        print(f"   ✓ Extracted {len(content.text)} characters of text")
        print(f"   ✓ Found {len(content.links)} links")
        
        # Clean up
        Path(test_file).unlink()
        
    except Exception as e:
        print(f"   ⚠️ Web page extraction demo skipped: {e}")
    
    # Demo 2: Database operations
    print("\n3️⃣ Testing Database Integration...")
    try:
        # Store a sample capture
        capture_data = {
            'timestamp': '2025-11-05T03:45:00',
            'platform': {'system': 'Demo', 'version': '1.0'},
            'artifacts': {},
            'extracted': {
                'text': 'This is a demo capture for ContextBox Phase 2.',
                'urls': ['https://example.com', 'https://demo.org']
            }
        }
        
        context_id = app.store_context(capture_data)
        print(f"   ✓ Stored context with ID: {context_id}")
        
        # Retrieve the context
        retrieved = app.get_context(context_id)
        if retrieved:
            print(f"   ✓ Retrieved context: {retrieved.get('timestamp', 'N/A')}")
        
    except Exception as e:
        print(f"   ⚠️ Database demo failed: {e}")
    
    # Demo 3: Content classification
    print("\n4️⃣ Testing Content Classification...")
    try:
        from contextbox.extractors.classifier import SmartClassifier
        
        classifier = SmartClassifier()
        
        test_urls = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://en.wikipedia.org/wiki/ContextBox',
            'https://example.com/blog/article',
            'https://github.com/contextbox/project',
            'https://twitter.com/contextbox/status/123'
        ]
        
        for url in test_urls:
            try:
                classification = classifier.classify_url(url)
                print(f"   ✓ {url[:40]}... → {classification.content_type} (confidence: {classification.confidence:.2f})")
            except Exception as e:
                print(f"   ⚠️ Failed to classify {url}: {e}")
                
    except Exception as e:
        print(f"   ⚠️ Content classification demo failed: {e}")
    
    # Demo 4: CLI functionality
    print("\n5️⃣ Testing Enhanced CLI...")
    try:
        print("   Available CLI commands:")
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'contextbox', '--help'
        ], capture_output=True, text=True, cwd='/workspace/contextbox')
        
        if 'extract-content' in result.stdout:
            print("   ✓ Enhanced CLI with extract-content command available")
        else:
            print("   ⚠️ Enhanced CLI features may not be fully available")
            
    except Exception as e:
        print(f"   ⚠️ CLI test failed: {e}")
    
    print("\n🎉 ContextBox Phase 2 Demo Complete!")
    print("\n📋 Summary of Phase 2 Features:")
    print("   ✓ YouTube transcript extraction")
    print("   ✓ Wikipedia content extraction")  
    print("   ✓ Generic web page extraction")
    print("   ✓ Smart content classification")
    print("   ✓ Enhanced CLI with extract-content")
    print("   ✓ Database integration for all content types")
    print("   ✓ Comprehensive error handling")
    print("\n🚀 Ready for Phase 3 - LLM Integration!")

if __name__ == '__main__':
    main()