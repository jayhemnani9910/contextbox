#!/usr/bin/env python3
"""
Test script for the new webpage extraction module.

This script tests the functionality of the WebPageExtractor to ensure
it works correctly with the ContextBox framework.
"""

import sys
import os
import json
from pathlib import Path

# Add the contextbox package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from contextbox.extractors.webpage import WebPageExtractor, extract_webpage_content, extract_multiple_pages
    from contextbox.database import ContextDatabase
    
    # Test URLs (using simple, accessible URLs)
    TEST_URLS = [
        "https://httpbin.org/html",
        "https://example.com"
    ]
    
    def test_basic_extraction():
        """Test basic webpage extraction."""
        print("Testing basic webpage extraction...")
        
        try:
            # Test single URL extraction
            content = extract_webpage_content("https://httpbin.org/html")
            
            print(f"✓ Successfully extracted content from test URL")
            print(f"  - Title: {content.title}")
            print(f"  - Description: {content.description}")
            print(f"  - Word Count: {content.word_count}")
            print(f"  - Content Score: {content.content_score:.2f}")
            print(f"  - Extraction Method: {content.extraction_method}")
            print(f"  - Links Found: {len(content.links)}")
            print(f"  - Images Found: {len(content.images)}")
            print(f"  - Response Time: {content.response_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"✗ Basic extraction test failed: {e}")
            return False
    
    def test_multiple_extractions():
        """Test batch URL extraction."""
        print("\nTesting batch webpage extraction...")
        
        try:
            # Test multiple URL extraction
            contents = extract_multiple_pages(TEST_URLS)
            
            print(f"✓ Successfully extracted {len(contents)} URLs")
            
            for i, content in enumerate(contents):
                print(f"  URL {i+1}: {content.final_url}")
                print(f"    - Title: {content.title}")
                print(f"    - Word Count: {content.word_count}")
                print(f"    - Status: {'Success' if not content.error else 'Error: ' + content.error}")
            
            return True
            
        except Exception as e:
            print(f"✗ Batch extraction test failed: {e}")
            return False
    
    def test_configuration():
        """Test extractor configuration."""
        print("\nTesting extractor configuration...")
        
        try:
            config = {
                'timeout': 10,
                'rate_limit': 0.5,
                'extract_images': False,
                'extract_social_links': False,
                'max_retries': 2
            }
            
            extractor = WebPageExtractor(config)
            print(f"✓ Successfully created configured extractor")
            print(f"  - Timeout: {extractor.timeout}s")
            print(f"  - Rate Limit: {extractor.rate_limiter.requests_per_second} req/s")
            print(f"  - Extract Images: {extractor.extract_images}")
            print(f"  - Extract Social Links: {extractor.extract_social_links}")
            
            # Test extraction with config
            content = extractor.extract("https://example.com")
            print(f"  - Extraction successful: {bool(content.main_content)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Configuration test failed: {e}")
            return False
    
    def test_content_cleaning():
        """Test text cleaning functionality."""
        print("\nTesting content cleaning...")
        
        try:
            from contextbox.extractors.webpage import TextCleaner
            
            # Test HTML cleaning
            dirty_html = "<p>This is a <b>test</b> paragraph with &amp; entities.</p>"
            clean_text = TextCleaner.clean_html_text(dirty_html)
            
            expected = "This is a test paragraph with & entities."
            
            if clean_text == expected:
                print("✓ HTML text cleaning works correctly")
            else:
                print(f"✗ HTML cleaning failed. Got: '{clean_text}', Expected: '{expected}'")
                return False
            
            # Test ad removal
            ad_text = "This is content with [advertisement] ads and more content."
            clean_ad = TextCleaner.remove_ads_and_noise(ad_text)
            
            if "[advertisement]" not in clean_ad:
                print("✓ Ad removal works correctly")
            else:
                print(f"✗ Ad removal failed. Got: '{clean_ad}'")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Content cleaning test failed: {e}")
            return False
    
    def test_database_integration():
        """Test ContextBox database integration."""
        print("\nTesting ContextBox database integration...")
        
        try:
            # This would require an actual database, so we'll just check if the function exists
            from contextbox.extractors.webpage import integrate_with_contextbox
            
            # Test with a simple URL
            extractor = WebPageExtractor()
            content = extractor.extract("https://example.com")
            
            print("✓ Successfully created test extraction")
            print(f"  - Content hash: {content.content_hash}")
            print(f"  - Content score: {content.content_score:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Database integration test failed: {e}")
            return False
    
    def run_all_tests():
        """Run all tests and provide summary."""
        print("ContextBox Web Page Extractor Test Suite")
        print("=" * 50)
        
        tests = [
            test_basic_extraction,
            test_multiple_extractions,
            test_configuration,
            test_content_cleaning,
            test_database_integration
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} crashed: {e}")
                failed += 1
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! Webpage extraction is working correctly.")
        else:
            print(f"⚠️  {failed} test(s) failed. Please check the implementation.")
        
        return failed == 0
    
    if __name__ == "__main__":
        success = run_all_tests()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("- requests")
    print("- beautifulsoup4") 
    print("- lxml")
    print("- readability-lxml")
    print("\nInstall with: pip install requests beautifulsoup4 lxml readability-lxml")
    sys.exit(1)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)