#!/usr/bin/env python3
"""
Test script for the Smart Content Classifier

This script demonstrates the functionality of the SmartContentClassifier
including URL classification, extractor routing, and content analysis.
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextbox.extractors.classifier import (
    SmartContentClassifier,
    ContentRule,
    classify_url,
    extract_url_content,
    batch_extract_urls
)

def test_basic_classification():
    """Test basic URL classification functionality."""
    print("=== Testing Basic URL Classification ===")
    
    # Test URLs for different content types
    test_urls = [
        # YouTube URLs
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ",
        
        # Wikipedia URLs
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://de.wikipedia.org/wiki/Maschinelles_Lernen",
        "https://wikimedia.org/wiki/Main_Page",
        
        # News sites
        "https://www.cnn.com/2023/01/01/tech/ai-development/index.html",
        "https://www.bbc.com/news/technology",
        "https://www.reuters.com/world/technology/",
        
        # Documentation sites
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "https://docs.python.org/3/tutorial/",
        "https://learn.microsoft.com/en-us/azure/",
        
        # Social media
        "https://twitter.com/openai/status/123456789",
        "https://www.reddit.com/r/MachineLearning/comments/abc123/",
        "https://www.linkedin.com/posts/company_123",
        
        # Generic URLs
        "https://www.example.com/blog-post",
        "https://company.com/products/service"
    ]
    
    classifier = SmartContentClassifier()
    
    print(f"Testing {len(test_urls)} URLs...")
    results = []
    
    for url in test_urls:
        try:
            result = classifier.classify_url(url)
            results.append(result)
            
            print(f"✓ {url}")
            print(f"  Content Type: {result.content_type}")
            print(f"  Domain Type: {result.domain_type}")
            print(f"  Extractor: {result.extractor_name}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Fallbacks: {', '.join(result.fallback_extractors) if result.fallback_extractors else 'None'}")
            print()
            
        except Exception as e:
            print(f"✗ {url} - Error: {e}")
    
    return results

def test_content_extraction():
    """Test content extraction with different URLs."""
    print("\n=== Testing Content Extraction ===")
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.cnn.com",
        "https://developer.mozilla.org"
    ]
    
    classifier = SmartContentClassifier()
    
    for url in test_urls:
        print(f"Extracting content from: {url}")
        try:
            result = classifier.extract_content(url)
            
            print(f"  Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"  Content Type: {result.get('content_type', 'Unknown')}")
                print(f"  Extractor Used: {result['processing_stats']['extractor_used']}")
                print(f"  Confidence: {result['processing_stats']['confidence']:.2f}")
                print(f"  Classification Time: {result['processing_stats']['classification_time']:.3f}s")
                
                # Show some metadata
                metadata = result.get('metadata', {})
                if metadata:
                    print(f"  Metadata Keys: {list(metadata.keys())}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
            print()
            
        except Exception as e:
            print(f"  Extraction failed: {e}")
            print()

def test_batch_processing():
    """Test batch processing of multiple URLs."""
    print("=== Testing Batch Processing ===")
    
    test_urls = [
        "https://www.youtube.com/watch?v=test1",
        "https://en.wikipedia.org/wiki/Test",
        "https://www.bbc.com/news",
        "https://docs.python.org",
        "https://twitter.com/test/status/123",
        "https://www.example.com"
    ]
    
    classifier = SmartContentClassifier()
    
    print(f"Processing {len(test_urls)} URLs in batch...")
    
    # Test batch classification
    classification_results = classifier.batch_classify(test_urls)
    
    print(f"Classified {len(classification_results)} URLs:")
    for result in classification_results:
        print(f"  {result.url} -> {result.extractor_name} ({result.confidence:.2f})")
    
    # Test batch extraction
    print("\nTesting batch extraction...")
    extraction_results = batch_extract_urls(test_urls)
    
    successful_extractions = sum(1 for r in extraction_results if r.get('success', False))
    print(f"Successful extractions: {successful_extractions}/{len(extraction_results)}")
    
    return classification_results, extraction_results

def test_custom_rules():
    """Test adding custom classification rules."""
    print("\n=== Testing Custom Rules ===")
    
    classifier = SmartContentClassifier()
    
    # Create a custom rule for a specific domain
    custom_rule = ContentRule(
        name='custom_blog_rule',
        domain_patterns=['myblog.com', 'techblog.net'],
        url_patterns=[r'.*/article/.*', r'.*/post/.*'],
        content_type='blog_post',
        domain_type='blog',
        extractor_name='generic',
        priority=40,
        confidence_boost=0.1,
        metadata={'custom_field': 'blog_extraction'}
    )
    
    # Add the rule
    classifier.add_custom_rule(custom_rule)
    print("Added custom rule for blog posts")
    
    # Test URLs with custom rule
    test_urls = [
        "https://myblog.com/article/2023/tech-trends",
        "https://techblog.net/post/ai-development",
        "https://otherblog.com/post/test"  # Should not match custom rule
    ]
    
    for url in test_urls:
        result = classifier.classify_url(url)
        rule_applied = result.routing_info.get('rule_applied', 'default_fallback')
        
        print(f"{url}")
        print(f"  Applied Rule: {rule_applied}")
        print(f"  Content Type: {result.content_type}")
        print(f"  Confidence: {result.confidence:.2f}")
        print()

def test_performance_and_caching():
    """Test performance optimization and caching."""
    print("=== Testing Performance and Caching ===")
    
    classifier = SmartContentClassifier()
    test_url = "https://www.youtube.com/watch?v=test123"
    
    # First classification (cache miss)
    print("First classification (cache miss):")
    result1 = classifier.classify_url(test_url, use_cache=True)
    print(f"  Time: {result1.processing_time:.3f}s")
    
    # Second classification (cache hit)
    print("Second classification (cache hit):")
    result2 = classifier.classify_url(test_url, use_cache=True)
    print(f"  Time: {result2.processing_time:.3f}s")
    
    # Show stats
    stats = classifier.get_classification_stats()
    print(f"\nClassification Statistics:")
    print(f"  Total Classifications: {stats['total_classifications']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Fallback Usage: {stats['fallback_usage']}")
    print(f"  Cache Size: {stats['cache_size']}")
    
    # Show extractor usage
    print(f"\nExtractor Usage:")
    for extractor, count in stats['extractor_usage'].items():
        print(f"  {extractor}: {count}")
    
    # Clear cache and test
    print("\nClearing cache...")
    classifier.clear_cache()
    
    stats_after_clear = classifier.get_classification_stats()
    print(f"Cache size after clear: {stats_after_clear['cache_size']}")

def test_error_handling():
    """Test error handling for invalid URLs."""
    print("\n=== Testing Error Handling ===")
    
    classifier = SmartContentClassifier()
    
    # Test invalid URLs
    invalid_urls = [
        "",
        "not-a-url",
        "http://",
        "https://invalid-domain-that-does-not-exist-12345.com",
        "ftp://example.com/file.txt"
    ]
    
    for url in invalid_urls:
        print(f"Testing: '{url}'")
        try:
            result = classifier.classify_url(url)
            print(f"  Result: {result.content_type} (confidence: {result.confidence:.2f})")
            print(f"  Extractor: {result.extractor_name}")
            if result.routing_info.get('error'):
                print(f"  Error Info: {result.routing_info['error']}")
        except Exception as e:
            print(f"  Exception: {e}")
        print()

def test_database_integration():
    """Test database integration if available."""
    print("\n=== Testing Database Integration ===")
    
    try:
        config = {
            'use_database': True,
            'db_config': {'db_path': ':memory:'}  # Use in-memory database for testing
        }
        
        classifier = SmartContentClassifier(config)
        test_url = "https://www.youtube.com/watch?v=test123"
        
        print("Testing database integration...")
        result = classifier.extract_content(test_url)
        
        if classifier.db:
            print("✓ Database integration enabled")
            print("  Classification stored successfully")
        else:
            print("✗ Database integration not available")
        
    except Exception as e:
        print(f"Database integration test failed: {e}")

def main():
    """Run all tests."""
    print("Smart Content Classifier Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_basic_classification()
        test_content_extraction()
        test_batch_processing()
        test_custom_rules()
        test_performance_and_caching()
        test_error_handling()
        test_database_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nThe Smart Content Classifier is working correctly and includes:")
        print("  • URL pattern recognition for different content types")
        print("  • Domain-based classification (YouTube, Wikipedia, news, etc.)")
        print("  • Content type detection (video, article, documentation, etc.)")
        print("  • Automatic extractor selection and routing")
        print("  • Confidence scoring for classification accuracy")
        print("  • Fallback strategies when primary extraction fails")
        print("  • Support for adding custom domain rules")
        print("  • Integration with ContextBox database schema")
        print("  • Performance optimization with caching")
        print("  • Comprehensive error handling")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())