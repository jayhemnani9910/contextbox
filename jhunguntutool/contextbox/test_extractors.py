#!/usr/bin/env python3
"""
Test script for the enhanced contextbox extractors module.

This script demonstrates the OCR and URL extraction functionality.
"""

import os
import sys
from pathlib import Path

# Add the contextbox module to Python path
sys.path.insert(0, str(Path(__file__).parent))

from contextbox.extractors import (
    URLExtractor, OCRExtractor, TextProcessor, EnhancedContextExtractor,
    extract_text_from_image, extract_urls_from_text, extract_clipboard_urls,
    process_screenshot_and_extract_urls, PIL_AVAILABLE, TESSERACT_AVAILABLE, CLIPBOARD_AVAILABLE
)


def test_url_extraction():
    """Test URL extraction functionality."""
    print("🧪 Testing URL extraction...")
    
    test_texts = [
        "Visit https://www.example.com for more info",
        "Check out our website at github.com/user/repo",
        "Contact us at support@example.com",
        "FTP server available at ftp.example.com",
        "IP address: 192.168.1.1",
        "Protocol relative: //cdn.example.com/js/app.js",
        "Inferred domain: google.com",
        "Multiple URLs: https://site1.com and site2.org and user@email.net"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n  Test {i}: {text}")
        urls = URLExtractor.extract_urls(text)
        for url in urls:
            print(f"    ✓ {url.url} -> {url.normalized_url} (confidence: {url.confidence:.2f})")
        
        # Test domain inference
        domains = URLExtractor.infer_domains(text)
        for domain in domains:
            print(f"    🌐 Inferred: {domain.domain} -> {domain.normalized_url}")


def test_text_processing():
    """Test text processing functionality."""
    print("\n🧪 Testing text processing...")
    
    test_texts = [
        "This   is    spaced   out    text",
        "Multiple\n\n\n\nnewlines",
        "Special chars: @#$%^&*()[]{}|\\/<>!",
        "Mixed\r\nwhitespace\r\ncontent"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n  Test {i}: '{text}'")
        cleaned = TextProcessor.clean_text(text)
        print(f"    Cleaned: '{cleaned}'")


def test_ocr_functionality():
    """Test OCR functionality if available."""
    if not (PIL_AVAILABLE and TESSERACT_AVAILABLE):
        print("\n⚠️  OCR functionality not available (missing PIL or Tesseract)")
        print(f"    PIL available: {PIL_AVAILABLE}")
        print(f"    Tesseract available: {TESSERACT_AVAILABLE}")
        return
    
    print("\n🧪 Testing OCR functionality...")
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a test image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    text = "Hello World! Visit: example.com"
    draw.text((10, 30), text, fill='black', font=font)
    
    # Save test image
    test_image_path = "/tmp/test_ocr_image.png"
    img.save(test_image_path)
    
    # Test OCR extraction
    print(f"  Testing OCR on generated image: {test_image_path}")
    result = OCRExtractor.extract_text_from_image(test_image_path)
    
    print(f"    OCR Text: '{result.text}'")
    print(f"    Confidence: {result.confidence:.2f}%")
    
    # Test URL extraction from OCR text
    if result.text:
        urls = URLExtractor.extract_urls(result.text)
        print(f"    URLs found in OCR text: {len(urls)}")
        for url in urls:
            print(f"      ✓ {url.normalized_url}")
    
    # Test convenience function
    ocr_text = extract_text_from_image(test_image_path)
    print(f"    Convenience function result: '{ocr_text}'")
    
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)


def test_clipboard_functionality():
    """Test clipboard URL extraction if available."""
    if not CLIPBOARD_AVAILABLE:
        print("\n⚠️  Clipboard functionality not available (missing pyperclip)")
        return
    
    print("\n🧪 Testing clipboard URL extraction...")
    
    # Note: This test would require actual clipboard content
    # In a real scenario, you would copy text to clipboard first
    print("  Clipboard extraction requires actual clipboard content")
    print("  Use extract_clipboard_urls() in your application")
    
    # Test fallback extraction
    fallback_urls = URLExtractor._try_fallback_extraction()
    print(f"  Fallback extraction found: {len(fallback_urls)} URLs")


def test_full_integration():
    """Test full extractor integration."""
    print("\n🧪 Testing full context extraction...")
    
    # Create enhanced extractor
    config = {
        'ocr': {
            'enhance_contrast': True,
            'sharpen': True,
            'grayscale': True,
            'min_image_size': 300
        },
        'enabled_extractors': ['text', 'system', 'network']
    }
    
    extractor = EnhancedContextExtractor(config)
    
    # Test data
    test_data = {
        'text': 'Check out https://github.com and visit python.org for tutorials',
        'clipboard': 'Visit example.com for more info',
        'active_window': {
            'title': 'GitHub - user/project: Main page',
            'application': 'Web Browser'
        },
        'recent_files': ['/home/user/document.pdf', '/home/user/image.png'],
        'extract_clipboard_urls': True
    }
    
    print("  Running enhanced extraction...")
    result = extractor.extract(test_data)
    
    print(f"    Extraction ID: {result['metadata']['extraction_id']}")
    print(f"    Features used: {', '.join(result['metadata']['features_used'])}")
    print(f"    Total URLs found: {result['metadata']['total_urls_found']}")
    print(f"    OCR available: {result['metadata']['ocr_available']}")
    print(f"    Clipboard available: {result['metadata']['clipboard_available']}")
    
    # Show summary
    if 'summary' in result:
        summary = result['summary']
        print(f"    URLs found in summary: {summary['urls_found']}")
        print(f"    OCR performed: {summary['ocr_performed']}")
        print(f"    Clipboard extracted: {summary['clipboard_extracted']}")


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🧪 Testing convenience functions...")
    
    # Test URL extraction
    test_text = "Visit https://www.python.org and check docs.python.org"
    urls = extract_urls_from_text(test_text)
    print(f"  extract_urls_from_text(): {len(urls)} URLs found")
    for url in urls:
        print(f"    ✓ {url}")
    
    # Test process screenshot function
    if PIL_AVAILABLE and TESSERACT_AVAILABLE:
        print("  process_screenshot_and_extract_urls(): Creating test image...")
        
        # Create test image
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (300, 80), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text = "Visit: github.com/user/project"
        draw.text((10, 25), text, fill='black', font=font)
        
        test_image_path = "/tmp/test_screenshot.png"
        img.save(test_image_path)
        
        result = process_screenshot_and_extract_urls(test_image_path)
        print(f"    OCR text: '{result['ocr_text']}'")
        print(f"    Total findings: {result['total_findings']}")
        
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    else:
        print("  process_screenshot_and_extract_urls(): OCR dependencies not available")


def main():
    """Run all tests."""
    print("🚀 ContextBox Extractors Test Suite")
    print("=" * 50)
    
    print(f"\n📋 System Status:")
    print(f"    PIL (Pillow): {PIL_AVAILABLE}")
    print(f"    Tesseract OCR: {TESSERACT_AVAILABLE}")
    print(f"    Clipboard: {CLIPBOARD_AVAILABLE}")
    
    # Run tests
    test_url_extraction()
    test_text_processing()
    test_ocr_functionality()
    test_clipboard_functionality()
    test_convenience_functions()
    test_full_integration()
    
    print("\n✅ Test suite completed!")
    print("\n📚 Usage Examples:")
    print("  # Extract URLs from text")
    print("  urls = URLExtractor.extract_urls('Visit https://example.com')")
    print("  ")
    print("  # Extract text from image")
    print("  result = OCRExtractor.extract_text_from_image('screenshot.png')")
    print("  ")
    print("  # Full context extraction")
    print("  extractor = EnhancedContextExtractor(config)")
    print("  result = extractor.extract(data)")


if __name__ == "__main__":
    main()