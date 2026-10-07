"""
Content Extractors Package

This package provides specialized content extraction capabilities for the ContextBox application,
including Wikipedia extraction, web page extraction, and other specialized extractors.

Note: Core extraction functionality is available in the main contextbox.extractors module.
"""

# Wikipedia extraction imports
try:
    from .wikipedia import (
        WikipediaExtractor,
        extract_wikipedia_content,
        extract_and_store_wikipedia,
        create_wikipedia_artifact_data,
        create_summary_artifact_data
    )
    WIKIPEDIA_AVAILABLE = True
except ImportError as e:
    # Dependencies not available
    WIKIPEDIA_AVAILABLE = False
    WikipediaExtractor = None
    extract_wikipedia_content = None
    extract_and_store_wikipedia = None
    create_wikipedia_artifact_data = None
    create_summary_artifact_data = None

# Web page extraction imports
try:
    from .webpage import (
        WebPageExtractor,
        WebPageContent,
        ExtractedLink,
        ExtractedImage,
        ExtractedMedia,
        TextCleaner,
        ContentAnalyzer,
        RateLimiter,
        extract_webpage_content,
        extract_multiple_pages,
        extract_and_store_in_contextbox,
        integrate_with_contextbox
    )
    WEBPAGE_AVAILABLE = True
except ImportError as e:
    # Dependencies not available
    WEBPAGE_AVAILABLE = False
    WebPageExtractor = None
    WebPageContent = None
    ExtractedLink = None
    ExtractedImage = None
    ExtractedMedia = None
    TextCleaner = None
    ContentAnalyzer = None
    RateLimiter = None
    extract_webpage_content = None
    extract_multiple_pages = None
    extract_and_store_in_contextbox = None
    integrate_with_contextbox = None

__all__ = [
    # Wikipedia extraction classes
    'WikipediaExtractor',
    
    # Convenience functions
    'extract_wikipedia_content',
    'extract_and_store_wikipedia',
    'create_wikipedia_artifact_data',
    'create_summary_artifact_data',
    
    # Web page extraction classes
    'WebPageExtractor',
    'WebPageContent',
    'ExtractedLink',
    'ExtractedImage',
    'ExtractedMedia',
    'TextCleaner',
    'ContentAnalyzer',
    'RateLimiter',
    
    # Additional convenience functions
    'extract_webpage_content',
    'extract_multiple_pages',
    'extract_and_store_in_contextbox',
    'integrate_with_contextbox',
    
    # Dependency flags
    'WIKIPEDIA_AVAILABLE',
    'WEBPAGE_AVAILABLE'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'ContextBox Team'
__description__ = 'Specialized content extraction capabilities for ContextBox'

# Feature availability summary
FEATURE_SUMMARY = {
    'wikipedia_extraction': WIKIPEDIA_AVAILABLE,
    'webpage_extraction': WEBPAGE_AVAILABLE
}

# Core extractors (the old sibling extractors.py) no longer exist; keep the names for callers
URLExtractor = OCRExtractor = TextProcessor = ExtractedURL = OCRResult = None
ContextExtractor = EnhancedContextExtractor = None
extract_text_from_image = extract_urls_from_text = extract_clipboard_urls = process_screenshot_and_extract_urls = None
PIL_AVAILABLE = TESSERACT_AVAILABLE = CLIPBOARD_AVAILABLE = False
