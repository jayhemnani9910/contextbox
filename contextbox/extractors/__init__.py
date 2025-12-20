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

# For backwards compatibility with main extractors module
# Import core classes from the sibling extractors.py module (not this package)
try:
    # Import from the parent package's extractors module by accessing it via __import__
    _extractors_module = __import__('contextbox.extractors', fromlist=[''])
    # Get the actual module file, not this package's __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "contextbox_extractors_module",
        __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(__file__)), 'extractors.py')
    )
    if spec and spec.loader:
        _core_extractors = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_core_extractors)
        
        URLExtractor = getattr(_core_extractors, 'URLExtractor', None)
        OCRExtractor = getattr(_core_extractors, 'OCRExtractor', None)
        TextProcessor = getattr(_core_extractors, 'TextProcessor', None)
        ExtractedURL = getattr(_core_extractors, 'ExtractedURL', None)
        OCRResult = getattr(_core_extractors, 'OCRResult', None)
        ContextExtractor = getattr(_core_extractors, 'ContextExtractor', None)
        EnhancedContextExtractor = getattr(_core_extractors, 'EnhancedContextExtractor', None)
        extract_text_from_image = getattr(_core_extractors, 'extract_text_from_image', None)
        extract_urls_from_text = getattr(_core_extractors, 'extract_urls_from_text', None)
        extract_clipboard_urls = getattr(_core_extractors, 'extract_clipboard_urls', None)
        process_screenshot_and_extract_urls = getattr(_core_extractors, 'process_screenshot_and_extract_urls', None)
        PIL_AVAILABLE = getattr(_core_extractors, 'PIL_AVAILABLE', False)
        TESSERACT_AVAILABLE = getattr(_core_extractors, 'TESSERACT_AVAILABLE', False)
        CLIPBOARD_AVAILABLE = getattr(_core_extractors, 'CLIPBOARD_AVAILABLE', False)
        
        # Add to __all__ for convenience
        __all__.extend([
            'URLExtractor', 'OCRExtractor', 'TextProcessor', 'ExtractedURL', 'OCRResult',
            'ContextExtractor', 'EnhancedContextExtractor',
            'extract_text_from_image', 'extract_urls_from_text', 'extract_clipboard_urls',
            'process_screenshot_and_extract_urls',
            'PIL_AVAILABLE', 'TESSERACT_AVAILABLE', 'CLIPBOARD_AVAILABLE'
        ])
        
        # Update feature summary
        FEATURE_SUMMARY.update({
            'ocr': TESSERACT_AVAILABLE and PIL_AVAILABLE,
            'clipboard': CLIPBOARD_AVAILABLE,
            'url_extraction': True,
            'text_processing': True
        })
    else:
        raise ImportError("Could not load extractors module")
    
except (ImportError, Exception):
    # Core extractors not available
    URLExtractor = OCRExtractor = TextProcessor = ExtractedURL = OCRResult = None
    ContextExtractor = EnhancedContextExtractor = None
    extract_text_from_image = extract_urls_from_text = extract_clipboard_urls = process_screenshot_and_extract_urls = None
    PIL_AVAILABLE = TESSERACT_AVAILABLE = CLIPBOARD_AVAILABLE = False