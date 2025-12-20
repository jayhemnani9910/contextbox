"""
Enhanced Content Extractors

This module provides the main content extraction orchestrator that integrates all extraction
modules with the ContextBox application. It manages the workflow for automatic content
extraction when URLs are found during capture, and provides manual content extraction
capabilities through CLI and API interfaces.

Key Features:
- Automatic content extraction during context capture
- Manual content extraction via CLI and API
- Configuration-driven extractor selection
- Comprehensive error handling and fallback strategies
- Rich output formatting for extracted content
- Database integration for storing extracted content
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path
import hashlib

from contextbox.extractors import (
    URLExtractor,
    OCRExtractor, 
    TextProcessor,
    EnhancedContextExtractor,
    ExtractedURL,
    OCRResult,
    PIL_AVAILABLE,
    TESSERACT_AVAILABLE,
    CLIPBOARD_AVAILABLE
)


class ContentExtractionError(Exception):
    """Custom exception for content extraction operations."""
    pass


class ContentExtractor:
    """
    Main content extraction orchestrator for ContextBox.
    
    This class provides the interface between the main ContextBox application and the
    various content extraction modules. It handles automatic and manual content extraction,
    configuration management, error handling, and database integration.
    
    Features:
    - Automatic extraction when URLs are found during capture
    - Manual extraction via CLI and API
    - Configurable extractor selection
    - Rich output formatting
    - Database storage integration
    - Comprehensive error handling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the content extractor.
        
        Args:
            config: Configuration dictionary for extractors and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize main extractor
        self.extractor = EnhancedContextExtractor(self.config.get('extractors', {}))
        
        # Configure settings
        self.auto_extract = self.config.get('auto_extract', True)
        self.output_format = self.config.get('output_format', 'json')
        self.store_in_database = self.config.get('store_in_database', True)
        self.enabled_extractors = self.config.get('enabled_extractors', [
            'text_extraction', 'system_extraction', 'network_extraction'
        ])
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Feature availability
        self.features = {
            'ocr': TESSERACT_AVAILABLE and PIL_AVAILABLE,
            'clipboard': CLIPBOARD_AVAILABLE,
            'database': self.store_in_database,
            'url_extraction': True
        }
        
        self.logger.info(f"ContentExtractor initialized with features: {self.features}")
        
        if not self.features['ocr'] and self.config.get('extract_images', True):
            self.logger.warning(
                "OCR dependencies not available (Pillow and Tesseract required). "
                "Image extraction will be disabled."
            )
        
        if not self.features['clipboard'] and self.config.get('extract_urls', True):
            self.logger.info(
                "Clipboard integration unavailable; install 'pyperclip' for clipboard URL extraction."
            )
    
    def extract_from_capture(self, capture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from a context capture automatically.
        
        This method is called when URLs are found during context capture to automatically
        extract and process content from those URLs.
        
        Args:
            capture_data: Context capture data containing URLs and other information
            
        Returns:
            Extracted content with rich metadata and formatting
        """
        self.logger.info("Starting automatic content extraction from capture")
        
        extraction_result = {
            'type': 'automatic_content_extraction',
            'timestamp': datetime.now().isoformat(),
            'extraction_id': self._generate_extraction_id(),
            'source_capture': self._sanitize_capture_data(capture_data),
            'extracted_content': [],
            'url_analysis': {},
            'metadata': {
                'extraction_method': 'automatic',
                'features_used': self._get_enabled_features(capture_data),
                'dependencies_status': self.features,
                'total_urls_found': 0,
                'successful_extractions': 0,
                'failed_extractions': 0
            }
        }
        
        try:
            # Run main context extraction first
            if self.enabled_extractors:
                context_result = self.extractor.extract(capture_data)
                extraction_result['context_extraction'] = context_result
            
            # Extract and analyze URLs
            urls = self._extract_urls_from_capture(capture_data)
            extraction_result['metadata']['total_urls_found'] = len(urls)
            
            if urls:
                self.logger.info(f"Found {len(urls)} URLs for content extraction")
                
                # Analyze URLs
                url_analysis = self._analyze_urls(urls)
                extraction_result['url_analysis'] = url_analysis
                
                # Attempt content extraction from URLs
                if self.auto_extract:
                    extracted_content = self._extract_content_from_urls(urls)
                    extraction_result['extracted_content'] = extracted_content
                    
                    # Update success/failure counts
                    for content in extracted_content:
                        if content.get('success', False):
                            extraction_result['metadata']['successful_extractions'] += 1
                        else:
                            extraction_result['metadata']['failed_extractions'] += 1
            
            # Add summary
            extraction_result['summary'] = self._create_extraction_summary(extraction_result)
            
            self.logger.info(f"Content extraction completed: {extraction_result['metadata']['successful_extractions']} successful, {extraction_result['metadata']['failed_extractions']} failed")
            
        except Exception as e:
            self.logger.error(f"Content extraction failed: {e}")
            extraction_result['extraction_error'] = str(e)
            extraction_result['metadata']['extraction_error'] = True
        
        return extraction_result
    
    def extract_content_manually(self, 
                                input_data: Dict[str, Any],
                                extract_urls: bool = True,
                                extract_text: bool = True,
                                extract_images: bool = True) -> Dict[str, Any]:
        """
        Manually extract content from provided data.
        
        This method provides manual content extraction capabilities for CLI and API use.
        
        Args:
            input_data: Input data for extraction (text, images, URLs, etc.)
            extract_urls: Whether to extract URLs from the data
            extract_text: Whether to extract text content
            extract_images: Whether to extract and process images
            
        Returns:
            Rich extraction results with formatting and metadata
        """
        self.logger.info("Starting manual content extraction")
        
        # Validate input data
        if not input_data or (isinstance(input_data, dict) and not any(input_data.values())):
            raise ContentExtractionError("No input data provided for extraction")
        
        manual_result = {
            'type': 'manual_content_extraction',
            'timestamp': datetime.now().isoformat(),
            'extraction_id': self._generate_extraction_id(),
            'request_params': {
                'extract_urls': extract_urls,
                'extract_text': extract_text,
                'extract_images': extract_images
            },
            'input_analysis': {},
            'extracted_content': {},
            'formatted_output': {},
            'metadata': {
                'extraction_method': 'manual',
                'processing_time': None,
                'dependencies_status': self.features,
                'input_size': self._calculate_input_size(input_data)
            }
        }
        
        start_time = datetime.now()
        
        try:
            # Analyze input data
            input_analysis = self._analyze_input_data(input_data)
            manual_result['input_analysis'] = input_analysis
            
            # Extract URLs if requested
            if extract_urls:
                urls = self._extract_urls_from_input(input_data)
                manual_result['extracted_content']['urls'] = {
                    'direct_urls': [url.__dict__ for url in urls['direct_urls']],
                    'inferred_domains': [domain.__dict__ for domain in urls['inferred_domains']],
                    'clipboard_urls': [url.__dict__ for url in urls['clipboard_urls']],
                    'total_found': urls['total_count']
                }
            
            # Extract text if requested
            if extract_text:
                text_content = self._extract_text_content(input_data)
                manual_result['extracted_content']['text'] = text_content
            
            # Extract and process images if requested
            if extract_images and self.features['ocr']:
                image_content = self._extract_image_content(input_data)
                manual_result['extracted_content']['images'] = image_content
            
            # Generate formatted output
            manual_result['formatted_output'] = self._format_extraction_output(manual_result['extracted_content'])
            
            # Calculate processing time
            end_time = datetime.now()
            manual_result['metadata']['processing_time'] = str(end_time - start_time)
            
            # Create summary
            manual_result['summary'] = self._create_manual_summary(manual_result)
            
            self.logger.info(f"Manual content extraction completed in {manual_result['metadata']['processing_time']}")
            
        except Exception as e:
            self.logger.error(f"Manual content extraction failed: {e}")
            manual_result['extraction_error'] = str(e)
            manual_result['metadata']['extraction_error'] = True
        
        return manual_result
    
    def extract_urls_only(self, text: str, include_clipboard: bool = True) -> List[ExtractedURL]:
        """
        Extract only URLs from text with optional clipboard integration.
        
        Args:
            text: Text to extract URLs from
            include_clipboard: Whether to also check clipboard for URLs
            
        Returns:
            List of ExtractedURL objects
        """
        urls = URLExtractor.extract_urls(text)
        
        if include_clipboard and CLIPBOARD_AVAILABLE:
            try:
                clipboard_urls = URLExtractor.extract_from_clipboard()
                urls.extend(clipboard_urls)
            except Exception as e:
                self.logger.debug(f"Clipboard URL extraction failed: {e}")
        
        # Remove duplicates
        unique_urls = {}
        for url in urls:
            if url.normalized_url not in unique_urls:
                unique_urls[url.normalized_url] = url
        
        return list(unique_urls.values())
    
    def extract_ocr_only(self, image_path: str, config: Optional[Dict[str, Any]] = None) -> OCRResult:
        """
        Extract only OCR text from image.
        
        Args:
            image_path: Path to image file
            config: OCR configuration options
            
        Returns:
            OCRResult object
        """
        if not self.features['ocr']:
            raise ContentExtractionError("OCR functionality not available")
        
        return OCRExtractor.extract_text_from_image(image_path, config)
    
    def format_extraction_result(self, result: Dict[str, Any], format_type: str = 'json') -> str:
        """
        Format extraction result in the specified format.
        
        Args:
            result: Extraction result dictionary
            format_type: Output format ('json', 'pretty', 'summary', 'detailed')
            
        Returns:
            Formatted result string
        """
        if format_type == 'json':
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif format_type == 'pretty':
            return self._format_pretty(result)
        
        elif format_type == 'summary':
            return self._format_summary(result)
        
        elif format_type == 'detailed':
            return self._format_detailed(result)
        
        else:
            raise ContentExtractionError(f"Unknown format type: {format_type}")
    
    def save_extraction_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save extraction result to file.
        
        Args:
            result: Extraction result dictionary
            output_path: Path to save the result
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Extraction result saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save extraction result: {e}")
            raise ContentExtractionError(f"Failed to save result: {e}")
    
    def get_extraction_config(self) -> Dict[str, Any]:
        """
        Get current extraction configuration.
        
        Returns:
            Current configuration dictionary
        """
        return {
            'auto_extract': self.auto_extract,
            'output_format': self.output_format,
            'store_in_database': self.store_in_database,
            'enabled_extractors': self.enabled_extractors,
            'features': self.features,
            'dependencies': {
                'ocr_available': self.features['ocr'],
                'clipboard_available': self.features['clipboard']
            }
        }
    
    def update_extraction_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update extraction configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        for key, value in config_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value
        
        self.logger.info(f"Updated extraction configuration: {config_updates}")
    
    def _extract_urls_from_capture(self, capture_data: Dict[str, Any]) -> List[ExtractedURL]:
        """Extract URLs from capture data."""
        all_text = ""
        
        # Collect text from various sources
        if 'text' in capture_data and capture_data['text']:
            all_text += capture_data['text'] + " "
        
        if 'active_window' in capture_data:
            window_data = capture_data['active_window']
            if 'title' in window_data:
                all_text += window_data['title'] + " "
            if 'application' in window_data:
                all_text += window_data['application'] + " "
        
        # OCR from screenshot if available
        if 'screenshot_path' in capture_data and self.features['ocr']:
            try:
                ocr_result = OCRExtractor.extract_text_from_image(capture_data['screenshot_path'])
                all_text += ocr_result.text + " "
            except Exception as e:
                self.logger.debug(f"OCR extraction failed: {e}")
        
        if all_text.strip():
            return URLExtractor.extract_urls(all_text)
        
        return []
    
    def _analyze_urls(self, urls: List[ExtractedURL]) -> Dict[str, Any]:
        """Analyze extracted URLs for patterns and metadata."""
        analysis = {
            'total_count': len(urls),
            'by_type': {},
            'by_domain': {},
            'high_confidence': [],
            'inferred_domains': []
        }
        
        for url in urls:
            # Count by type
            url_type = 'direct' if url.is_direct else 'inferred'
            analysis['by_type'][url_type] = analysis['by_type'].get(url_type, 0) + 1
            
            # Count by domain
            domain = url.domain
            analysis['by_domain'][domain] = analysis['by_domain'].get(domain, 0) + 1
            
            # High confidence URLs
            if url.confidence > 0.8:
                analysis['high_confidence'].append(url.normalized_url)
            
            # Inferred domains
            if not url.is_direct:
                analysis['inferred_domains'].append(url.normalized_url)
        
        return analysis
    
    def _extract_content_from_urls(self, urls: List[ExtractedURL]) -> List[Dict[str, Any]]:
        """Attempt to extract content from URLs."""
        extracted_content = []
        
        for url in urls:
            try:
                content_item = {
                    'url': url.normalized_url,
                    'domain': url.domain,
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': None,
                    'metadata': {
                        'confidence': url.confidence,
                        'is_direct': url.is_direct,
                        'context': url.context
                    }
                }
                
                # Basic URL validation and content analysis
                if self._is_accessible_url(url.normalized_url):
                    content_item['success'] = True
                    content_item['analysis'] = self._analyze_url_content(url.normalized_url)
                else:
                    content_item['error'] = 'URL not accessible or invalid'
                
                extracted_content.append(content_item)
                
            except Exception as e:
                self.logger.debug(f"Content extraction failed for URL {url.normalized_url}: {e}")
                extracted_content.append({
                    'url': url.normalized_url,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return extracted_content
    
    def _is_accessible_url(self, url: str) -> bool:
        """Basic URL accessibility check."""
        import urllib.parse
        
        try:
            parsed = urllib.parse.urlparse(url)
            return bool(parsed.scheme and parsed.netloc) and url.startswith(('http://', 'https://'))
        except Exception:
            return False
    
    def _analyze_url_content(self, url: str) -> Dict[str, Any]:
        """Analyze URL content (basic analysis without actual fetching)."""
        import urllib.parse
        
        parsed = urllib.parse.urlparse(url)
        
        analysis = {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'query': parsed.query,
            'url_type': self._classify_url_type(url)
        }
        
        return analysis
    
    def _classify_url_type(self, url: str) -> str:
        """Classify URL type based on domain and structure."""
        domain = url.lower()
        
        if any(social in domain for social in ['twitter.com', 'facebook.com', 'linkedin.com']):
            return 'social_media'
        elif any(search in domain for search in ['google.com', 'bing.com', 'yahoo.com']):
            return 'search_engine'
        elif any(dev in domain for dev in ['github.com', 'gitlab.com', 'stackoverflow.com']):
            return 'developer'
        elif any(news in domain for news in ['news', 'times', 'post', 'herald']):
            return 'news'
        elif any(store in domain for store in ['shop', 'store', 'amazon', 'ebay']):
            return 'ecommerce'
        else:
            return 'general'
    
    def _create_extraction_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of extraction results."""
        return {
            'total_urls': result['metadata']['total_urls_found'],
            'successful_extractions': result['metadata']['successful_extractions'],
            'failed_extractions': result['metadata']['failed_extractions'],
            'success_rate': self._calculate_success_rate(result['metadata']),
            'top_domains': self._get_top_domains(result),
            'url_types': result['url_analysis'].get('by_type', {}),
            'high_confidence_count': len(result['url_analysis'].get('high_confidence', [])),
            'extraction_method': result['metadata']['extraction_method']
        }
    
    def _sanitize_capture_data(self, capture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize capture data for storage (remove sensitive info)."""
        sanitized = {}
        
        allowed_keys = ['timestamp', 'platform', 'active_window', 'text', 'artifacts']
        
        for key, value in capture_data.items():
            if key in allowed_keys:
                sanitized[key] = value
        
        return sanitized
    
    def _get_enabled_features(self, data: Dict[str, Any]) -> List[str]:
        """Get enabled features based on input data and configuration."""
        features = []
        
        if 'text' in data:
            features.append('url_extraction')
            features.append('text_analysis')
        
        if ('screenshot_path' in data or 'image' in data) and self.features['ocr']:
            features.append('ocr_extraction')
        
        if self.features['clipboard']:
            features.append('clipboard_extraction')
        
        return features
    
    def _calculate_input_size(self, input_data: Dict[str, Any]) -> str:
        """Calculate input data size for logging."""
        try:
            data_str = json.dumps(input_data)
            size_bytes = len(data_str.encode('utf-8'))
            
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            return "unknown"
    
    def _analyze_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data structure and content."""
        analysis = {
            'data_types': [],
            'content_summary': {},
            'estimated_processing_time': 'fast'
        }
        
        for key, value in input_data.items():
            analysis['data_types'].append(key)
            
            if isinstance(value, str):
                analysis['content_summary'][key] = {
                    'type': 'text',
                    'length': len(value),
                    'has_urls': 'http' in value or 'www.' in value,
                    'has_email': '@' in value
                }
            elif isinstance(value, dict):
                analysis['content_summary'][key] = {
                    'type': 'dict',
                    'keys': list(value.keys())
                }
            else:
                analysis['content_summary'][key] = {
                    'type': str(type(value).__name__),
                    'str_length': len(str(value)) if value else 0
                }
        
        return analysis
    
    def _extract_urls_from_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract URLs from input data."""
        all_text = ""
        
        for key, value in input_data.items():
            if isinstance(value, str):
                all_text += value + " "
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        all_text += sub_value + " "
        
        # Extract different types of URLs
        direct_urls = URLExtractor.extract_urls(all_text)
        inferred_domains = URLExtractor.infer_domains(all_text)
        clipboard_urls = URLExtractor.extract_from_clipboard() if CLIPBOARD_AVAILABLE else []
        
        total_count = len(direct_urls) + len(inferred_domains) + len(clipboard_urls)
        
        return {
            'direct_urls': direct_urls,
            'inferred_domains': inferred_domains,
            'clipboard_urls': clipboard_urls,
            'total_count': total_count
        }
    
    def _extract_text_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and process text content."""
        all_text_parts = []
        
        for key, value in input_data.items():
            if isinstance(value, str) and value.strip():
                all_text_parts.append(value)
        
        combined_text = " ".join(all_text_parts)
        
        if not combined_text.strip():
            return {'found': False, 'reason': 'no_text_content'}
        
        # Process text
        cleaned_text = self.text_processor.clean_text(combined_text)
        
        return {
            'found': True,
            'original_length': len(combined_text),
            'cleaned_length': len(cleaned_text),
            'processed_text': cleaned_text,
            'word_count': len(cleaned_text.split()),
            'has_urls': 'http' in cleaned_text or 'www.' in cleaned_text,
            'has_emails': '@' in cleaned_text
        }
    
    def _extract_image_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and process image content."""
        images_found = []
        
        # Look for image paths or data
        for key, value in input_data.items():
            if 'image' in key.lower() or 'screenshot' in key.lower():
                if isinstance(value, str) and Path(value).exists():
                    images_found.append(value)
                elif isinstance(value, dict) and 'path' in value:
                    images_found.append(value['path'])
        
        if not images_found:
            return {'found': False, 'reason': 'no_images_found'}
        
        # Process images
        processed_images = []
        for image_path in images_found:
            try:
                ocr_result = OCRExtractor.extract_text_from_image(image_path)
                processed_images.append({
                    'path': image_path,
                    'ocr_success': bool(ocr_result.text.strip()),
                    'text_length': len(ocr_result.text),
                    'confidence': ocr_result.confidence,
                    'extracted_text': ocr_result.text[:500] + "..." if len(ocr_result.text) > 500 else ocr_result.text
                })
            except Exception as e:
                processed_images.append({
                    'path': image_path,
                    'ocr_success': False,
                    'error': str(e)
                })
        
        return {
            'found': True,
            'images_processed': len(processed_images),
            'successful_ocr': len([img for img in processed_images if img.get('ocr_success')]),
            'image_results': processed_images
        }
    
    def _format_extraction_output(self, extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """Format extracted content for output."""
        formatted = {
            'summary': {},
            'detailed_results': {},
            'formatted_text': {}
        }
        
        # URL summary
        if 'urls' in extracted_content:
            urls_data = extracted_content['urls']
            formatted['summary']['urls'] = {
                'total_found': urls_data['total_found'],
                'direct_urls': len(urls_data['direct_urls']),
                'inferred_domains': len(urls_data['inferred_domains'])
            }
        
        # Text summary
        if 'text' in extracted_content:
            text_data = extracted_content['text']
            if text_data.get('found'):
                formatted['summary']['text'] = {
                    'word_count': text_data['word_count'],
                    'contains_urls': text_data['has_urls'],
                    'contains_emails': text_data['has_emails']
                }
                formatted['formatted_text'] = {
                    'preview': text_data['processed_text'][:200] + "..." if len(text_data['processed_text']) > 200 else text_data['processed_text']
                }
        
        # Image summary
        if 'images' in extracted_content:
            images_data = extracted_content['images']
            if images_data.get('found'):
                formatted['summary']['images'] = {
                    'processed': images_data['images_processed'],
                    'successful_ocr': images_data['successful_ocr']
                }
        
        return formatted
    
    def _create_manual_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for manual extraction results."""
        summary = {
            'extraction_id': result['extraction_id'],
            'processing_time': result['metadata']['processing_time'],
            'input_size': result['metadata']['input_size'],
            'features_enabled': [
                feature for feature, enabled in {
                    'url_extraction': result['request_params']['extract_urls'],
                    'text_extraction': result['request_params']['extract_text'],
                    'image_extraction': result['request_params']['extract_images']
                }.items() if enabled
            ]
        }
        
        # Add counts
        extracted_content = result['extracted_content']
        
        if 'urls' in extracted_content:
            summary['urls_found'] = extracted_content['urls']['total_found']
        
        if 'text' in extracted_content:
            text_data = extracted_content['text']
            if text_data.get('found'):
                summary['text_words'] = text_data['word_count']
        
        if 'images' in extracted_content:
            images_data = extracted_content['images']
            if images_data.get('found'):
                summary['images_processed'] = images_data['images_processed']
                summary['successful_ocr'] = images_data['successful_ocr']
        
        return summary
    
    def _generate_extraction_id(self) -> str:
        """Generate unique extraction ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _calculate_success_rate(self, metadata: Dict[str, Any]) -> float:
        """Calculate success rate for extraction."""
        successful = metadata.get('successful_extractions', 0)
        total = metadata.get('total_urls_found', 0)
        
        if total == 0:
            return 0.0
        
        return (successful / total) * 100
    
    def _get_top_domains(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top domains from URL analysis."""
        if 'url_analysis' not in result or 'by_domain' not in result['url_analysis']:
            return []
        
        domains = result['url_analysis']['by_domain']
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        
        return [{'domain': domain, 'count': count} for domain, count in sorted_domains[:5]]
    
    def _format_pretty(self, result: Dict[str, Any]) -> str:
        """Format result in pretty text format."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CONTENT EXTRACTION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Type: {result.get('type', 'Unknown').upper()}")
        lines.append(f"Timestamp: {result['timestamp']}")
        lines.append(f"Extraction ID: {result['extraction_id']}")
        
        if 'summary' in result:
            summary = result['summary']
            lines.append("\n--- SUMMARY ---")
            for key, value in summary.items():
                lines.append(f"{key}: {value}")
        
        if 'url_analysis' in result and result['url_analysis']:
            analysis = result['url_analysis']
            lines.append(f"\n--- URL ANALYSIS ---")
            lines.append(f"Total URLs: {analysis.get('total_count', 0)}")
            
            if analysis.get('high_confidence'):
                lines.append(f"High confidence URLs: {len(analysis['high_confidence'])}")
            
            if analysis.get('by_domain'):
                lines.append("Top domains:")
                for domain, count in analysis['by_domain'].items():
                    lines.append(f"  {domain}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _format_summary(self, result: Dict[str, Any]) -> str:
        """Format result as concise summary."""
        lines = []
        
        if 'summary' in result:
            summary = result['summary']
            lines.append(f"Extraction: {summary.get('total_urls', 0)} URLs, "
                        f"{summary.get('successful_extractions', 0)} successful")
        
        return " ".join(lines)
    
    def _format_detailed(self, result: Dict[str, Any]) -> str:
        """Format result in detailed format."""
        formatted = json.dumps(result, indent=2, ensure_ascii=False)
        return formatted
