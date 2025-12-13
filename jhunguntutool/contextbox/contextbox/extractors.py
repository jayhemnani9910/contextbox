"""
Context extraction module for processing captured context data.

This module provides comprehensive OCR text extraction and URL extraction
functionality, including Tesseract OCR integration, smart domain detection,
and clipboard URL extraction with fallback mechanisms.
"""

import logging
import re
import hashlib
import urllib.parse
import io
import json
import base64
from typing import Dict, Any, List, Optional, Type, Tuple, Set, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Optional dependencies
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    pyperclip = None

# Forward declaration to avoid circular reference
class EnhancedContextExtractor(ABC):
    pass

# Backward compatibility alias
ContextExtractor = EnhancedContextExtractor


@dataclass
class ExtractedURL:
    """Represents an extracted URL with metadata."""
    url: str
    normalized_url: str
    domain: str
    is_direct: bool = True
    confidence: float = 1.0
    context: Optional[str] = None
    position: Optional[Tuple[int, int]] = None


@dataclass
class OCRResult:
    """Represents OCR extraction result."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    words: Optional[List[Dict[str, Any]]] = None


class TextProcessor:
    """Text processing and cleaning utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        
        # Join with single newlines
        return '\n'.join(lines).strip()
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'[ \t]+', ' ', text)
    
    @staticmethod
    def remove_special_chars(text: str, keep_newlines: bool = True) -> str:
        """Remove special characters while optionally preserving newlines."""
        if keep_newlines:
            return re.sub(r'[^\w\s\-_.,;:!?()\[\]{}@#$%^&*+=|<>/\\"\'\n]', '', text)
        else:
            return re.sub(r'[^\w\s\-_.,;:!?()\[\]{}@#$%^&*+=|<>/\\"\']', '', text)


class URLExtractor:
    """Advanced URL extraction with pattern matching and normalization."""
    
    # Comprehensive URL patterns
    URL_PATTERNS = [
        # Standard HTTP/HTTPS URLs
        r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/[\w/_.]*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?',
        # www domains (add http:// prefix)
        r'(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/[^\s]*)?',
        # IP addresses
        r'(?:https?://)?(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
        # Email addresses (convert to mailto)
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # FTP URLs
        r'ftp://(?:[-\w.])+(?:\:[0-9]+)?(?:/[\w/_.]*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?',
        # File URLs
        r'file:///[^\s]+',
        # Protocol-relative URLs
        r'//(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/[^\s]*)?',
        # Domain with path (no protocol)
        r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/[\w/._~:/?#[\]@!$&\'()*+,;=]*)?',
    ]
    
    # Common TLDs for better domain detection
    COMMON_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'uk', 'de', 'fr', 'jp', 
        'cn', 'ru', 'ca', 'au', 'br', 'in', 'it', 'es', 'nl', 'se', 'no', 'fi', 'dk',
        'ch', 'at', 'be', 'pl', 'cz', 'pt', 'gr', 'hu', 'ie', 'lt', 'lv', 'ee', 'mt',
        'cy', 'lu', 'sk', 'si', 'is', 'li', 'mc', 'sm', 'va', 'xk', 'eu'
    }
    
    # Social media and common platform domains
    SOCIAL_DOMAINS = {
        'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com', 'youtube.com',
        'github.com', 'gitlab.com', 'bitbucket.org', 'stackoverflow.com', 'reddit.com',
        'discord.com', 'telegram.org', 'whatsapp.com', 'tiktok.com', 'snapchat.com',
        'pinterest.com', 'tumblr.com', 'medium.com', 'wordpress.com', 'blogspot.com'
    }
    
    @classmethod
    def extract_urls(cls, text: str, confidence_threshold: float = 0.5) -> List[ExtractedURL]:
        """
        Extract URLs from text with confidence scoring.
        
        Args:
            text: Text to extract URLs from
            confidence_threshold: Minimum confidence for extraction
            
        Returns:
            List of ExtractedURL objects
        """
        if not text:
            return []
        
        urls = []
        text_length = len(text)
        
        for pattern in cls.URL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                url = match.group(0).strip()
                if not url:
                    continue
                
                # Calculate confidence based on match quality
                confidence = cls._calculate_confidence(url, match, text_length)
                
                if confidence >= confidence_threshold:
                    extracted_url = cls._process_url_match(url, match.group(0), confidence)
                    if extracted_url:
                        urls.append(extracted_url)
        
        # Remove duplicates based on normalized URL
        unique_urls = {}
        for url in urls:
            if url.normalized_url not in unique_urls:
                unique_urls[url.normalized_url] = url
        
        return list(unique_urls.values())
    
    @classmethod
    def _calculate_confidence(cls, url: str, match, text_length: int) -> float:
        """Calculate confidence score for URL extraction."""
        base_confidence = 1.0
        
        # Reduce confidence for very short matches
        if len(url) < 4:
            base_confidence *= 0.5
        
        # Increase confidence for well-formed URLs
        if url.startswith(('http://', 'https://', 'ftp://', 'www.')):
            base_confidence *= 1.2
        
        # Reduce confidence for URLs in suspicious context
        context_before = match.string[max(0, match.start()-50):match.start()].lower()
        suspicious_words = ['click', 'link', 'visit', 'go to', 'check out']
        if any(word in context_before for word in suspicious_words):
            base_confidence *= 1.1
        
        # Reduce confidence for very long URLs (likely false positives)
        if len(url) > 200:
            base_confidence *= 0.3
        
        # Cap confidence at 1.0
        return min(base_confidence, 1.0)
    
    @classmethod
    def _process_url_match(cls, url: str, original_match: str, confidence: float) -> Optional[ExtractedURL]:
        """Process a URL match and return ExtractedURL object."""
        try:
            # Normalize the URL
            normalized_url = cls._normalize_url(url)
            if not normalized_url:
                return None
            
            # Extract domain
            domain = cls._extract_domain(normalized_url)
            
            # Determine if it's a direct URL or inferred domain
            is_direct = original_match.startswith(('http://', 'https://', 'ftp://', 'file://'))
            
            return ExtractedURL(
                url=url,
                normalized_url=normalized_url,
                domain=domain,
                is_direct=is_direct,
                confidence=confidence,
                context=original_match
            )
            
        except Exception as e:
            logging.debug(f"Failed to process URL match '{url}': {e}")
            return None
    
    @classmethod
    def _normalize_url(cls, url: str) -> Optional[str]:
        """Normalize URL to standard format."""
        url = url.strip()
        
        # Handle email addresses
        if '@' in url and '.' in url and not url.startswith(('http://', 'https://')):
            if not url.startswith('mailto:'):
                return f'mailto:{url}'
        
        # Add protocol for www domains
        if url.startswith('www.'):
            url = 'http://' + url
        # Add protocol for domain names without protocol
        elif '.' in url and not url.startswith(('http://', 'https://', 'ftp://', 'file://', 'mailto:')):
            # Handle protocol-relative URLs
            if url.startswith('//'):
                url = 'https:' + url
            # Check if it looks like a domain name
            elif re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', url):
                url = 'https://' + url
        
        # Skip if doesn't look like a URL
        if not re.match(r'^[a-zA-Z0-9.-]+://', url):
            if '.' not in url and not url.startswith('localhost'):
                return None
        
        # Ensure valid URL structure
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.netloc and not parsed.path:
                return None
        except Exception:
            return None
        
        return url
    
    @classmethod
    def _extract_domain(cls, url: str) -> str:
        """Extract domain from URL."""
        try:
            if url.startswith('mailto:'):
                return url.split('@')[1] if '@' in url else ''
            
            parsed = urllib.parse.urlparse(url)
            if parsed.netloc:
                return parsed.netloc
            elif '.' in url:
                # Handle cases without protocol
                parts = url.split('/')[0].split('.')
                if len(parts) >= 2:
                    return '.'.join(parts[-2:])
            return url
        except Exception:
            return url
    
    @classmethod
    def extract_from_clipboard(cls) -> List[ExtractedURL]:
        """Extract URLs from clipboard content with fallback."""
        urls = []
        
        # Try clipboard first
        if CLIPBOARD_AVAILABLE:
            try:
                clipboard_text = pyperclip.paste()
                if clipboard_text:
                    urls.extend(cls.extract_urls(clipboard_text))
            except Exception as e:
                logging.debug(f"Clipboard extraction failed: {e}")
        
        # If no URLs from clipboard or clipboard unavailable, try common fallback methods
        if not urls:
            urls.extend(cls._try_fallback_extraction())
        
        return urls
    
    @classmethod
    def _try_fallback_extraction(cls) -> List[ExtractedURL]:
        """Try fallback methods for URL extraction."""
        urls = []
        
        # Try common environment variables that might contain URLs
        env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                   'HOME', 'USERPROFILE', 'PATH']
        
        for var in env_vars:
            try:
                import os
                value = os.environ.get(var)
                if value and ('http' in value or 'www.' in value):
                    urls.extend(cls.extract_urls(value))
            except Exception:
                continue
        
        return urls
    
    @classmethod
    def infer_domains(cls, text: str) -> List[ExtractedURL]:
        """Infer potential domains from context."""
        domains = []
        
        # Look for domain-like patterns without full URLs
        domain_pattern = r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
        matches = re.finditer(domain_pattern, text, re.IGNORECASE)
        
        for match in matches:
            domain = match.group(0).strip().lower()
            
            # Skip if it's clearly not a domain (too short, no TLD)
            parts = domain.split('.')
            if len(parts) < 2:
                continue
            
            tld = parts[-1]
            if tld not in cls.COMMON_TLDS and not tld.isdigit():
                continue
            
            # Skip common false positives
            false_positives = {'example.com', 'localhost', 'test.com', 'sample.com'}
            if domain in false_positives:
                continue
            
            # Create inferred URL
            inferred_url = f"https://{domain}"
            domains.append(ExtractedURL(
                url=domain,
                normalized_url=inferred_url,
                domain=domain,
                is_direct=False,
                confidence=0.6,  # Lower confidence for inferred domains
                context=f"Inferred domain from text: {match.group(0)}"
            ))
        
        return domains


class OCRExtractor:
    """OCR text extraction using Tesseract and PIL."""
    
    @classmethod
    def extract_text_from_image(cls, image_data: Any, config: Optional[Dict[str, Any]] = None) -> OCRResult:
        """
        Extract text from image using OCR.
        
        Args:
            image_data: Image data (file path, bytes, PIL Image, or base64 string)
            config: OCR configuration options
            
        Returns:
            OCRResult object with extracted text and metadata
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for OCR functionality")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required for OCR functionality")
        
        config = config or {}
        
        try:
            # Load and preprocess image
            image = cls._load_image(image_data)
            if image is None:
                return OCRResult("", 0.0)
            
            # Preprocess image for better OCR
            processed_image = cls._preprocess_image(image, config)
            
            # Configure Tesseract
            tess_config = cls._build_tesseract_config(config)
            
            # Perform OCR
            custom_config = tess_config.get('custom_config', '--psm 3')
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=custom_config)
            confidence = cls._calculate_confidence(data)
            
            # Get bounding boxes if requested
            bbox = None
            words = None
            if config.get('extract_words', False):
                words = cls._extract_words(data)
                bbox = cls._calculate_bbox(data)
            
            # Clean the extracted text
            cleaned_text = TextProcessor.clean_text(text)
            
            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                bbox=bbox,
                words=words
            )
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return OCRResult("", 0.0)
    
    @classmethod
    def _load_image(cls, image_data: Any) -> Optional[Image.Image]:
        """Load image from various input formats."""
        try:
            # Handle file path
            if isinstance(image_data, (str, Path)):
                image_path = Path(image_data)
                if image_path.exists():
                    return Image.open(image_path)
            
            # Handle bytes
            elif isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data))
            
            # Handle base64 string
            elif isinstance(image_data, str):
                # Check if it's base64
                try:
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]  # Remove data URL prefix
                    image_bytes = base64.b64decode(image_data)
                    return Image.open(io.BytesIO(image_bytes))
                except Exception:
                    # Try as file path
                    if Path(image_data).exists():
                        return Image.open(image_data)
            
            # Handle PIL Image
            elif PIL_AVAILABLE and isinstance(image_data, Image.Image):
                return image_data
            
            return None
            
        except Exception as e:
            logging.debug(f"Failed to load image: {e}")
            return None
    
    @classmethod
    def _preprocess_image(cls, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to RGB if necessary
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        # Resize if image is too small
        min_size = config.get('min_image_size', 300)
        if min(image.size) < min_size:
            scale_factor = min_size / min(image.size)
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast if requested
        if config.get('enhance_contrast', True):
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Apply sharpening filter
        if config.get('sharpen', True):
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Convert to grayscale for better OCR
        if config.get('grayscale', True):
            image = image.convert('L')
        
        return image
    
    @classmethod
    def _build_tesseract_config(cls, config: Dict[str, Any]) -> Dict[str, str]:
        """Build Tesseract configuration."""
        tess_config = {}
        
        # Page segmentation mode
        psm = config.get('psm', 3)  # Default: Fully automatic page segmentation
        tess_config['custom_config'] = f'--psm {psm}'
        
        # OCR Engine Mode
        oem = config.get('oem', 3)  # Default: Neural nets LSTM + legacy
        tess_config['custom_config'] += f' --oem {oem}'
        
        # Languages
        languages = config.get('languages', 'eng')
        tess_config['custom_config'] += f' -l {languages}'
        
        return tess_config
    
    @classmethod
    def _calculate_confidence(cls, data: Dict[str, Any]) -> float:
        """Calculate average confidence from OCR data."""
        confidences = []
        for conf in data.get('conf', []):
            if int(conf) > 0:  # Only consider positive confidences
                confidences.append(int(conf))
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    @classmethod
    def _extract_words(cls, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract word-level OCR results."""
        words = []
        for i in range(len(data.get('text', []))):
            conf = data['conf'][i]
            if int(conf) > 0 and data['text'][i].strip():
                words.append({
                    'text': data['text'][i],
                    'confidence': int(conf),
                    'bbox': (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                })
        return words
    
    @classmethod
    def _calculate_bbox(cls, data: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for entire text."""
        x_coords = []
        y_coords = []
        
        for i in range(len(data.get('text', []))):
            if int(data['conf'][i]) > 0 and data['text'][i].strip():
                x_coords.extend([data['left'][i], data['left'][i] + data['width'][i]])
                y_coords.extend([data['top'][i], data['top'][i] + data['height'][i]])
        
        if x_coords and y_coords:
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        return None
    
    @classmethod
    def extract_text_from_screenshot(cls, screenshot_path: str) -> OCRResult:
        """Extract text from screenshot file."""
        return cls.extract_text_from_image(screenshot_path)
    
    @classmethod
    def batch_extract_from_images(cls, image_paths: List[str], config: Optional[Dict[str, Any]] = None) -> List[OCRResult]:
        """Extract text from multiple images."""
        results = []
        for path in image_paths:
            try:
                result = cls.extract_text_from_image(path, config)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to extract text from {path}: {e}")
                results.append(OCRResult("", 0.0))
        
        return results


# Original abstract base class
class ContextExtractor(ABC):
    """Abstract base class for context extractors."""
    
    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from data."""
        pass


# Original TextExtractor (enhanced)
class TextExtractor(ContextExtractor):
    """Extract and analyze text content from context data."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text-related context information.
        
        Args:
            data: Context data to analyze
            
        Returns:
            Extracted text information
        """
        extracted = {
            'type': 'text_extraction',
            'timestamp': datetime.now().isoformat(),
            'text_content': [],
            'keywords': [],
            'entities': [],
            'sentiment': 'neutral'
        }
        
        # Extract text from various sources
        text_sources = self._extract_text_sources(data)
        
        for source, text in text_sources.items():
            if text:
                extracted['text_content'].append({
                    'source': source,
                    'text': text,
                    'length': len(text)
                })
        
        # Extract keywords
        extracted['keywords'] = self._extract_keywords(extracted['text_content'])
        
        # Extract entities (enhanced with new URL extractor)
        extracted['entities'] = self._extract_entities(extracted['text_content'])
        
        return extracted
    
    def _extract_text_sources(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract text from various data sources."""
        sources = {}
        
        # Get text from clipboard
        if 'clipboard' in data and data['clipboard']:
            sources['clipboard'] = data['clipboard']
        
        # Get text from active window title
        if 'active_window' in data and 'title' in data['active_window']:
            sources['window_title'] = data['active_window']['title']
        
        # Get recent file names
        if 'recent_files' in data:
            sources['file_list'] = ', '.join(data['recent_files'])
        
        return sources
    
    def _flatten_text_segments(self, text_content: Union[str, Dict[str, Any], List[Any]]) -> List[str]:
        """Normalize heterogeneous text_content structures into a list of strings."""
        flattened: List[str] = []
        
        if text_content is None:
            return flattened
        
        if isinstance(text_content, str):
            if text_content:
                flattened.append(text_content)
            return flattened
        
        if isinstance(text_content, dict):
            text_value = text_content.get('text')
            if isinstance(text_value, str) and text_value:
                flattened.append(text_value)
            return flattened
        
        if isinstance(text_content, list):
            for item in text_content:
                flattened.extend(self._flatten_text_segments(item))
        
        return flattened
    
    def _extract_keywords(self, text_content: Union[str, List[Dict[str, Any]]]) -> List[str]:
        """Extract keywords from text content."""
        keywords: List[str] = []
        seen: Set[str] = set()
        
        # Common words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        texts = self._flatten_text_segments(text_content)
        
        for text in texts:
            lowered = text.lower()
            # Simple word extraction (allow short acronyms like "ai")
            words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', lowered)
            filtered_words = [w for w in words if w not in stop_words]
            
            for word in filtered_words:
                if word not in seen:
                    seen.add(word)
                    keywords.append(word)
            
            # Add simple bigrams for contextual phrases
            for idx in range(len(filtered_words) - 1):
                first, second = filtered_words[idx], filtered_words[idx + 1]
                bigram = f"{first} {second}"
                if bigram not in seen:
                    seen.add(bigram)
                    keywords.append(bigram)
        
        return keywords[:20]  # Return top 20 keywords
    
    def _extract_entities(self, text_content: Union[str, List[Dict[str, Any]]]) -> List[str]:
        """Extract named entities from text content (enhanced)."""
        entities: List[str] = []
        texts = self._flatten_text_segments(text_content)
        
        for text in texts:
            if not text:
                continue
            
            # Use enhanced URL extractor for better results
            url_extractor = URLExtractor()
            urls = url_extractor.extract_urls(text)
            entities.extend([url.normalized_url for url in urls])
            
            # Extract file paths
            file_paths = re.findall(r'/[^/\s]+\.[a-zA-Z0-9]+', text)
            entities.extend(file_paths)
        
        return entities


class SystemExtractor(ContextExtractor):
    """Extract system-related context information."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract system-related context information.
        
        Args:
            data: Context data to analyze
            
        Returns:
            Extracted system information
        """
        extracted = {
            'type': 'system_extraction',
            'timestamp': datetime.now().isoformat(),
            'system_profile': {},
            'activity_patterns': [],
            'resource_usage': {}
        }
        
        # Extract system information
        if 'system_info' in data:
            extracted['system_profile'] = self._analyze_system_info(data['system_info'])
        
        # Extract activity patterns
        if 'active_window' in data:
            extracted['activity_patterns'].append({
                'type': 'window_activity',
                'application': data['active_window'].get('application', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Analyze file access patterns
        if 'recent_files' in data:
            extracted['activity_patterns'].extend(self._analyze_file_patterns(data['recent_files']))
        
        return extracted
    
    def _analyze_system_info(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system information."""
        profile = {
            'platform': system_info.get('os', 'Unknown'),
            'architecture': system_info.get('architecture', 'Unknown'),
            'hostname': system_info.get('hostname', 'Unknown')
        }
        
        return profile
    
    def _analyze_file_patterns(self, files: List[str]) -> List[Dict[str, Any]]:
        """Analyze file access patterns."""
        patterns = []
        
        for file_path in files:
            patterns.append({
                'type': 'file_access',
                'file_path': file_path,
                'timestamp': datetime.now().isoformat()
            })
        
        return patterns


class NetworkExtractor(ContextExtractor):
    """Extract network-related context information."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract network-related context information.
        
        Args:
            data: Context data to analyze
            
        Returns:
            Extracted network information
        """
        extracted = {
            'type': 'network_extraction',
            'timestamp': datetime.now().isoformat(),
            'connectivity': {},
            'network_context': {}
        }
        
        # Extract network status
        if 'network_status' in data:
            extracted['connectivity'] = data['network_status']
        
        # Extract network-related entities from text
        if 'active_window' in data and 'title' in data['active_window']:
            title = data['active_window']['title']
            network_terms = self._extract_network_terms(title)
            if network_terms:
                extracted['network_context']['title_network_terms'] = network_terms
        
        return extracted
    
    def _extract_network_terms(self, text: str) -> List[str]:
        """Extract network-related terms from text."""
        network_terms = []
        
        # Use enhanced URL extractor for better network term detection
        url_extractor = URLExtractor()
        urls = url_extractor.extract_urls(text)
        network_terms.extend([url.normalized_url for url in urls])
        
        # Common network-related patterns
        network_patterns = [
            r'\b(?:wifi|ethernet|bluetooth|network|connection|internet)\b'  # Network keywords
        ]
        
        for pattern in network_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            network_terms.extend(matches)
        
        return network_terms


class EnhancedContextExtractor(ContextExtractor):
    """
    Main context extraction orchestrator with OCR and URL extraction.
    
    Manages multiple extractors and combines their results with new
    OCR text extraction and comprehensive URL extraction capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced context extractor.
        
        Args:
            config: Extractor configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.extractors = {
            'text': TextExtractor(),
            'system': SystemExtractor(),
            'network': NetworkExtractor()
        }
        
        # Initialize enhanced extractors
        self.ocr_extractor = OCRExtractor()
        self.url_extractor = URLExtractor()
        self.text_processor = TextProcessor()
        
        # Configure enabled extractors
        enabled = config.get('enabled_extractors', ['text', 'system', 'network'])
        self.enabled_extractors = {name: extractor for name, extractor in self.extractors.items() 
                                 if name in enabled}
        
        self.logger.info(f"Enhanced context extractor initialized with features: OCR={PIL_AVAILABLE and TESSERACT_AVAILABLE}, Clipboard={CLIPBOARD_AVAILABLE}")
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context using all extractors including OCR and URL extraction.
        
        Args:
            data: Context data to analyze
            
        Returns:
            Combined extraction results
        """
        self.logger.info("Starting enhanced context extraction")
        
        combined_result = {
            'type': 'enhanced_combined_extraction',
            'timestamp': datetime.now().isoformat(),
            'extractions': {},
            'enhanced_features': {},
            'metadata': {}
        }
        
        # Run original extractors
        for name, extractor in self.enabled_extractors.items():
            try:
                self.logger.debug(f"Running {name} extractor")
                result = extractor.extract(data)
                combined_result['extractions'][name] = result
            except Exception as e:
                self.logger.error(f"Error in {name} extractor: {e}")
                combined_result['extractions'][name] = {
                    'type': f'{name}_extraction',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        try:
            # Add OCR extraction if image data is provided
            if 'image' in data or 'screenshot' in data:
                image_data = data.get('image') or data.get('screenshot')
                ocr_result = self.ocr_extractor.extract_text_from_image(
                    image_data, 
                    self.config.get('ocr', {})
                )
                
                if ocr_result.text:
                    combined_result['enhanced_features']['ocr_extraction'] = {
                        'type': 'ocr_extraction',
                        'text': ocr_result.text,
                        'confidence': ocr_result.confidence,
                        'bbox': ocr_result.bbox,
                        'words': ocr_result.words,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Extract URLs from OCR text
                    urls = self.url_extractor.extract_urls(ocr_result.text)
                    domains = self.url_extractor.infer_domains(ocr_result.text)
                    
                    combined_result['enhanced_features']['ocr_urls'] = {
                        'direct_urls': [url.__dict__ for url in urls],
                        'inferred_domains': [domain.__dict__ for domain in domains],
                        'total_found': len(urls) + len(domains)
                    }
            
            # Enhanced URL extraction from existing text
            if 'text' in data and data['text']:
                text = data['text']
                enhanced_urls = self.url_extractor.extract_urls(text)
                inferred_domains = self.url_extractor.infer_domains(text)
                
                combined_result['enhanced_features']['enhanced_url_extraction'] = {
                    'direct_urls': [url.__dict__ for url in enhanced_urls],
                    'inferred_domains': [domain.__dict__ for domain in inferred_domains],
                    'total_found': len(enhanced_urls) + len(inferred_domains)
                }
            
            # Clipboard URL extraction
            if data.get('extract_clipboard_urls', False):
                clipboard_urls = self.url_extractor.extract_from_clipboard()
                if clipboard_urls:
                    combined_result['enhanced_features']['clipboard_urls'] = [url.__dict__ for url in clipboard_urls]
            
            # Legacy-friendly content block for downstream compatibility
            combined_result['content'] = self._build_legacy_content(combined_result, data)
            
            # Create comprehensive summary
            combined_result['summary'] = self._create_enhanced_summary(combined_result['extractions'], combined_result['enhanced_features'])
            
            # Add comprehensive metadata
            combined_result['metadata'] = {
                'extraction_id': self._generate_extraction_id(),
                'timestamp': datetime.now().isoformat(),
                'features_used': self._get_enabled_features(data),
                'total_urls_found': self._count_total_urls(combined_result),
                'ocr_available': PIL_AVAILABLE and TESSERACT_AVAILABLE,
                'clipboard_available': CLIPBOARD_AVAILABLE,
                'dependencies_status': {
                    'pil_available': PIL_AVAILABLE,
                    'tesseract_available': TESSERACT_AVAILABLE,
                    'clipboard_available': CLIPBOARD_AVAILABLE
                }
            }
            
            self.logger.info(f"Enhanced extraction completed: {combined_result['metadata']['total_urls_found']} URLs found")
            
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {e}")
            combined_result['enhanced_error'] = str(e)
        
        return combined_result
    
    def extract_urls_only(self, text: str) -> List[ExtractedURL]:
        """Extract only URLs from text."""
        return self.url_extractor.extract_urls(text)
    
    def extract_ocr_only(self, image_data: Any, config: Optional[Dict[str, Any]] = None) -> OCRResult:
        """Extract only OCR from image."""
        return self.ocr_extractor.extract_text_from_image(image_data, config)
    
    def _create_enhanced_summary(self, extractions: Dict[str, Any], enhanced_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enhanced summary of all extractions."""
        summary = {
            'extraction_count': len(extractions) + len(enhanced_features),
            'keywords': [],
            'entities': [],
            'applications': [],
            'network_context': {},
            'urls_found': 0,
            'confidence': 0.8,
            'ocr_performed': False,
            'clipboard_extracted': False
        }
        
        # Collect keywords from text extraction
        if 'text' in extractions and 'keywords' in extractions['text']:
            summary['keywords'] = extractions['text']['keywords']
        
        # Collect entities from text extraction
        if 'text' in extractions and 'entities' in extractions['text']:
            summary['entities'] = extractions['text']['entities']
        
        # Collect applications from system extraction
        if 'system' in extractions:
            system_extraction = extractions['system']
            if 'activity_patterns' in system_extraction:
                for pattern in system_extraction['activity_patterns']:
                    if pattern.get('type') == 'window_activity':
                        app = pattern.get('application', 'Unknown')
                        if app not in summary['applications']:
                            summary['applications'].append(app)
        
        # Collect network context
        if 'network' in extractions and 'network_context' in extractions['network']:
            summary['network_context'] = extractions['network']['network_context']
        
        # Count URLs from enhanced features
        if 'enhanced_url_extraction' in enhanced_features:
            summary['urls_found'] += enhanced_features['enhanced_url_extraction']['total_found']
        
        if 'ocr_urls' in enhanced_features:
            summary['urls_found'] += enhanced_features['ocr_urls']['total_found']
            summary['ocr_performed'] = True
        
        if 'clipboard_urls' in enhanced_features:
            summary['urls_found'] += len(enhanced_features['clipboard_urls'])
            summary['clipboard_extracted'] = True
        
        return summary
    
    def _build_legacy_content(self, combined_result: Dict[str, Any], original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create simplified content view expected by legacy integrations/tests."""
        text_extraction = combined_result['extractions'].get('text', {}) if combined_result else {}
        text_entries = text_extraction.get('text_content') if isinstance(text_extraction, dict) else []
        
        primary_text = None
        if isinstance(text_entries, list):
            clipboard_entry = next(
                (
                    entry for entry in text_entries
                    if isinstance(entry, dict) and entry.get('source') == 'clipboard' and entry.get('text')
                ),
                None
            )
            if clipboard_entry:
                primary_text = clipboard_entry.get('text')
            elif text_entries:
                first_entry = text_entries[0]
                if isinstance(first_entry, dict):
                    primary_text = first_entry.get('text')
                elif isinstance(first_entry, str):
                    primary_text = first_entry
        elif isinstance(text_entries, str):
            primary_text = text_entries
        
        if not primary_text:
            primary_text = original_data.get('clipboard') or original_data.get('text') or ''
        
        keywords = text_extraction.get('keywords') if isinstance(text_extraction, dict) else []
        entities = text_extraction.get('entities') if isinstance(text_extraction, dict) else []
        
        return {
            'text': primary_text or '',
            'keywords': keywords or [],
            'entities': entities or []
        }
    
    def _generate_extraction_id(self) -> str:
        """Generate unique extraction ID."""
        timestamp = str(datetime.now().isoformat())
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _get_enabled_features(self, data: Dict[str, Any]) -> List[str]:
        """Get list of enabled features based on input data."""
        features = ['text_extraction', 'system_extraction', 'network_extraction']
        
        if 'image' in data or 'screenshot' in data:
            features.append('ocr_extraction')
        
        if 'text' in data:
            features.append('enhanced_url_extraction')
        
        if data.get('extract_clipboard_urls', False):
            features.append('clipboard_extraction')
        
        return features
    
    def _count_total_urls(self, result: Dict[str, Any]) -> int:
        """Count total URLs found across all extraction methods."""
        count = 0
        
        # Count from enhanced features
        if 'enhanced_features' in result:
            features = result['enhanced_features']
            
            if 'enhanced_url_extraction' in features:
                count += features['enhanced_url_extraction']['total_found']
            
            if 'ocr_urls' in features:
                count += features['ocr_urls']['total_found']
            
            if 'clipboard_urls' in features:
                count += len(features['clipboard_urls'])
        
        # Count from original entity extraction
        if 'extractions' in result and 'text' in result['extractions']:
            if 'entities' in result['extractions']['text']:
                # Count URLs in entities (basic pattern matching)
                entities = result['extractions']['text']['entities']
                count += len([e for e in entities if 'http' in e or '@' in e])
        
        return count
    
    def save_extraction_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save extraction results to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


# Backward compatibility alias for external imports
ContextExtractor = EnhancedContextExtractor


# Convenience functions for common use cases
def extract_text_from_image(image_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Quick function to extract text from image."""
    try:
        result = OCRExtractor.extract_text_from_image(image_path, config)
        return result.text
    except Exception as e:
        logging.error(f"Image text extraction failed: {e}")
        return ""


def extract_urls_from_text(text: str) -> List[str]:
    """Quick function to extract URLs from text."""
    try:
        urls = URLExtractor.extract_urls(text)
        return [url.normalized_url for url in urls]
    except Exception as e:
        logging.error(f"URL extraction failed: {e}")
        return []


def extract_clipboard_urls() -> List[str]:
    """Quick function to extract URLs from clipboard."""
    try:
        urls = URLExtractor.extract_from_clipboard()
        return [url.normalized_url for url in urls]
    except Exception as e:
        logging.error(f"Clipboard URL extraction failed: {e}")
        return []


def process_screenshot_and_extract_urls(screenshot_path: str) -> Dict[str, Any]:
    """Process screenshot: OCR + URL extraction."""
    try:
        # Extract text from screenshot
        ocr_result = OCRExtractor.extract_text_from_image(screenshot_path)
        
        # Extract URLs from OCR result
        urls = URLExtractor.extract_urls(ocr_result.text)
        domains = URLExtractor.infer_domains(ocr_result.text)
        
        return {
            'ocr_text': ocr_result.text,
            'ocr_confidence': ocr_result.confidence,
            'urls': [url.__dict__ for url in urls],
            'domains': [domain.__dict__ for domain in domains],
            'total_findings': len(urls) + len(domains)
        }
        
    except Exception as e:
        logging.error(f"Screenshot processing failed: {e}")
        return {'error': str(e)}
