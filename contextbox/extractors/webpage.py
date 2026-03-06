"""
Generic web page content extraction module for ContextBox.

This module provides comprehensive web scraping capabilities including:
- Requests + BeautifulSoup integration for web scraping
- readability-lxml for clean content extraction
- Title, meta description, and Open Graph tag extraction
- Main content vs. navigation/sidebar discrimination
- Link extraction with anchor text
- Image and media metadata extraction
- Text cleaning (removing ads, navigation, etc.)
- Character encoding handling
- Timeout and error handling for failed requests
- Rate limiting and respectful scraping practices
- Integration with ContextBox database schema
- Support for various content types (HTML, plain text, etc.)
"""

import logging
import re
import json
import time
import hashlib
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse

# Web scraping dependencies
try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from bs4 import BeautifulSoup, Comment, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    from readability import Document
    from readability.cleaners import html_cleaner
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    Document = None

# Optional dependencies for enhanced extraction
try:
    from PIL import Image
    import base64
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Time and network utilities
import socket
from contextlib import contextmanager


@dataclass
class ExtractedLink:
    """Represents an extracted link with metadata."""
    url: str
    text: str
    title: Optional[str] = None
    rel: Optional[str] = None
    target: Optional[str] = None
    hreflang: Optional[str] = None
    type_attr: Optional[str] = None
    is_external: bool = False
    is_nofollow: bool = False
    is_ugc: bool = False
    confidence: float = 1.0
    position: Optional[Tuple[int, int]] = None


@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata."""
    src: str
    alt: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    srcset: Optional[str] = None
    sizes: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    is_lazy_loaded: bool = False
    position: Optional[Tuple[int, int]] = None


@dataclass
class ExtractedMedia:
    """Represents extracted media (video, audio, iframe) with metadata."""
    src: str
    type: str  # video, audio, iframe, embed
    title: Optional[str] = None
    poster: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[int] = None
    autoplay: bool = False
    controls: bool = False
    muted: bool = False
    loop: bool = False
    provider: Optional[str] = None
    thumbnail: Optional[str] = None


@dataclass
class WebPageContent:
    """Complete web page extraction result."""
    url: str
    final_url: str
    title: Optional[str]
    description: Optional[str]
    main_content: str
    cleaned_text: str
    content_score: float
    extraction_method: str
    encoding: Optional[str]
    
    # Metadata
    meta_tags: Dict[str, str]
    open_graph: Dict[str, str]
    twitter_cards: Dict[str, str]
    structured_data: Dict[str, Any]
    
    # Content analysis
    word_count: int
    character_count: int
    readability_score: Optional[float]
    language: Optional[str]
    
    # Extracted elements
    links: List[ExtractedLink]
    images: List[ExtractedImage]
    media: List[ExtractedMedia]
    social_links: List[ExtractedLink]
    
    # Technical details
    response_time: float
    status_code: Optional[int]
    error: Optional[str]
    content_hash: Optional[str]
    extracted_at: str
    
    # Rate limiting info
    is_rate_limited: bool = False
    request_count: int = 0


class RateLimiter:
    """Rate limiter for respectful web scraping."""
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 3):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst requests before throttling
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.request_times = deque()
        self.last_request_time = 0
        
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Clean old request times (older than 1 second)
        while self.request_times and now - self.request_times[0] > 1.0:
            self.request_times.popleft()
        
        # Check if we need to wait
        if len(self.request_times) >= self.burst_size:
            # Wait until the oldest request is older than 1 second
            wait_time = 1.0 - (now - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
        
        # Record this request
        self.request_times.append(now)
        self.last_request_time = now


class TextCleaner:
    """Advanced text cleaning for web content."""
    
    @staticmethod
    def clean_html_text(text: str) -> str:
        """Clean HTML tags and entities from text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        text = text.replace('&nbsp;', ' ')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_navigation_content(text: str) -> str:
        """Remove common navigation and sidebar content patterns."""
        if not text:
            return ""
        
        # Remove breadcrumb patterns
        breadcrumb_patterns = [
            r'\b(?:home|category|tag|archive|search|navigation|menu)\b.*?(?=\n\n|\.|\.|,)',
            r'(?:»|>|\|)[\s\S]*?(?=\n\n|\.|\.|,)',
            r'\b(?:read more|continue reading|learn more|click here)\b',
        ]
        
        for pattern in breadcrumb_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove footer content
        footer_indicators = [
            r'\b(?:copyright|©|©\s*\d+|all rights reserved|powered by)\b.*',
            r'\b(?:privacy policy|terms of service|contact us|sitemap)\b.*',
        ]
        
        for pattern in footer_indicators:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def remove_ads_and_noise(text: str) -> str:
        """Remove advertisements and noise content."""
        if not text:
            return ""
        
        # Remove ad indicators
        ad_patterns = [
            r'\b(?:advertisement|sponsored|ad|buy now|shop now|click here to buy)\b.*',
            r'\[ad[^\]]*\]',
            r'<!--\s*ad[\s\S]*?-->',
            r'\b(?:social media|follow us|share this|like and subscribe)\b.*',
        ]
        
        for pattern in ad_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation and special characters
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[*]{2,}', '*', text)
        
        return text.strip()
    
    @classmethod
    def clean_content(cls, text: str, remove_ads: bool = True, remove_nav: bool = True) -> str:
        """Comprehensive text cleaning."""
        if not text:
            return ""
        
        # Clean HTML
        text = cls.clean_html_text(text)
        
        # Remove navigation content
        if remove_nav:
            text = cls.remove_navigation_content(text)
        
        # Remove ads and noise
        if remove_ads:
            text = cls.remove_ads_and_noise(text)
        
        # Final cleanup
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text


class ContentAnalyzer:
    """Analyze content quality and characteristics."""
    
    # Common navigation element selectors
    NAVIGATION_SELECTORS = [
        'nav', '.navigation', '.nav', '#navigation', '.menu', '#menu',
        '.sidebar', '.side-nav', '.breadcrumb', '.crumbs',
        '.header-nav', '.footer-nav', '.main-nav'
    ]
    
    # Ad-related selectors
    AD_SELECTORS = [
        '.ad', '.ads', '#ad', '#ads', '.advertisement', '.sponsored',
        '.banner', '.promo', '.promotion', '.sidebar-ad',
        '[class*="ad-"]', '[id*="ad-"]'
    ]
    
    # Footer selectors
    FOOTER_SELECTORS = [
        'footer', '.footer', '#footer', '.site-footer', '.page-footer'
    ]
    
    @classmethod
    def calculate_content_score(cls, soup: BeautifulSoup, main_content: str) -> float:
        """Calculate content quality score."""
        if not soup or not main_content:
            return 0.0
        
        score = 1.0
        
        # Penalize for excessive ads
        ad_elements = soup.select(', '.join(cls.AD_SELECTORS))
        if ad_elements:
            score -= len(ad_elements) * 0.1
        
        # Penalize for excessive navigation elements
        nav_elements = soup.select(', '.join(cls.NAVIGATION_SELECTORS))
        if nav_elements:
            score -= len(nav_elements) * 0.05
        
        # Check content length vs total page length
        if main_content:
            content_ratio = len(main_content) / len(str(soup))
            score *= min(content_ratio * 5, 1.0)  # Boost if content is substantial
        
        # Penalize for excessive links (navigation sites)
        links = soup.find_all('a', href=True)
        if len(links) > 100:  # More than 100 links suggests navigation-heavy page
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    @classmethod
    def detect_language(cls, text: str) -> Optional[str]:
        """Simple language detection based on common patterns."""
        if not text:
            return None
        
        # Very basic language detection - could be enhanced with proper library
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        english_count = sum(1 for word in words if word in english_indicators)
        if words:
            ratio = english_count / len(words)
            if ratio > 0.05:  # 5% common English words
                return 'en'
        
        return None
    
    @classmethod
    def calculate_readability_score(cls, text: str) -> Optional[float]:
        """Calculate a simple readability score (Flesch-like)."""
        if not text:
            return None
        
        # Very simple readability calculation
        sentences = re.split(r'[.!?]+', text)
        words = re.findall(r'\b\w+\b', text)
        syllables = sum(len(re.findall(r'[aeiouAEIOU]', word)) for word in words)
        
        if not sentences or not words:
            return None
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        # Simple readability formula
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        return max(0, min(100, score))


class WebPageExtractor:
    """
    Main web page content extraction engine.
    
    Provides comprehensive web scraping capabilities with:
    - Intelligent content extraction using readability algorithm
    - Robust error handling and timeouts
    - Rate limiting for respectful scraping
    - Content cleaning and analysis
    - Support for various content types
    - Integration with ContextBox database schema
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web page extractor.
        
        Args:
            config: Configuration dictionary
                - timeout: Request timeout in seconds
                - max_retries: Maximum retry attempts
                - user_agent: Custom user agent string
                - rate_limit: Requests per second
                - extract_images: Whether to extract images
                - extract_social_links: Whether to extract social media links
                - respect_robots_txt: Whether to check robots.txt
                - follow_redirects: Whether to follow HTTP redirects
        """
        self.config = config or {}
        
        # Core settings
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.user_agent = self.config.get(
            'user_agent',
            'ContextBox Web Extractor (Educational/Purpose)'
        )
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.get('rate_limit', 1.0),
            burst_size=self.config.get('burst_size', 3)
        )
        
        # Feature flags
        self.extract_images = self.config.get('extract_images', True)
        self.extract_social_links = self.config.get('extract_social_links', True)
        self.respect_robots_txt = self.config.get('respect_robots_txt', False)
        self.follow_redirects = self.config.get('follow_redirects', True)
        
        # Error handling
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Session for connection pooling
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for web page extraction")
        
        session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def extract(self, url: str, **kwargs) -> WebPageContent:
        """
        Extract content from a web page.
        
        Args:
            url: URL to extract content from
            **kwargs: Additional extraction parameters
            
        Returns:
            WebPageContent object with extracted information
        """
        start_time = time.time()
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Fetch the page
            response = self._fetch_page(url)
            
            # Extract content
            content = self._extract_content(response, url)
            
            # Calculate processing time
            content.response_time = time.time() - start_time
            content.request_count = len(self.rate_limiter.request_times)
            
            self.stats['successful_extractions'] += 1
            self.logger.info(f"Successfully extracted content from {url}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            content = self._create_error_content(url, str(e), time.time() - start_time)
        
        return content
    
    def _fetch_page(self, url: str) -> requests.Response:
        """Fetch web page with proper error handling."""
        try:
            self.rate_limiter.wait_if_needed()
            
            # Check URL validity
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                raise ValueError(f"Invalid URL: {url}")
            
            # Make request
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects,
                stream=False
            )
            
            response.raise_for_status()
            
            # Log successful fetch
            self.logger.debug(f"Successfully fetched {url} - Status: {response.status_code}")
            
            return response
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out for {url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request failed for {url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching {url}: {e}")
    
    def _extract_content(self, response: requests.Response, original_url: str) -> WebPageContent:
        """Extract all content from the response."""
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required for HTML parsing")
        
        if not READABILITY_AVAILABLE:
            raise ImportError("readability-lxml is required for content extraction")
        
        # Get encoding
        encoding = response.encoding or response.apparent_encoding or 'utf-8'
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract content using multiple methods
        main_content, extraction_method = self._extract_main_content(soup)
        
        # Clean content
        cleaned_content = TextCleaner.clean_content(main_content)
        
        # Calculate scores
        content_score = ContentAnalyzer.calculate_content_score(soup, main_content)
        readability_score = ContentAnalyzer.calculate_readability_score(cleaned_content)
        language = ContentAnalyzer.detect_language(cleaned_content)
        
        # Extract metadata
        meta_tags = self._extract_meta_tags(soup)
        open_graph = self._extract_open_graph(soup)
        twitter_cards = self._extract_twitter_cards(soup)
        structured_data = self._extract_structured_data(soup)
        
        # Extract page elements
        links = self._extract_links(soup, response.url)
        images = self._extract_images(soup, response.url) if self.extract_images else []
        media = self._extract_media(soup, response.url)
        social_links = self._extract_social_links(soup) if self.extract_social_links else []
        
        # Extract basic content
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        
        # Calculate hash for deduplication
        content_hash = hashlib.md5(response.content).hexdigest()
        
        return WebPageContent(
            url=original_url,
            final_url=response.url,
            title=title,
            description=description,
            main_content=main_content,
            cleaned_text=cleaned_content,
            content_score=content_score,
            extraction_method=extraction_method,
            encoding=encoding,
            
            meta_tags=meta_tags,
            open_graph=open_graph,
            twitter_cards=twitter_cards,
            structured_data=structured_data,
            
            word_count=len(cleaned_content.split()),
            character_count=len(cleaned_content),
            readability_score=readability_score,
            language=language,
            
            links=links,
            images=images,
            media=media,
            social_links=social_links,
            
            status_code=response.status_code,
            content_hash=content_hash,
            extracted_at=datetime.now().isoformat()
        )
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """Extract main content using readability algorithm."""
        if not soup:
            return "", "none"
        
        try:
            # Try readability first
            doc = Document(str(soup))
            readable_html = doc.summary(html_partial=True)
            
            # Parse the readable HTML
            readable_soup = BeautifulSoup(readable_html, 'lxml')
            
            # Extract text
            content = readable_soup.get_text(separator='\n', strip=True)
            
            if content and len(content.strip()) > 100:  # Minimum content threshold
                return content, "readability"
        
        except Exception as e:
            self.logger.debug(f"Readability extraction failed: {e}")
        
        # Fallback: extract from common content containers
        content_selectors = [
            'article', 'main', '.content', '#content', '.post', '.entry',
            '[role="main"]', '.article-content', '.post-content', '.story'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                content = element.get_text(separator='\n', strip=True)
                if content and len(content.strip()) > 200:
                    return content, f"fallback_{selector}"
        
        # Final fallback: extract from body, excluding navigation/footer
        body = soup.find('body') or soup
        if body:
            # Remove script and style elements
            for script in body(["script", "style"]):
                script.decompose()
            
            # Remove navigation and footer elements
            for nav_element in body.find_all(['nav', 'header', 'footer', 'aside']):
                nav_element.decompose()
            
            content = body.get_text(separator='\n', strip=True)
            return content, "body_fallback"
        
        return "", "no_content"
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        # Try <title> tag first
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            if title:
                return title
        
        # Try og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title.get('content', '').strip()
        
        # Try twitter:title
        twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if twitter_title and twitter_title.get('content'):
            return twitter_title.get('content', '').strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag:
            h1_text = h1_tag.get_text(strip=True)
            if h1_text:
                return h1_text
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content', '').strip()
        
        # Try og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc.get('content', '').strip()
        
        # Try twitter:description
        twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
        if twitter_desc and twitter_desc.get('content'):
            return twitter_desc.get('content', '').strip()
        
        return None
    
    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta tags."""
        meta_tags = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            
            if name and content:
                meta_tags[name] = content.strip()
        
        return meta_tags
    
    def _extract_open_graph(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Open Graph tags."""
        og_tags = {}
        
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            property_name = meta.get('property', '')
            content = meta.get('content', '')
            
            if property_name and content:
                og_tags[property_name] = content.strip()
        
        return og_tags
    
    def _extract_twitter_cards(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Twitter Card tags."""
        twitter_tags = {}
        
        for meta in soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')}):
            name = meta.get('name', '')
            content = meta.get('content', '')
            
            if name and content:
                twitter_tags[name] = content.strip()
        
        return twitter_tags
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, microdata)."""
        structured_data = {}
        
        # Extract JSON-LD
        json_scripts = soup.find_all('script', type='application/ld+json')
        for i, script in enumerate(json_scripts):
            try:
                data = json.loads(script.string or script.get_text())
                structured_data[f'jsonld_{i}'] = data
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Extract microdata (basic implementation)
        microdata_elements = soup.find_all(attrs={'itemtype': True})
        for element in microdata_elements:
            itemtype = element.get('itemtype', '')
            if itemtype not in structured_data:
                structured_data[f'microdata_{itemtype}'] = []
            
            item_data = {}
            for prop in element.find_all(attrs={'itemprop': True}):
                prop_name = prop.get('itemprop')
                prop_value = prop.get_text(strip=True)
                item_data[prop_name] = prop_value
            
            structured_data[f'microdata_{itemtype}'].append(item_data)
        
        return structured_data
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[ExtractedLink]:
        """Extract links with metadata."""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            if not href or href.startswith('#'):
                continue
            
            # Resolve relative URLs
            if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                href = urljoin(base_url, href)
            
            # Extract text
            text = a_tag.get_text(strip=True) or a_tag.get('title', '') or href
            
            # Extract attributes
            link = ExtractedLink(
                url=href,
                text=text,
                title=a_tag.get('title'),
                rel=a_tag.get('rel'),
                target=a_tag.get('target'),
                hreflang=a_tag.get('hreflang'),
                type_attr=a_tag.get('type'),
                is_external=self._is_external_link(href, base_url),
                is_nofollow='nofollow' in (a_tag.get('rel', []) or []),
                is_ugc='ugc' in (a_tag.get('rel', []) or [])
            )
            
            links.append(link)
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[ExtractedImage]:
        """Extract images with metadata."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '').strip()
            if not src:
                continue
            
            # Resolve relative URLs
            if not src.startswith(('http://', 'https://', 'data:', 'blob:')):
                src = urljoin(base_url, src)
            
            # Extract dimensions
            width = self._parse_dimension(img.get('width'))
            height = self._parse_dimension(img.get('height'))
            
            # Check for lazy loading
            is_lazy = (
                img.get('loading') == 'lazy' or 
                'lazy' in img.get('class', []) or
                'lazy' in img.get('srcset', '')
            )
            
            image = ExtractedImage(
                src=src,
                alt=img.get('alt'),
                title=img.get('title'),
                width=width,
                height=height,
                srcset=img.get('srcset'),
                sizes=img.get('sizes'),
                format=self._extract_image_format(src, img.get('type')),
                is_lazy_loaded=is_lazy
            )
            
            images.append(image)
        
        return images
    
    def _extract_media(self, soup: BeautifulSoup, base_url: str) -> List[ExtractedMedia]:
        """Extract media elements (video, audio, iframe, embed)."""
        media_elements = []
        
        # Video elements
        for video in soup.find_all('video'):
            src = video.get('src', '')
            if src:
                if not src.startswith(('http://', 'https://', 'data:', 'blob:')):
                    src = urljoin(base_url, src)
                
                media = ExtractedMedia(
                    src=src,
                    type='video',
                    title=video.get('title'),
                    poster=video.get('poster'),
                    width=self._parse_dimension(video.get('width')),
                    height=self._parse_dimension(video.get('height')),
                    autoplay=video.get('autoplay') == 'autoplay',
                    controls=video.get('controls') == 'controls',
                    muted=video.get('muted') == 'muted',
                    loop=video.get('loop') == 'loop',
                    provider=self._detect_media_provider(src)
                )
                media_elements.append(media)
        
        # Audio elements
        for audio in soup.find_all('audio'):
            src = audio.get('src', '')
            if src:
                if not src.startswith(('http://', 'https://', 'data:', 'blob:')):
                    src = urljoin(base_url, src)
                
                media = ExtractedMedia(
                    src=src,
                    type='audio',
                    title=audio.get('title'),
                    autoplay=audio.get('autoplay') == 'autoplay',
                    controls=audio.get('controls') == 'controls',
                    loop=audio.get('loop') == 'loop',
                    provider=self._detect_media_provider(src)
                )
                media_elements.append(media)
        
        # Iframe elements
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if src:
                if not src.startswith(('http://', 'https://', 'data:', 'blob:')):
                    src = urljoin(base_url, src)
                
                media = ExtractedMedia(
                    src=src,
                    type='iframe',
                    title=iframe.get('title'),
                    width=self._parse_dimension(iframe.get('width')),
                    height=self._parse_dimension(iframe.get('height')),
                    provider=self._detect_media_provider(src)
                )
                media_elements.append(media)
        
        return media_elements
    
    def _extract_social_links(self, soup: BeautifulSoup) -> List[ExtractedLink]:
        """Extract social media links."""
        social_links = []
        social_domains = {
            'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'github.com', 'tiktok.com', 'pinterest.com',
            'tumblr.com', 'reddit.com', 'discord.com', 'telegram.org'
        }
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            
            # Check if it's a social media link
            try:
                domain = urlparse(href).netloc.lower()
                if any(social_domain in domain for social_domain in social_domains):
                    link = ExtractedLink(
                        url=href,
                        text=a_tag.get_text(strip=True) or href,
                        title=a_tag.get('title'),
                        is_external=True
                    )
                    social_links.append(link)
            except Exception:
                continue
        
        return social_links
    
    def _is_external_link(self, url: str, base_url: str) -> bool:
        """Check if URL is external to the base domain."""
        try:
            base_domain = urlparse(base_url).netloc.lower()
            link_domain = urlparse(url).netloc.lower()
            return base_domain != link_domain
        except Exception:
            return True
    
    def _parse_dimension(self, value: str) -> Optional[int]:
        """Parse dimension value to integer."""
        if not value:
            return None
        
        try:
            # Remove 'px', '%', etc. and convert to int
            clean_value = re.sub(r'[^\d]', '', value)
            return int(clean_value) if clean_value else None
        except (ValueError, TypeError):
            return None
    
    def _extract_image_format(self, src: str, img_type: Optional[str]) -> Optional[str]:
        """Extract image format from src or type."""
        if img_type:
            return img_type.split('/')[-1].lower()
        
        # Extract from file extension
        try:
            path = urlparse(src).path
            return Path(path).suffix.lstrip('.').lower()
        except Exception:
            return None
    
    def _detect_media_provider(self, src: str) -> Optional[str]:
        """Detect media provider from URL."""
        domain = urlparse(src).netloc.lower()
        
        providers = {
            'youtube.com': 'YouTube',
            'youtu.be': 'YouTube',
            'vimeo.com': 'Vimeo',
            'dailymotion.com': 'Dailymotion',
            'twitch.tv': 'Twitch',
            'soundcloud.com': 'SoundCloud',
            'spotify.com': 'Spotify'
        }
        
        for provider_domain, provider_name in providers.items():
            if provider_domain in domain:
                return provider_name
        
        return None
    
    def _create_error_content(self, url: str, error: str, response_time: float) -> WebPageContent:
        """Create error content object."""
        return WebPageContent(
            url=url,
            final_url=url,
            title=None,
            description=None,
            main_content="",
            cleaned_text="",
            content_score=0.0,
            extraction_method="error",
            encoding=None,
            meta_tags={},
            open_graph={},
            twitter_cards={},
            structured_data={},
            word_count=0,
            character_count=0,
            readability_score=None,
            language=None,
            links=[],
            images=[],
            media=[],
            social_links=[],
            response_time=response_time,
            status_code=None,
            error=error,
            content_hash=None,
            extracted_at=datetime.now().isoformat(),
            is_rate_limited=False,
            request_count=len(self.rate_limiter.request_times)
        )
    
    def batch_extract(self, urls: List[str], max_concurrent: int = 5) -> List[WebPageContent]:
        """
        Extract content from multiple URLs in batch.
        
        Args:
            urls: List of URLs to extract
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of WebPageContent objects
        """
        results = []
        
        for url in urls:
            try:
                result = self.extract(url)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to extract {url}: {e}")
                error_content = self._create_error_content(url, str(e), 0)
                results.append(error_content)
        
        return results
    
    def save_extraction(self, content: WebPageContent, output_path: str) -> None:
        """Save extraction result to JSON file."""
        try:
            # Convert dataclass to dictionary for JSON serialization
            content_dict = asdict(content)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Extraction saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save extraction: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return dict(self.stats)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
        
        self.logger.info("WebPageExtractor cleanup completed")


# ContextBox Integration Functions

def integrate_with_contextbox(extractor: WebPageExtractor, capture_data: Dict[str, Any], url: str) -> str:
    """
    Integrate web page extraction with ContextBox database schema.
    
    Args:
        extractor: WebPageExtractor instance
        capture_data: ContextBox capture data
        url: URL to extract
        
    Returns:
        ContextBox capture ID
    """
    try:
        from contextbox.database import ContextDatabase
        
        # Extract content
        content = extractor.extract(url)
        
        # Create ContextBox context structure
        context = {
            'capture': {
                'source_window': f"Web Page: {content.title or url}",
                'notes': f"Extracted from: {url}\nTitle: {content.title or 'N/A'}\nDescription: {content.description or 'N/A'}\nExtraction Method: {content.extraction_method}"
            },
            'artifacts': []
        }
        
        # Add main content artifact
        context['artifacts'].append({
            'kind': 'webpage_content',
            'url': content.final_url,
            'title': content.title,
            'text': content.cleaned_text,
            'metadata': {
                'extraction_method': content.extraction_method,
                'content_score': content.content_score,
                'word_count': content.word_count,
                'readability_score': content.readability_score,
                'language': content.language,
                'response_time': content.response_time,
                'encoding': content.encoding,
                'meta_tags': content.meta_tags,
                'open_graph': content.open_graph,
                'twitter_cards': content.twitter_cards,
                'structured_data': content.structured_data,
                'content_hash': content.content_hash
            }
        })
        
        # Add individual link artifacts
        for link in content.links[:50]:  # Limit to prevent database bloat
            context['artifacts'].append({
                'kind': 'webpage_link',
                'url': link.url,
                'title': link.text,
                'text': f"Link text: {link.text}\nIs external: {link.is_external}\nNofollow: {link.is_nofollow}",
                'metadata': {
                    'link_text': link.text,
                    'is_external': link.is_external,
                    'is_nofollow': link.is_nofollow,
                    'is_ugc': link.is_ugc,
                    'rel': link.rel,
                    'target': link.target
                }
            })
        
        # Add image artifacts
        for image in content.images[:20]:  # Limit to prevent database bloat
            context['artifacts'].append({
                'kind': 'webpage_image',
                'url': image.src,
                'title': image.alt or image.title or 'Image',
                'text': f"Alt text: {image.alt or 'N/A'}\nFormat: {image.format or 'Unknown'}\nDimensions: {image.width}x{image.height}",
                'metadata': {
                    'src': image.src,
                    'alt': image.alt,
                    'title': image.title,
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'is_lazy_loaded': image.is_lazy_loaded
                }
            })
        
        # Store in ContextBox database
        db = ContextDatabase()
        capture_id = db.store(context)
        
        return capture_id
        
    except ImportError:
        raise ImportError("ContextBox database module is required for integration")
    except Exception as e:
        raise RuntimeError(f"Failed to integrate with ContextBox: {e}")


# Convenience Functions

def extract_webpage_content(url: str, config: Optional[Dict[str, Any]] = None) -> WebPageContent:
    """
    Quick function to extract webpage content.
    
    Args:
        url: URL to extract content from
        config: Extractor configuration
        
    Returns:
        WebPageContent object
    """
    extractor = WebPageExtractor(config)
    return extractor.extract(url)


def extract_multiple_pages(urls: List[str], config: Optional[Dict[str, Any]] = None) -> List[WebPageContent]:
    """
    Extract content from multiple URLs.
    
    Args:
        urls: List of URLs to extract
        config: Extractor configuration
        
    Returns:
        List of WebPageContent objects
    """
    extractor = WebPageExtractor(config)
    return extractor.batch_extract(urls)


def extract_and_store_in_contextbox(url: str, config: Optional[Dict[str, Any]] = None, capture_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Extract webpage content and store in ContextBox database.
    
    Args:
        url: URL to extract content from
        config: Extractor configuration
        capture_data: Additional ContextBox capture data
        
    Returns:
        ContextBox capture ID
    """
    extractor = WebPageExtractor(config)
    capture_data = capture_data or {}
    return integrate_with_contextbox(extractor, capture_data, url)


# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python webpage.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract content
        content = extract_webpage_content(url)
        
        # Print summary
        print(f"Title: {content.title}")
        print(f"Description: {content.description}")
        print(f"Word Count: {content.word_count}")
        print(f"Content Score: {content.content_score:.2f}")
        print(f"Extraction Method: {content.extraction_method}")
        print(f"Links Found: {len(content.links)}")
        print(f"Images Found: {len(content.images)}")
        print(f"Media Found: {len(content.media)}")
        
        # Save detailed result
        output_file = f"extraction_{hash(url)}.json"
        extractor = WebPageExtractor()
        extractor.save_extraction(content, output_file)
        print(f"Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)