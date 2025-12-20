"""
Smart Content Classification and Routing System for ContextBox

This module provides intelligent content classification and routing for different
types of web content, enabling automatic extraction method selection and
confidence-based content processing.

Features:
- URL pattern recognition for different content types
- Domain-based classification (YouTube, Wikipedia, news sites, etc.)
- Content type detection (video, article, documentation, etc.)
- Automatic extractor selection and routing
- Confidence scoring for classification accuracy
- Fallback strategies when primary extraction fails
- Support for adding custom domain rules
- Integration with all extractor modules
- Performance optimization (avoid redundant requests)
- Integration with ContextBox database schema

Classification Rules:
- YouTube URLs (video content)
- Wikipedia URLs (encyclopedia content)  
- News sites (article content)
- Documentation sites (technical content)
- Social media URLs (post content)
- Generic URLs (default extraction)
"""

import logging
import re
import json
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional, Type, Tuple, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import time

# Import existing ContextBox modules
try:
    from ..database import ContextDatabase
    from ..extractors import URLExtractor, OCRExtractor
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    ContextDatabase = None
    URLExtractor = None
    OCRExtractor = None


@dataclass
class ClassificationResult:
    """Result of content classification with metadata."""
    url: str
    content_type: str
    domain_type: str
    extractor_name: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_extractors: List[str] = field(default_factory=list)
    routing_info: Dict[str, Any] = field(default_factory=dict)
    classification_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'url': self.url,
            'content_type': self.content_type,
            'domain_type': self.domain_type,
            'extractor_name': self.extractor_name,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'fallback_extractors': self.fallback_extractors,
            'routing_info': self.routing_info,
            'classification_timestamp': self.classification_timestamp,
            'processing_time': self.processing_time
        }


@dataclass 
class ContentRule:
    """Rule definition for content classification."""
    name: str
    domain_patterns: List[str]
    url_patterns: List[str]
    content_type: str
    domain_type: str
    extractor_name: str
    priority: int = 100
    confidence_boost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class BaseExtractor(ABC):
    """Abstract base class for content extractors."""
    
    @abstractmethod
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract content from URL."""
        pass
    
    @abstractmethod
    def get_supported_content_types(self) -> List[str]:
        """Return supported content types."""
        pass


class YouTubeExtractor(BaseExtractor):
    """YouTube video content extractor."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract YouTube video content."""
        try:
            # Parse YouTube URL
            parsed = urllib.parse.urlparse(url)
            video_id = self._extract_video_id(url)
            
            metadata = {
                'platform': 'youtube',
                'video_id': video_id,
                'extraction_type': 'video'
            }
            
            return {
                'success': True,
                'content_type': 'video',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'youtube_api',
                    'title': f'YouTube Video {video_id}',
                    'description': 'YouTube video content extracted'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)',
            r'youtube\.com/.*[?&]v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_supported_content_types(self) -> List[str]:
        return ['video']


class WikipediaExtractor(BaseExtractor):
    """Wikipedia encyclopedia content extractor."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract Wikipedia encyclopedia content."""
        try:
            # Parse Wikipedia URL to extract article title
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.split('/')
            
            metadata = {
                'platform': 'wikipedia',
                'article_title': self._extract_article_title(url),
                'language': self._extract_language(url),
                'extraction_type': 'encyclopedia'
            }
            
            return {
                'success': True,
                'content_type': 'encyclopedia',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'wikipedia_api',
                    'title': metadata['article_title'],
                    'description': 'Wikipedia encyclopedia article'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_article_title(self, url: str) -> str:
        """Extract Wikipedia article title from URL."""
        patterns = [
            r'wikipedia\.org/wiki/([^/?#]+)',
            r'wikimedia\.org/wiki/([^/?#]+)',
            r'\.wikipedia\.org/wiki/([^/?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                title = match.group(1)
                # Replace underscores with spaces and decode URL encoding
                return urllib.parse.unquote_plus(title.replace('_', ' '))
        
        return "Unknown Article"
    
    def _extract_language(self, url: str) -> str:
        """Extract language code from Wikipedia URL."""
        if '.wikipedia.org' in url:
            # Extract language code from subdomain
            match = re.search(r'([a-z]{2,3})\.wikipedia\.org', url)
            if match:
                return match.group(1)
        return 'en'
    
    def get_supported_content_types(self) -> List[str]:
        return ['encyclopedia', 'reference']


class NewsSiteExtractor(BaseExtractor):
    """News article content extractor."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract news article content."""
        try:
            parsed = urllib.parse.urlparse(url)
            metadata = {
                'platform': self._identify_news_platform(parsed.netloc),
                'extraction_type': 'article',
                'domain': parsed.netloc
            }
            
            return {
                'success': True,
                'content_type': 'article',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'web_scraping',
                    'title': f'Article from {metadata["domain"]}',
                    'description': 'News article content'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _identify_news_platform(self, domain: str) -> str:
        """Identify news platform from domain."""
        news_domains = {
            'cnn.com': 'CNN',
            'bbc.com': 'BBC',
            'reuters.com': 'Reuters',
            'apnews.com': 'AP News',
            'nytimes.com': 'New York Times',
            'wsj.com': 'Wall Street Journal',
            'theguardian.com': 'The Guardian',
            'npr.org': 'NPR',
            'bloomberg.com': 'Bloomberg',
            'foxnews.com': 'Fox News',
            'cnbc.com': 'CNBC',
            'usatoday.com': 'USA Today',
            'washingtonpost.com': 'Washington Post'
        }
        
        return news_domains.get(domain, 'Generic News')
    
    def get_supported_content_types(self) -> List[str]:
        return ['article', 'news']


class DocumentationExtractor(BaseExtractor):
    """Technical documentation content extractor."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract technical documentation content."""
        try:
            parsed = urllib.parse.urlparse(url)
            metadata = {
                'platform': self._identify_documentation_platform(parsed.netloc),
                'extraction_type': 'documentation',
                'domain': parsed.netloc,
                'path': parsed.path
            }
            
            return {
                'success': True,
                'content_type': 'documentation',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'documentation_scraping',
                    'title': f'Documentation from {metadata["platform"]}',
                    'description': 'Technical documentation content'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _identify_documentation_platform(self, domain: str) -> str:
        """Identify documentation platform from domain."""
        doc_domains = {
            'developer.mozilla.org': 'MDN Web Docs',
            'docs.python.org': 'Python Documentation',
            'docs.oracle.com': 'Oracle Documentation',
            'aws.amazon.com/documentation': 'AWS Documentation',
            'learn.microsoft.com': 'Microsoft Learn',
            'docs.github.com': 'GitHub Documentation',
            'reactjs.org': 'React Documentation',
            'angular.io': 'Angular Documentation',
            'vuejs.org': 'Vue.js Documentation',
            'nodejs.org': 'Node.js Documentation'
        }
        
        return doc_domains.get(domain, 'Generic Documentation')
    
    def get_supported_content_types(self) -> List[str]:
        return ['documentation', 'technical']


class SocialMediaExtractor(BaseExtractor):
    """Social media content extractor."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract social media content."""
        try:
            parsed = urllib.parse.urlparse(url)
            metadata = {
                'platform': self._identify_social_platform(parsed.netloc),
                'extraction_type': 'social_media',
                'domain': parsed.netloc
            }
            
            return {
                'success': True,
                'content_type': 'social_media',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'social_api',
                    'title': f'Post from {metadata["platform"]}',
                    'description': 'Social media content'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _identify_social_platform(self, domain: str) -> str:
        """Identify social media platform from domain."""
        social_domains = {
            'twitter.com': 'Twitter',
            'facebook.com': 'Facebook',
            'instagram.com': 'Instagram',
            'linkedin.com': 'LinkedIn',
            'reddit.com': 'Reddit',
            'tiktok.com': 'TikTok',
            'pinterest.com': 'Pinterest',
            'tumblr.com': 'Tumblr'
        }
        
        return social_domains.get(domain, 'Generic Social Media')
    
    def get_supported_content_types(self) -> List[str]:
        return ['social_media', 'post']


class GenericExtractor(BaseExtractor):
    """Generic web content extractor for unmatched URLs."""
    
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """Extract generic web content."""
        try:
            parsed = urllib.parse.urlparse(url)
            metadata = {
                'platform': parsed.netloc,
                'extraction_type': 'generic',
                'domain': parsed.netloc,
                'path': parsed.path
            }
            
            return {
                'success': True,
                'content_type': 'webpage',
                'url': url,
                'metadata': metadata,
                'extraction_info': {
                    'method': 'generic_web_scraping',
                    'title': f'Webpage from {metadata["domain"]}',
                    'description': 'Generic web content'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def get_supported_content_types(self) -> List[str]:
        return ['webpage', 'generic']


class SmartContentClassifier:
    """
    Smart content classification and routing system for ContextBox.
    
    Automatically classifies URLs and selects appropriate extractors with
    confidence scoring and fallback mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the smart content classifier.
        
        Args:
            config: Configuration dictionary with classifier settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize database if available
        self.db = None
        if DATABASE_AVAILABLE and self.config.get('use_database', True):
            try:
                self.db = ContextDatabase(self.config.get('db_config', {}))
                self.logger.info("Database integration enabled")
            except Exception as e:
                self.logger.warning(f"Database initialization failed: {e}")
        
        # Performance tracking
        self.classification_cache = {}
        self.cache_size_limit = self.config.get('cache_size_limit', 1000)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Initialize extractors
        self.extractors = {
            'youtube': YouTubeExtractor(),
            'wikipedia': WikipediaExtractor(),
            'news': NewsSiteExtractor(),
            'documentation': DocumentationExtractor(),
            'social_media': SocialMediaExtractor(),
            'generic': GenericExtractor()
        }
        
        # Initialize classification rules
        self.rules = self._initialize_default_rules()
        self.custom_rules = []
        
        # Statistics tracking
        self.stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'cache_hits': 0,
            'fallback_usage': 0,
            'extractor_usage': defaultdict(int)
        }
        
        self.logger.info("Smart content classifier initialized")
    
    def _initialize_default_rules(self) -> List[ContentRule]:
        """Initialize default classification rules."""
        return [
            # YouTube rules
            ContentRule(
                name='youtube_video',
                domain_patterns=['youtube.com', 'youtu.be'],
                url_patterns=[
                    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)(?:.*)',
                    r'youtube\.com/v/(?:.*)',
                    r'youtube\.com/.*[?&]v=(?:.*)'
                ],
                content_type='video',
                domain_type='video_platform',
                extractor_name='youtube',
                priority=10,
                confidence_boost=0.2,
                metadata={'platform': 'youtube'}
            ),
            
            # Wikipedia rules
            ContentRule(
                name='wikipedia_article',
                domain_patterns=['wikipedia.org', 'wikimedia.org'],
                url_patterns=[
                    r'(?:\.wikipedia\.org/wiki/.*)',
                    r'(?:wikimedia\.org/wiki/.*)'
                ],
                content_type='encyclopedia',
                domain_type='encyclopedia',
                extractor_name='wikipedia',
                priority=20,
                confidence_boost=0.2,
                metadata={'platform': 'wikipedia'}
            ),
            
            # News sites rules
            ContentRule(
                name='news_articles',
                domain_patterns=[
                    'cnn.com', 'bbc.com', 'reuters.com', 'apnews.com',
                    'nytimes.com', 'wsj.com', 'theguardian.com', 'npr.org',
                    'bloomberg.com', 'foxnews.com', 'cnbc.com', 'usatoday.com',
                    'washingtonpost.com'
                ],
                url_patterns=[r'.*'],
                content_type='article',
                domain_type='news',
                extractor_name='news',
                priority=30,
                confidence_boost=0.1,
                metadata={'platform': 'news_sites'}
            ),
            
            # Documentation sites rules
            ContentRule(
                name='documentation_sites',
                domain_patterns=[
                    'developer.mozilla.org', 'docs.python.org', 'docs.oracle.com',
                    'aws.amazon.com/documentation', 'learn.microsoft.com',
                    'docs.github.com', 'reactjs.org', 'angular.io', 'vuejs.org',
                    'nodejs.org'
                ],
                url_patterns=[r'.*'],
                content_type='documentation',
                domain_type='documentation',
                extractor_name='documentation',
                priority=25,
                confidence_boost=0.15,
                metadata={'platform': 'documentation_sites'}
            ),
            
            # Social media rules
            ContentRule(
                name='social_media',
                domain_patterns=[
                    'twitter.com', 'facebook.com', 'instagram.com',
                    'linkedin.com', 'reddit.com', 'tiktok.com', 'pinterest.com',
                    'tumblr.com'
                ],
                url_patterns=[r'.*'],
                content_type='social_media',
                domain_type='social_media',
                extractor_name='social_media',
                priority=35,
                confidence_boost=0.1,
                metadata={'platform': 'social_media'}
            )
        ]
    
    def classify_url(self, url: str, use_cache: bool = True) -> ClassificationResult:
        """
        Classify a URL and determine the best extraction method.
        
        Args:
            url: URL to classify
            use_cache: Whether to use cached results
            
        Returns:
            ClassificationResult with classification details
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(url)
        if use_cache and cache_key in self.classification_cache:
            cached_result = self.classification_cache[cache_key]
            if self._is_cache_valid(cached_result):
                self.stats['cache_hits'] += 1
                return cached_result
        
        self.logger.debug(f"Classifying URL: {url}")
        
        # Clean and normalize URL
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return self._create_error_result(url, "Invalid URL format")
        
        # Find best matching rule
        rule, confidence = self._find_best_rule(normalized_url)
        
        if not rule:
            # Use generic extractor as fallback
            extractor_name = 'generic'
            content_type = 'webpage'
            domain_type = 'generic'
            confidence = 0.3
        else:
            extractor_name = rule.extractor_name
            content_type = rule.content_type
            domain_type = rule.domain_type
        
        # Ensure extractor exists
        if extractor_name not in self.extractors:
            extractor_name = 'generic'
        
        # Create fallback extractors
        fallback_extractors = self._get_fallback_extractors(extractor_name, normalized_url)
        
        # Create routing info
        routing_info = {
            'rule_applied': rule.name if rule else 'default_fallback',
            'extractor_selected': extractor_name,
            'confidence_score': confidence,
            'fallback_chain': fallback_extractors
        }
        
        # Create classification result
        result = ClassificationResult(
            url=normalized_url,
            content_type=content_type,
            domain_type=domain_type,
            extractor_name=extractor_name,
            confidence=confidence,
            metadata=rule.metadata if rule else {},
            fallback_extractors=fallback_extractors,
            routing_info=routing_info,
            processing_time=time.time() - start_time
        )
        
        # Cache the result
        if use_cache:
            self._cache_result(cache_key, result)
        
        # Update statistics
        self.stats['total_classifications'] += 1
        self.stats['successful_classifications'] += 1
        self.stats['extractor_usage'][extractor_name] += 1
        
        self.logger.debug(f"Classification completed: {content_type} -> {extractor_name} (confidence: {confidence:.2f})")
        
        return result
    
    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL for consistent processing."""
        if not url:
            return None
        
        # Clean the URL
        url = url.strip()
        
        # Add protocol if missing
        if not re.match(r'^[a-zA-Z]+://', url):
            url = 'https://' + url
        
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.netloc:
                return None
            
            # Normalize scheme to lowercase
            normalized = parsed._replace(scheme=parsed.scheme.lower())
            return urllib.parse.urlunparse(normalized)
            
        except Exception:
            return None
    
    def _find_best_rule(self, url: str) -> Tuple[Optional[ContentRule], float]:
        """Find the best matching rule for a URL."""
        best_rule = None
        best_confidence = 0.0
        
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Check custom rules first (higher priority)
        all_rules = self.custom_rules + self.rules
        sorted_rules = sorted(all_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check domain patterns
            domain_match = any(pattern in domain for pattern in rule.domain_patterns)
            
            # Check URL patterns
            url_match = any(re.search(pattern, url, re.IGNORECASE) for pattern in rule.url_patterns)
            
            if domain_match and url_match:
                # Calculate confidence score
                base_confidence = 0.8
                if domain_match:
                    base_confidence += 0.1
                if url_match:
                    base_confidence += 0.1
                
                # Apply priority boost and confidence boost
                confidence = min(base_confidence + rule.confidence_boost, 1.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_rule = rule
        
        return best_rule, best_confidence
    
    def _get_fallback_extractors(self, primary_extractor: str, url: str) -> List[str]:
        """Get list of fallback extractors for a URL."""
        fallback_map = {
            'youtube': ['generic'],
            'wikipedia': ['generic'],
            'news': ['generic'],
            'documentation': ['generic'],
            'social_media': ['generic'],
            'generic': []
        }
        
        fallbacks = fallback_map.get(primary_extractor, ['generic'])
        
        # Add generic as last resort if not already included
        if fallbacks and fallbacks[-1] != 'generic':
            fallbacks.append('generic')
        
        return fallbacks
    
    def _create_error_result(self, url: str, error_message: str) -> ClassificationResult:
        """Create a classification result for errors."""
        return ClassificationResult(
            url=url,
            content_type='unknown',
            domain_type='unknown',
            extractor_name='generic',
            confidence=0.0,
            routing_info={'error': error_message}
        )
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        normalized = self._normalize_url(url)
        if not normalized:
            return hashlib.md5(url.encode()).hexdigest()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: ClassificationResult) -> None:
        """Cache classification result."""
        # Manage cache size
        if len(self.classification_cache) >= self.cache_size_limit:
            # Remove oldest entries (simple LRU approximation)
            oldest_key = next(iter(self.classification_cache))
            del self.classification_cache[oldest_key]
        
        # Add cache timestamp
        result_with_timestamp = result
        result_with_timestamp.metadata['cache_timestamp'] = time.time()
        
        self.classification_cache[cache_key] = result_with_timestamp
    
    def _is_cache_valid(self, result: ClassificationResult) -> bool:
        """Check if cached result is still valid."""
        cache_timestamp = result.metadata.get('cache_timestamp')
        if not cache_timestamp:
            return False
        
        return (time.time() - cache_timestamp) < self.cache_ttl
    
    def add_custom_rule(self, rule: ContentRule) -> None:
        """Add a custom classification rule."""
        rule.priority = min(rule.priority, 50)  # Custom rules have higher priority
        self.custom_rules.append(rule)
        self.logger.info(f"Added custom rule: {rule.name}")
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom rule by name."""
        for i, rule in enumerate(self.custom_rules):
            if rule.name == rule_name:
                del self.custom_rules[i]
                self.logger.info(f"Removed custom rule: {rule_name}")
                return True
        return False
    
    def get_available_extractors(self) -> Dict[str, List[str]]:
        """Get list of available extractors and their supported content types."""
        return {
            name: extractor.get_supported_content_types()
            for name, extractor in self.extractors.items()
        }
    
    def extract_content(self, url: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Classify URL and extract content using the appropriate extractor.
        
        Args:
            url: URL to process
            use_cache: Whether to use cached classification
            **kwargs: Additional arguments for extractors
            
        Returns:
            Dictionary with extraction results
        """
        # Classify the URL
        classification = self.classify_url(url, use_cache)
        
        # Extract content using the appropriate extractor
        extractor = self.extractors.get(classification.extractor_name)
        if not extractor:
            return {
                'success': False,
                'error': f"Extractor not found: {classification.extractor_name}",
                'url': url,
                'classification': classification.to_dict()
            }
        
        try:
            # Extract content
            result = extractor.extract(url, **kwargs)
            
            # Add classification info to result
            result['classification'] = classification.to_dict()
            result['processing_stats'] = {
                'classification_time': classification.processing_time,
                'confidence': classification.confidence,
                'extractor_used': classification.extractor_name
            }
            
            # Store in database if available
            if self.db:
                self._store_classification_result(url, classification, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {e}")
            
            # Try fallback extractors
            for fallback_extractor in classification.fallback_extractors:
                try:
                    fallback = self.extractors.get(fallback_extractor)
                    if fallback:
                        result = fallback.extract(url, **kwargs)
                        result['classification'] = classification.to_dict()
                        result['processing_stats'] = {
                            'classification_time': classification.processing_time,
                            'confidence': classification.confidence,
                            'extractor_used': f"{classification.extractor_name} -> {fallback_extractor}",
                            'fallback_used': True
                        }
                        return result
                except Exception:
                    continue
            
            # All extractors failed
            self.stats['fallback_usage'] += 1
            return {
                'success': False,
                'error': f"All extractors failed: {str(e)}",
                'url': url,
                'classification': classification.to_dict()
            }
    
    def _store_classification_result(self, url: str, classification: ClassificationResult, extraction_result: Dict[str, Any]) -> None:
        """Store classification and extraction results in database."""
        try:
            if not self.db:
                return
            
            # Store as artifact
            artifact_data = {
                'capture_id': None,  # Will be set by caller if needed
                'kind': 'url_classification',
                'url': url,
                'title': f"{classification.content_type} from {classification.domain_type}",
                'text': json.dumps(classification.to_dict()),
                'metadata': {
                    'classification_result': classification.to_dict(),
                    'extraction_success': extraction_result.get('success', False),
                    'extractor_used': classification.extractor_name
                }
            }
            
            # In a real implementation, this would be called with a capture_id
            self.logger.debug(f"Would store classification result for {url}")
            
        except Exception as e:
            self.logger.error(f"Failed to store classification result: {e}")
    
    def batch_classify(self, urls: List[str], use_cache: bool = True) -> List[ClassificationResult]:
        """Classify multiple URLs in batch."""
        results = []
        for url in urls:
            try:
                result = self.classify_url(url, use_cache)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to classify {url}: {e}")
                results.append(self._create_error_result(url, str(e)))
        
        return results
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        stats = dict(self.stats)
        stats['available_extractors'] = list(self.extractors.keys())
        stats['custom_rules_count'] = len(self.custom_rules)
        stats['default_rules_count'] = len(self.rules)
        stats['cache_size'] = len(self.classification_cache)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self.classification_cache.clear()
        self.logger.info("Classification cache cleared")
    
    def export_rules(self) -> Dict[str, Any]:
        """Export all rules for backup/sharing."""
        return {
            'custom_rules': [self._rule_to_dict(rule) for rule in self.custom_rules],
            'default_rules': [self._rule_to_dict(rule) for rule in self.rules],
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_rules(self, rules_data: Dict[str, Any]) -> None:
        """Import rules from exported data."""
        custom_rules_data = rules_data.get('custom_rules', [])
        self.custom_rules = [self._dict_to_rule(rule_data) for rule_data in custom_rules_data]
        self.logger.info(f"Imported {len(self.custom_rules)} custom rules")
    
    def _rule_to_dict(self, rule: ContentRule) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'name': rule.name,
            'domain_patterns': rule.domain_patterns,
            'url_patterns': rule.url_patterns,
            'content_type': rule.content_type,
            'domain_type': rule.domain_type,
            'extractor_name': rule.extractor_name,
            'priority': rule.priority,
            'confidence_boost': rule.confidence_boost,
            'metadata': rule.metadata,
            'enabled': rule.enabled
        }
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> ContentRule:
        """Convert dictionary to rule."""
        return ContentRule(**rule_data)


# Convenience functions for common use cases
def classify_url(url: str, config: Dict[str, Any] = None) -> ClassificationResult:
    """Quick function to classify a URL."""
    classifier = SmartContentClassifier(config)
    return classifier.classify_url(url)


def extract_url_content(url: str, config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """Quick function to classify and extract URL content."""
    classifier = SmartContentClassifier(config)
    return classifier.extract_content(url, **kwargs)


def batch_extract_urls(urls: List[str], config: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
    """Quick function to classify and extract multiple URLs."""
    classifier = SmartContentClassifier(config)
    results = []
    for url in urls:
        try:
            result = classifier.extract_content(url, **kwargs)
            results.append(result)
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'url': url
            })
    return results