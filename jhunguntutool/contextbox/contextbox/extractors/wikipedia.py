"""
Wikipedia content extraction module for ContextBox.

This module provides comprehensive Wikipedia content extraction using the MediaWiki API,
including content cleaning, summary generation, and integration with the ContextBox database.

Supported URL formats:
- https://en.wikipedia.org/wiki/Article_Title
- https://en.wikipedia.org/w/index.php?title=Article_Title
- https://de.wikipedia.org/wiki/Artikel_Titel (other languages)
"""

import logging
import re
import json
from urllib.parse import urlparse, urljoin, parse_qs, unquote
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class WikipediaExtractor:
    """
    Comprehensive Wikipedia content extractor using MediaWiki API.
    
    Handles multi-language Wikipedia sites, content cleaning, and extraction
    of text, links, images, references, and summaries.
    """
    
    # Supported Wikipedia domains
    WIKIPEDIA_DOMAINS = {
        'en.wikipedia.org',
        'de.wikipedia.org', 'fr.wikipedia.org', 'es.wikipedia.org', 'it.wikipedia.org',
        'pt.wikipedia.org', 'ru.wikipedia.org', 'ja.wikipedia.org', 'zh.wikipedia.org',
        'ar.wikipedia.org', 'nl.wikipedia.org', 'pl.wikipedia.org', 'sv.wikipedia.org',
        'cs.wikipedia.org', 'fi.wikipedia.org', 'hu.wikipedia.org', 'ko.wikipedia.org',
        'no.wikipedia.org', 'tr.wikipedia.org', 'th.wikipedia.org', 'vi.wikipedia.org'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Wikipedia extractor.
        
        Args:
            config: Configuration dictionary with optional settings:
                - user_agent: Custom User-Agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - api_language: Default language for API requests
                - extract_images: Whether to extract image information
                - extract_references: Whether to extract references
                - max_summary_length: Maximum length for generated summaries
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration settings
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.default_language = self.config.get('api_language', 'en')
        self.extract_images = self.config.get('extract_images', True)
        self.extract_references = self.config.get('extract_references', True)
        self.max_summary_length = self.config.get('max_summary_length', 500)
        
        # User agent for API requests
        self.user_agent = self.config.get('user_agent', 
            'ContextBox-WikipediaExtractor/1.0 (https://github.com/contextbox)')
        
        # Session with configured headers
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        if self.session:
            self.session.headers.update({
                'User-Agent': self.user_agent,
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9'
            })
        
        self.logger.info("Wikipedia extractor initialized")
    
    def extract_from_url(self, wikipedia_url: str) -> Dict[str, Any]:
        """
        Extract content from a Wikipedia URL.
        
        Args:
            wikipedia_url: Wikipedia URL to extract from
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Parse and validate the URL
            parsed_info = self._parse_wikipedia_url(wikipedia_url)
            if not parsed_info:
                return self._create_error_result("Invalid Wikipedia URL", wikipedia_url)
            
            page_title, language, wiki_site = parsed_info
            
            # Get page content using MediaWiki API
            api_content = self._fetch_page_content(page_title, language, wiki_site)
            if not api_content:
                return self._create_error_result("Failed to fetch page content", wikipedia_url)
            
            # Extract and process the content
            result = self._process_page_content(api_content, wikipedia_url, page_title, language)
            
            self.logger.info(f"Successfully extracted content from Wikipedia: {page_title}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract from Wikipedia URL {wikipedia_url}: {e}")
            return self._create_error_result(f"Extraction failed: {str(e)}", wikipedia_url)
    
    def _parse_wikipedia_url(self, url: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse a Wikipedia URL to extract page title, language, and domain.
        
        Args:
            url: Wikipedia URL to parse
            
        Returns:
            Tuple of (page_title, language, wiki_site) or None if invalid
        """
        try:
            parsed = urlparse(url)
            
            # Check if it's a Wikipedia domain
            if not any(domain in parsed.netloc for domain in self.WIKIPEDIA_DOMAINS):
                return None
            
            # Extract language from domain
            language = parsed.netloc.split('.')[0] if '.' in parsed.netloc else 'en'
            wiki_site = f"{language}.wikipedia.org"
            
            # Handle different URL formats
            if '/wiki/' in parsed.path:
                # Format: /wiki/Article_Title
                page_title = parsed.path.split('/wiki/')[-1]
                
            elif '/w/index.php' in parsed.path:
                # Format: /w/index.php?title=Article_Title
                query_params = parse_qs(parsed.query)
                page_title = query_params.get('title', [''])[0]
                
            else:
                # Try to extract from path
                page_title = parsed.path.split('/')[-1]
            
            # URL decode and clean the title
            if page_title:
                page_title = unquote(page_title.replace('_', ' ')).strip()
                return page_title, language, wiki_site
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to parse Wikipedia URL {url}: {e}")
            return None
    
    def _fetch_page_content(self, page_title: str, language: str, wiki_site: str) -> Optional[Dict[str, Any]]:
        """
        Fetch page content using MediaWiki API.
        
        Args:
            page_title: Title of the Wikipedia page
            language: Language code (e.g., 'en', 'de')
            wiki_site: Full wiki site domain
            
        Returns:
            API response data or None if failed
        """
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available")
            return None
        
        # MediaWiki API endpoint
        api_url = f"https://{wiki_site}/w/api.php"
        
        # API parameters for comprehensive content extraction
        params = {
            'action': 'parse',
            'page': page_title,
            'prop': 'text|categories|sections|links|images|references|parsetree',
            'format': 'json',
            'formatversion': 2,
            'disablepp': True,
            'continue': ''
        }
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Fetching page content (attempt {attempt + 1}): {page_title}")
                
                response = self.session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if 'error' in data:
                    self.logger.error(f"MediaWiki API error: {data['error']}")
                    return None
                
                if 'parse' not in data:
                    self.logger.warning(f"No parse data returned for {page_title}")
                    return data  # May contain redirects or other info
                
                return data['parse']
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to fetch page content after {self.max_retries} attempts")
                    return None
                
                # Wait before retry with exponential backoff
                import time
                time.sleep(2 ** attempt)
        
        return None
    
    def _process_page_content(self, api_content: Dict[str, Any], original_url: str, 
                            page_title: str, language: str) -> Dict[str, Any]:
        """
        Process the raw API content and extract structured information.
        
        Args:
            api_content: Raw API response data
            original_url: Original Wikipedia URL
            page_title: Page title
            language: Language code
            
        Returns:
            Processed and structured content data
        """
        # Handle case where api_content might be a string or other unexpected format
        if not isinstance(api_content, dict):
            return self._create_error_result(f"Unexpected API response format: {type(api_content)}", original_url)
        
        result = {
            'type': 'wikipedia_extraction',
            'timestamp': datetime.now().isoformat(),
            'source_url': original_url,
            'metadata': {
                'title': page_title,
                'language': language,
                'extraction_method': 'mediawiki_api'
            },
            'content': {
                'raw_html': api_content.get('text', ''),
                'main_text': '',
                'summary': '',
                'lead_section': ''
            },
            'structure': {
                'sections': [],
                'categories': [],
                'infobox': {}
            },
            'links': {
                'internal': [],
                'external': []
            },
            'media': {
                'images': [],
                'other_media': []
            },
            'references': {
                'count': 0,
                'list': []
            },
            'statistics': {
                'word_count': 0,
                'character_count': 0,
                'extraction_confidence': 0.0
            }
        }
        
        # Process HTML content
        if 'text' in api_content:
            html_content = api_content['text']
            processed_content = self._extract_and_clean_html(html_content, language)
            
            result['content'].update(processed_content)
        
        # Extract sections
        if 'sections' in api_content:
            result['structure']['sections'] = self._extract_sections(api_content['sections'])
        
        # Extract categories
        if 'categories' in api_content:
            result['structure']['categories'] = [
                cat.get('title', '').replace('Category:', '') 
                for cat in api_content['categories'] if 'title' in cat
            ]
        
        # Extract links
        if 'links' in api_content:
            result['links']['internal'] = [
                link.get('title', '') for link in api_content['links'][:100]  # Limit to first 100
            ]
        
        # Extract images
        if self.extract_images and 'images' in api_content:
            wiki_site = f"{language}.wikipedia.org"
            result['media']['images'] = self._extract_image_info(api_content['images'], wiki_site)
        
        # Extract references
        if self.extract_references and 'references' in api_content:
            result['references'] = self._extract_references_info(api_content['references'])
        
        # Calculate statistics
        result['statistics'] = self._calculate_content_statistics(result['content'])
        
        # Generate summary if not provided by API
        if not result['content'].get('summary'):
            result['content']['summary'] = self._generate_summary(
                result['content']['main_text'], 
                result['content']['lead_section']
            )
        
        return result
    
    def _extract_and_clean_html(self, html_content: str, language: str) -> Dict[str, str]:
        """
        Extract and clean HTML content to get readable text.
        
        Args:
            html_content: Raw HTML content from MediaWiki API
            language: Language code for text processing
            
        Returns:
            Dictionary containing cleaned text content
        """
        if not BS4_AVAILABLE or not html_content:
            return {
                'raw_html': html_content,
                'main_text': '',
                'lead_section': ''
            }
        
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Extract lead section (first section before first heading)
            lead_section = self._extract_lead_section(soup)
            
            # Extract main text content
            main_text = self._extract_main_text(soup)
            
            # Clean and format the text
            lead_section = self._clean_text_content(lead_section)
            main_text = self._clean_text_content(main_text)
            
            return {
                'raw_html': html_content,
                'main_text': main_text,
                'lead_section': lead_section
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to process HTML content: {e}")
            return {
                'raw_html': html_content,
                'main_text': self._strip_html_tags(html_content),
                'lead_section': ''
            }
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """
        Remove unwanted HTML elements that shouldn't be in the main content.
        
        Args:
            soup: BeautifulSoup object to clean
        """
        # Remove elements that typically contain navigation/content we don't want
        unwanted_selectors = [
            'table.infobox',
            '.navbox',
            '.vertical-navbox',
            '.wikitable',
            '.reference',
            '.mw-editsection',
            '.mw-editsection-bracket',
            '.external',
            '.reference-text',
            'sup.reference',
            'a[href^="#"]',  # Anchor links within page
            '.mw-cite-backlink',
            '.mw-references-wrap',
            '.printonly',
            '.noprint'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove script and style elements
        for element in soup.find_all(['script', 'style']):
            element.decompose()
        
        # Remove reference numbers and citations
        for sup in soup.find_all('sup'):
            if 'reference' in sup.get('class', []) or 'cite' in sup.get('class', []):
                sup.decompose()
    
    def _extract_lead_section(self, soup: BeautifulSoup) -> str:
        """
        Extract the lead section (intro) of the article.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Lead section text
        """
        # Find the first heading
        first_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if first_heading:
            # Get content until the next heading
            lead_text = []
            current = first_heading.next_sibling
            
            while current:
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                
                if isinstance(current, NavigableString):
                    text = str(current).strip()
                    if text:
                        lead_text.append(text)
                elif isinstance(current, Tag):
                    # Handle inline elements
                    text = current.get_text().strip()
                    if text:
                        lead_text.append(text)
                
                current = current.next_sibling
            
            return ' '.join(lead_text)
        
        # Fallback: get first few paragraphs
        paragraphs = soup.find_all('p')[:2]
        lead_paragraphs = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        return '\n\n'.join(lead_paragraphs)
    
    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        """
        Extract main article text content.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Main article text
        """
        # Get all paragraphs in the main content area
        paragraphs = soup.find_all('p')
        
        main_text_parts = []
        for p in paragraphs:
            # Skip paragraphs in infoboxes or other unwanted areas
            if p.find_parent(['table', 'div']):
                continue
            
            text = p.get_text().strip()
            if text and len(text) > 20:  # Filter out very short paragraphs
                main_text_parts.append(text)
        
        return '\n\n'.join(main_text_parts)
    
    def _clean_text_content(self, text: str) -> str:
        """
        Clean and format text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and formatted text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove unwanted characters and formatting
        text = re.sub(r'\[edit\]', '', text)  # Remove "edit" links
        text = re.sub(r'\[citation needed\]', '', text)  # Remove citation placeholders
        
        # Clean up common artifacts
        text = text.replace('  ', ' ').strip()
        
        # Remove leading/trailing newlines
        text = text.strip()
        
        return text
    
    def _strip_html_tags(self, html: str) -> str:
        """
        Simple HTML tag stripping as fallback.
        
        Args:
            html: HTML content
            
        Returns:
            Text content with HTML tags removed
        """
        # Simple regex-based HTML stripping
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract section information from API data.
        
        Args:
            sections: List of section data from API
            
        Returns:
            List of section information
        """
        processed_sections = []
        
        for section in sections:
            processed_sections.append({
                'title': section.get('line', ''),
                'level': section.get('level', 0),
                'index': section.get('index', '')
            })
        
        return processed_sections
    
    def _extract_image_info(self, images: List[Dict[str, Any]], wiki_site: str) -> List[Dict[str, str]]:
        """
        Extract image information from API data.
        
        Args:
            images: List of image data from API
            wiki_site: Wikipedia site domain
            
        Returns:
            List of image information
        """
        processed_images = []
        
        for image in images[:20]:  # Limit to first 20 images
            image_title = image.get('title', '')
            if image_title:
                processed_images.append({
                    'title': image_title,
                    'wiki_url': f"https://{wiki_site}/wiki/{image_title.replace(' ', '_')}",
                    'description_url': f"https://{wiki_site}/wiki/File:{image_title.replace(' ', '_')}"
                })
        
        return processed_images
    
    def _extract_references_info(self, references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract reference information from API data.
        
        Args:
            references: List of reference data from API
            
        Returns:
            Dictionary with reference count and list
        """
        ref_count = len(references)
        ref_list = []
        
        for i, ref in enumerate(references[:50]):  # Limit to first 50 references
            ref_text = ref.get('text', '') or ref.get('title', '')
            if ref_text:
                ref_list.append({
                    'index': i + 1,
                    'text': ref_text[:200] + '...' if len(ref_text) > 200 else ref_text
                })
        
        return {
            'count': ref_count,
            'list': ref_list
        }
    
    def _generate_summary(self, main_text: str, lead_section: str) -> str:
        """
        Generate a summary from the extracted content.
        
        Args:
            main_text: Main article text
            lead_section: Lead section text
            
        Returns:
            Generated summary
        """
        # Prefer lead section if available
        content_to_summarize = lead_section if lead_section else main_text
        
        if not content_to_summarize:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content_to_summarize)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return ""
        
        # Take first few sentences to create a summary
        max_sentences = 3
        summary_sentences = sentences[:max_sentences]
        
        # Ensure we don't exceed the maximum length
        summary = '. '.join(summary_sentences)
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + '...'
        
        return summary
    
    def _calculate_content_statistics(self, content: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate statistics about the extracted content.
        
        Args:
            content: Dictionary containing content fields
            
        Returns:
            Dictionary with content statistics
        """
        # Get the main text content
        text_content = []
        for field in ['main_text', 'lead_section', 'summary']:
            if content.get(field):
                text_content.append(content[field])
        
        combined_text = ' '.join(text_content)
        
        # Calculate statistics
        word_count = len(combined_text.split())
        character_count = len(combined_text)
        
        # Calculate confidence based on content availability and quality
        confidence = 0.0
        if content.get('main_text'):
            confidence += 0.4
        if content.get('lead_section'):
            confidence += 0.3
        if content.get('summary'):
            confidence += 0.2
        if content.get('raw_html'):
            confidence += 0.1
        
        return {
            'word_count': word_count,
            'character_count': character_count,
            'extraction_confidence': min(confidence, 1.0)
        }
    
    def _create_error_result(self, error_message: str, source_url: str) -> Dict[str, Any]:
        """
        Create a standardized error result.
        
        Args:
            error_message: Error message to include
            source_url: Source URL that caused the error
            
        Returns:
            Error result dictionary
        """
        return {
            'type': 'wikipedia_extraction',
            'timestamp': datetime.now().isoformat(),
            'source_url': source_url,
            'error': error_message,
            'content': {
                'raw_html': '',
                'main_text': '',
                'summary': '',
                'lead_section': ''
            },
            'structure': {
                'sections': [],
                'categories': [],
                'infobox': {}
            },
            'links': {
                'internal': [],
                'external': []
            },
            'media': {
                'images': [],
                'other_media': []
            },
            'references': {
                'count': 0,
                'list': []
            },
            'statistics': {
                'word_count': 0,
                'character_count': 0,
                'extraction_confidence': 0.0
            }
        }
    
    def extract_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple Wikipedia URLs.
        
        Args:
            urls: List of Wikipedia URLs to extract from
            
        Returns:
            List of extraction results
        """
        results = []
        
        for url in urls:
            self.logger.info(f"Extracting content from: {url}")
            result = self.extract_from_url(url)
            results.append(result)
        
        return results
    
    def save_extraction_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save extraction result to JSON file.
        
        Args:
            result: Extraction result dictionary
            output_path: Path to save the JSON file
        """
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Extraction result saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save extraction result: {e}")
            raise


# ContextBox database integration functions

def create_wikipedia_artifact_data(extraction_result: Dict[str, Any], 
                                 capture_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert Wikipedia extraction result to ContextBox artifact format.
    
    Args:
        extraction_result: Wikipedia extraction result
        capture_id: Optional capture ID for linking
        
    Returns:
        Artifact data dictionary suitable for ContextBox database
    """
    metadata = {
        'extraction_type': 'wikipedia',
        'language': extraction_result.get('metadata', {}).get('language', 'en'),
        'confidence': extraction_result.get('statistics', {}).get('extraction_confidence', 0.0),
        'word_count': extraction_result.get('statistics', {}).get('word_count', 0),
        'sections_count': len(extraction_result.get('structure', {}).get('sections', [])),
        'images_count': len(extraction_result.get('media', {}).get('images', [])),
        'references_count': extraction_result.get('references', {}).get('count', 0),
        'categories': extraction_result.get('structure', {}).get('categories', []),
        'internal_links_count': len(extraction_result.get('links', {}).get('internal', [])),
        'timestamp': extraction_result.get('timestamp'),
        'extraction_method': extraction_result.get('metadata', {}).get('extraction_method', 'mediawiki_api')
    }
    
    # Add content statistics to metadata
    metadata.update(extraction_result.get('statistics', {}))
    
    return {
        'kind': 'wikipedia_article',
        'url': extraction_result.get('source_url'),
        'title': extraction_result.get('metadata', {}).get('title'),
        'text': extraction_result.get('content', {}).get('main_text', ''),
        'metadata': metadata
    }


def create_summary_artifact_data(extraction_result: Dict[str, Any], 
                               capture_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a separate artifact for the Wikipedia article summary.
    
    Args:
        extraction_result: Wikipedia extraction result
        capture_id: Optional capture ID for linking
        
    Returns:
        Summary artifact data dictionary
    """
    summary_content = extraction_result.get('content', {}).get('summary', '')
    
    if not summary_content:
        return None  # Don't create summary artifact if no summary available
    
    summary_metadata = {
        'extraction_type': 'wikipedia_summary',
        'source_title': extraction_result.get('metadata', {}).get('title'),
        'source_url': extraction_result.get('source_url'),
        'language': extraction_result.get('metadata', {}).get('language', 'en'),
        'summary_length': len(summary_content),
        'generation_method': 'extracted_from_wikipedia',
        'confidence': extraction_result.get('statistics', {}).get('extraction_confidence', 0.0),
        'timestamp': extraction_result.get('timestamp')
    }
    
    return {
        'kind': 'wikipedia_summary',
        'url': extraction_result.get('source_url'),
        'title': f"Summary: {extraction_result.get('metadata', {}).get('title')}",
        'text': summary_content,
        'metadata': summary_metadata
    }


# Convenience functions for common use cases

def extract_wikipedia_content(url: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick function to extract Wikipedia content.
    
    Args:
        url: Wikipedia URL to extract from
        config: Optional configuration dictionary
        
    Returns:
        Extraction result dictionary
    """
    extractor = WikipediaExtractor(config)
    return extractor.extract_from_url(url)


def extract_and_store_wikipedia(capture_id: int, wikipedia_url: str, 
                               database, config: Optional[Dict[str, Any]] = None) -> int:
    """
    Extract Wikipedia content and store it directly in ContextBox database.
    
    Args:
        capture_id: ContextBox capture ID to link artifacts to
        wikipedia_url: Wikipedia URL to extract from
        database: ContextBox database instance
        config: Optional configuration dictionary
        
    Returns:
        Artifact ID of the created Wikipedia artifact
    """
    # Extract content
    extractor = WikipediaExtractor(config)
    result = extractor.extract_from_url(wikipedia_url)
    
    if 'error' in result:
        raise ValueError(f"Wikipedia extraction failed: {result['error']}")
    
    # Create artifacts
    main_artifact_data = create_wikipedia_artifact_data(result, capture_id)
    summary_artifact_data = create_summary_artifact_data(result, capture_id)
    
    # Store main artifact
    artifact_id = database.create_artifact(
        capture_id=capture_id,
        kind=main_artifact_data['kind'],
        url=main_artifact_data['url'],
        title=main_artifact_data['title'],
        text=main_artifact_data['text'],
        metadata=main_artifact_data['metadata']
    )
    
    # Store summary artifact if available
    if summary_artifact_data:
        database.create_artifact(
            capture_id=capture_id,
            kind=summary_artifact_data['kind'],
            url=summary_artifact_data['url'],
            title=summary_artifact_data['title'],
            text=summary_artifact_data['text'],
            metadata=summary_artifact_data['metadata']
        )
    
    return artifact_id


# Example usage and testing functions

def test_wikipedia_extraction():
    """
    Test function to demonstrate Wikipedia extraction capabilities.
    """
    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz",
        "https://en.wikipedia.org/w/index.php?title=Machine_learning"
    ]
    
    config = {
        'extract_images': True,
        'extract_references': True,
        'max_summary_length': 300
    }
    
    extractor = WikipediaExtractor(config)
    
    for url in test_urls:
        print(f"\nExtracting: {url}")
        result = extractor.extract_from_url(url)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Title: {result['metadata']['title']}")
            print(f"Language: {result['metadata']['language']}")
            print(f"Word Count: {result['statistics']['word_count']}")
            print(f"Summary: {result['content']['summary'][:200]}...")


if __name__ == "__main__":
    test_wikipedia_extraction()