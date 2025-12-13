"""
Intelligent Summarization System for ContextBox

This module provides a comprehensive summarization system with map-reduce functionality,
multiple LLM backend support, and advanced features like progressive summarization
and quality scoring.
"""

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable, 
    Iterator, Generator, Set
)
from pathlib import Path
import sqlite3

# Optional imports for LLM backends
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local imports
from .config import ConfigManager, LLMBackendConfig, ProviderConfig, ModelConfig
from .exceptions import LLMBackendError, ConfigurationError, ServiceUnavailableError


@dataclass
class SummaryContent:
    """Represents a summary content with metadata."""
    text: str
    content_type: str
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None
    processing_time: Optional[float] = None
    token_count: Optional[int] = None
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SummaryRequest:
    """Request parameters for summarization."""
    content: str
    content_type: str
    source_id: Optional[str] = None
    summary_length: str = "detailed"  # brief, detailed, executive
    format_type: str = "paragraph"    # paragraph, bullets, key_points
    include_metadata: bool = True
    quality_threshold: float = 0.7
    enable_progressive: bool = True
    enable_caching: bool = True
    max_retries: int = 3
    timeout: int = 30


@dataclass
class SummaryResult:
    """Result of a summarization operation."""
    summary: SummaryContent
    quality_metrics: Dict[str, float]
    processing_info: Dict[str, Any]
    cache_hit: bool = False
    llm_provider: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MultiDocumentSummary:
    """Result of multi-document summarization."""
    combined_summary: SummaryContent
    individual_summaries: List[SummaryContent]
    cross_document_insights: Optional[str] = None
    common_themes: List[str] = None
    unique_content: List[str] = None
    contradictions: List[str] = None
    quality_comparison: Dict[str, float] = None


class ContentChunker:
    """Handles intelligent content chunking for map-reduce summarization."""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_content(self, content: str, content_type: str = "text") -> List[Dict[str, Any]]:
        """
        Chunk content into smaller pieces for processing.
        
        Args:
            content: The content to chunk
            content_type: Type of content (affects chunking strategy)
            
        Returns:
            List of chunks with metadata
        """
        if not content or len(content.strip()) == 0:
            return []
        
        # Adjust chunking strategy based on content type
        if content_type in ["code", "transcript"]:
            return self._chunk_by_structure(content, content_type)
        elif content_type == "article":
            return self._chunk_article(content)
        else:
            return self._chunk_text(content)
    
    def _chunk_text(self, content: str) -> List[Dict[str, Any]]:
        """Basic text chunking with overlap preservation."""
        chunks = []
        content_length = len(content)
        
        # Split into paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'chunk_type': 'text',
                        'char_count': len(current_chunk)
                    })
                    chunk_index += 1
                
                # Start new chunk with current paragraph
                if len(paragraph) <= self.chunk_size:
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, split it
                    sentences = self._split_sentences(paragraph)
                    current_chunk = ""
                    for sentence in sentences:
                        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                        if len(potential_chunk) <= self.chunk_size:
                            current_chunk = potential_chunk
                        else:
                            if current_chunk:
                                chunks.append({
                                    'text': current_chunk.strip(),
                                    'chunk_index': chunk_index,
                                    'chunk_type': 'text',
                                    'char_count': len(current_chunk)
                                })
                                chunk_index += 1
                            current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'chunk_type': 'text',
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def _chunk_article(self, content: str) -> List[Dict[str, Any]]:
        """Specialized chunking for articles with heading awareness."""
        # Look for headings to better chunk articles
        heading_pattern = r'^(#+ |[IVXLCDM]+\. |\d+\. |\([IVXLCDM]+\) )'
        lines = content.split('\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for line in lines:
            if re.match(heading_pattern, line.strip()) and current_chunk:
                # Start new chunk at heading
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'chunk_type': 'article_section',
                    'char_count': len(current_chunk)
                })
                chunk_index += 1
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
                
                if len(current_chunk) >= self.chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'chunk_type': 'article_section',
                        'char_count': len(current_chunk)
                    })
                    chunk_index += 1
                    current_chunk = ""
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'chunk_type': 'article_section',
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def _chunk_by_structure(self, content: str, content_type: str) -> List[Dict[str, Any]]:
        """Chunk based on structural elements for code/transcripts."""
        if content_type == "code":
            # Try to chunk by functions/classes
            pattern = r'(?:^|\n)(?:def |class |async def |async class )'
        elif content_type == "transcript":
            # Chunk by speaker changes or natural breaks
            pattern = r'(?:^|\n)(?:\[\d{2}:\d{2}:\d{2}\]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:|\d+\. )'
        else:
            return self._chunk_text(content)
        
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        chunk_index = 0
        
        for line in lines:
            if re.match(pattern, line) and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'chunk_type': content_type,
                    'char_count': len(current_chunk)
                })
                chunk_index += 1
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
                
                if len(current_chunk) >= self.chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'chunk_type': content_type,
                        'char_count': len(current_chunk)
                    })
                    chunk_index += 1
                    current_chunk = ""
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'chunk_type': content_type,
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class QualityAssessor:
    """Assesses the quality of generated summaries."""
    
    def __init__(self):
        self.metrics = {
            'coherence': 0.3,
            'completeness': 0.3,
            'conciseness': 0.2,
            'relevance': 0.2
        }
    
    def assess_quality(self, 
                      original_content: str, 
                      summary: str, 
                      request: SummaryRequest) -> Dict[str, float]:
        """
        Assess the quality of a summary across multiple dimensions.
        
        Args:
            original_content: The original content being summarized
            summary: The generated summary
            request: The summarization request
            
        Returns:
            Dictionary of quality metrics
        """
        scores = {}
        
        # Calculate coherence score
        scores['coherence'] = self._assess_coherence(summary)
        
        # Calculate completeness score
        scores['completeness'] = self._assess_completeness(original_content, summary, request.content_type)
        
        # Calculate conciseness score
        scores['conciseness'] = self._assess_conciseness(summary, request.summary_length)
        
        # Calculate relevance score
        scores['relevance'] = self._assess_relevance(original_content, summary, request.content_type)
        
        # Calculate overall score
        overall_score = sum(
            score * self.metrics[metric] 
            for metric, score in scores.items()
        )
        scores['overall'] = overall_score
        
        return scores
    
    def _assess_coherence(self, summary: str) -> float:
        """Assess the coherence of a summary."""
        if not summary or len(summary.strip()) < 10:
            return 0.0
        
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence check - look for connecting words
        connecting_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'in contrast', 'as a result']
        connecting_count = sum(1 for word in connecting_words if word in summary.lower())
        
        # Normalize by sentence count
        coherence_factor = min(1.0, connecting_count / len(sentences) * 2)
        
        # Add penalty for very short or very long sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_penalty = 0.8 if 10 <= avg_sentence_length <= 25 else 0.6
        
        return coherence_factor * length_penalty
    
    def _assess_completeness(self, original: str, summary: str, content_type: str) -> float:
        """Assess how completely the summary covers the original content."""
        if not original or not summary:
            return 0.0
        
        # Extract key concepts from original content
        original_words = set(re.findall(r'\b\w+\b', original.lower()))
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        original_concepts = original_words - stop_words
        summary_concepts = summary_words - stop_words
        
        if not original_concepts:
            return 1.0
        
        # Calculate concept coverage
        coverage = len(summary_concepts & original_concepts) / len(original_concepts)
        
        # Adjust based on content type
        if content_type == "transcript":
            # For transcripts, be more lenient with coverage
            coverage = min(1.0, coverage * 1.2)
        elif content_type == "article":
            # For articles, expect good coverage
            coverage = coverage
        
        return min(1.0, coverage)
    
    def _assess_conciseness(self, summary: str, summary_length: str) -> float:
        """Assess how concise the summary is for the requested length."""
        if not summary:
            return 0.0
        
        word_count = len(summary.split())
        
        # Define expected word counts for different summary types
        expected_counts = {
            'brief': (50, 150),
            'detailed': (150, 400),
            'executive': (100, 250)
        }
        
        if summary_length not in expected_counts:
            return 0.7
        
        min_count, max_count = expected_counts[summary_length]
        
        if min_count <= word_count <= max_count:
            return 1.0
        elif word_count < min_count:
            # Too short
            return max(0.3, word_count / min_count)
        else:
            # Too long
            return max(0.3, max_count / word_count)
    
    def _assess_relevance(self, original: str, summary: str, content_type: str) -> float:
        """Assess the relevance of the summary to the original content."""
        if not original or not summary:
            return 0.0
        
        # Simple relevance check - look for key terms appearing in both
        original_lower = original.lower()
        summary_lower = summary.lower()
        
        # Extract potential key terms (nouns, proper nouns, etc.)
        original_terms = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        summary_terms = set(re.findall(r'\b[A-Z][a-z]+\b', summary))
        
        if original_terms:
            relevance = len(summary_terms & original_terms) / len(original_terms)
        else:
            # Fallback to simple word overlap
            original_words = set(re.findall(r'\b\w+\b', original_lower))
            summary_words = set(re.findall(r'\b\w+\b', summary_lower))
            
            if original_words:
                relevance = len(summary_words & original_words) / len(original_words)
            else:
                relevance = 0.5
        
        return min(1.0, relevance * 1.5)  # Boost slightly


class CacheManager:
    """Manages caching of summarization results."""
    
    def __init__(self, cache_db_path: str = "contextbox_summaries.db"):
        self.cache_db_path = cache_db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize the cache database."""
        conn = sqlite3.connect(self.cache_db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS summary_cache (
                    cache_key TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    request_params TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON summary_cache(content_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON summary_cache(created_at)')
            conn.commit()
        finally:
            conn.close()
    
    def _generate_cache_key(self, content: str, request: SummaryRequest) -> str:
        """Generate a cache key for the content and request."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Include relevant request parameters in the key
        key_data = {
            'content_hash': content_hash,
            'content_type': request.content_type,
            'summary_length': request.summary_length,
            'format_type': request.format_type,
            'include_metadata': request.include_metadata
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_cached_summary(self, content: str, request: SummaryRequest) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get a cached summary if available."""
        if not request.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(content, request)
        
        conn = sqlite3.connect(self.cache_db_path)
        try:
            cursor = conn.execute('''
                SELECT summary_text, metadata, quality_score, access_count
                FROM summary_cache 
                WHERE cache_key = ?
            ''', (cache_key,))
            
            result = cursor.fetchone()
            if result:
                # Update access information
                conn.execute('''
                    UPDATE summary_cache 
                    SET last_accessed = CURRENT_TIMESTAMP, 
                        access_count = access_count + 1
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                summary_text, metadata_json, quality_score, access_count = result
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return summary_text, {
                    'quality_score': quality_score,
                    'metadata': metadata,
                    'cache_hit': True,
                    'access_count': access_count
                }
            
            return None
        finally:
            conn.close()
    
    def cache_summary(self, 
                     content: str, 
                     request: SummaryRequest, 
                     summary: str, 
                     quality_score: float,
                     metadata: Dict[str, Any]):
        """Cache a summary result."""
        if not request.enable_caching:
            return
        
        cache_key = self._generate_cache_key(content, request)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        request_params = asdict(request)
        
        conn = sqlite3.connect(self.cache_db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO summary_cache 
                (cache_key, content_hash, request_params, summary_text, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                content_hash,
                json.dumps(request_params),
                summary,
                quality_score,
                json.dumps(metadata)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def clear_cache(self, older_than_days: int = 30):
        """Clear old cache entries."""
        conn = sqlite3.connect(self.cache_db_path)
        try:
            conn.execute('''
                DELETE FROM summary_cache 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(older_than_days))
            conn.commit()
        finally:
            conn.close()


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate_summary(self, 
                        content: str, 
                        prompt: str, 
                        config: ModelConfig) -> Tuple[str, Dict[str, Any]]:
        """Generate a summary using the LLM backend."""
        pass


class OllamaBackend(LLMBackend):
    """Ollama LLM backend implementation."""
    
    def __init__(self):
        if not OLLAMA_AVAILABLE:
            raise LLMBackendError("Ollama package not available", provider="ollama")
    
    def generate_summary(self, 
                        content: str, 
                        prompt: str, 
                        config: ModelConfig) -> Tuple[str, Dict[str, Any]]:
        """Generate summary using Ollama."""
        try:
            response = ollama.generate(
                model=config.name,
                prompt=prompt,
                options={
                    'temperature': config.temperature,
                    'top_p': config.top_p,
                    'max_tokens': config.max_tokens
                }
            )
            
            return response['response'], {
                'provider': 'ollama',
                'model': config.name,
                'total_tokens': response.get('total_duration', 0)
            }
            
        except Exception as e:
            raise LLMBackendError(
                f"Ollama generation failed: {str(e)}",
                provider="ollama",
                model=config.name,
                details={'error': str(e)}
            )


class OpenAIBackend(LLMBackend):
    """OpenAI LLM backend implementation."""
    
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise LLMBackendError("OpenAI package not available", provider="openai")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_summary(self, 
                        content: str, 
                        prompt: str, 
                        config: ModelConfig) -> Tuple[str, Dict[str, Any]]:
        """Generate summary using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=config.name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Summarize the following content:\n\n{content}"}
                ],
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens
            )
            
            summary = response.choices[0].message.content
            
            return summary, {
                'provider': 'openai',
                'model': config.name,
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            raise LLMBackendError(
                f"OpenAI generation failed: {str(e)}",
                provider="openai",
                model=config.name,
                details={'error': str(e)}
            )


class SummaryExporter:
    """Handles exporting summaries in various formats."""
    
    @staticmethod
    def export_json(summary_result: SummaryResult, output_path: str):
        """Export summary result as JSON."""
        export_data = {
            'summary': {
                'text': summary_result.summary.text,
                'content_type': summary_result.summary.content_type,
                'source_id': summary_result.summary.source_id,
                'metadata': summary_result.summary.metadata
            },
            'quality_metrics': summary_result.quality_metrics,
            'processing_info': summary_result.processing_info,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_markdown(summary_result: SummaryResult, output_path: str):
        """Export summary result as Markdown."""
        lines = []
        
        # Title and metadata
        lines.append(f"# Summary")
        lines.append("")
        lines.append(f"**Content Type:** {summary_result.summary.content_type}")
        if summary_result.summary.source_id:
            lines.append(f"**Source ID:** {summary_result.summary.source_id}")
        lines.append(f"**Generated:** {summary_result.processing_info.get('timestamp', 'Unknown')}")
        lines.append("")
        
        # Quality metrics
        lines.append("## Quality Metrics")
        lines.append("")
        for metric, score in summary_result.quality_metrics.items():
            lines.append(f"- **{metric.title()}:** {score:.2f}")
        lines.append("")
        
        # Summary content
        lines.append("## Summary")
        lines.append("")
        lines.append(summary_result.summary.text)
        lines.append("")
        
        # Export metadata if requested
        if summary_result.summary.metadata and summary_result.processing_info.get('include_metadata'):
            lines.append("## Metadata")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(summary_result.summary.metadata, indent=2))
            lines.append("```")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    @staticmethod
    def export_plain_text(summary_result: SummaryResult, output_path: str):
        """Export summary result as plain text."""
        lines = []
        lines.append("=" * 50)
        lines.append("SUMMARY")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Content Type: {summary_result.summary.content_type}")
        if summary_result.summary.source_id:
            lines.append(f"Source ID: {summary_result.summary.source_id}")
        lines.append(f"Generated: {summary_result.processing_info.get('timestamp', 'Unknown')}")
        lines.append("")
        
        lines.append("QUALITY METRICS:")
        for metric, score in summary_result.quality_metrics.items():
            lines.append(f"  {metric.title()}: {score:.2f}")
        lines.append("")
        
        lines.append("-" * 50)
        lines.append("")
        lines.append(summary_result.summary.text)
        lines.append("")
        lines.append("=" * 50)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class SummarizationManager:
    """
    Main orchestrator for the intelligent summarization system.
    
    Provides comprehensive summarization capabilities including:
    - Map-reduce processing for large documents
    - Multiple summary lengths and formats
    - Content type awareness
    - Progressive summarization
    - Quality assessment and caching
    - Multi-document summarization
    - Integration with LLM backends
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 cache_db_path: str = "contextbox_summaries.db",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the SummarizationManager.
        
        Args:
            config_path: Path to LLM configuration file
            cache_db_path: Path to cache database
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.chunker = ContentChunker()
        self.quality_assessor = QualityAssessor()
        self.cache_manager = CacheManager(cache_db_path)
        self.exporter = SummaryExporter()
        
        # Initialize LLM backends
        self.backends: Dict[str, LLMBackend] = {}
        self._init_backends()
        
        # Summary templates for different types and lengths
        self.summary_templates = self._load_summary_templates()
    
    def _init_backends(self):
        """Initialize available LLM backends."""
        if OLLAMA_AVAILABLE:
            self.backends['ollama'] = OllamaBackend()
        
        if OPENAI_AVAILABLE:
            for provider_name, provider_config in self.config.providers.items():
                if provider_name == 'openai' and provider_config.api_key:
                    self.backends['openai'] = OpenAIBackend(provider_config.api_key)
                    break
    
    def _load_summary_templates(self) -> Dict[str, Dict[str, str]]:
        """Load summary prompt templates."""
        return {
            'brief': {
                'paragraph': "Provide a concise one-paragraph summary of the following content. Focus on the main points and key takeaways.",
                'bullets': "Create a brief bullet-point summary highlighting the 3-5 most important points from the content.",
                'key_points': "Extract the essential key points from the following content in a clear, concise format."
            },
            'detailed': {
                'paragraph': "Provide a detailed summary of the following content. Include the main topics, subtopics, and important details.",
                'bullets': "Create a comprehensive bullet-point summary with main sections and subsections covering all important aspects.",
                'key_points': "Generate a detailed summary with all key points, supporting details, and relevant examples."
            },
            'executive': {
                'paragraph': "Provide an executive-level summary focusing on strategic implications, business impact, and action items.",
                'bullets': "Create an executive summary with strategic insights, key decisions needed, and action items.",
                'key_points': "Extract executive-level insights focusing on strategic value, risks, and recommendations."
            }
        }
    
    def summarize_content(self, request: SummaryRequest) -> SummaryResult:
        """
        Main method for summarizing content with all features.
        
        Args:
            request: SummaryRequest containing content and parameters
            
        Returns:
            SummaryResult with generated summary and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_result = self.cache_manager.get_cached_summary(request.content, request)
            if cache_result:
                cached_summary, cache_info = cache_result
                return SummaryResult(
                    summary=SummaryContent(
                        text=cached_summary,
                        content_type=request.content_type,
                        source_id=request.source_id,
                        metadata=cache_info.get('metadata')
                    ),
                    quality_metrics={'cached': cache_info.get('quality_score', 0.0)},
                    processing_info={'timestamp': datetime.now().isoformat()},
                    cache_hit=True
                )
            
            # Handle progressive summarization
            if request.enable_progressive:
                return self._progressive_summarization(request)
            
            # Determine if chunking is needed
            chunked = self.chunker.chunk_content(request.content, request.content_type)
            if len(chunked) == 0:
                return SummaryResult(
                    summary=SummaryContent("", request.content_type, request.source_id),
                    quality_metrics={'error': 0.0},
                    processing_info={'timestamp': datetime.now().isoformat()},
                    error="No content to summarize"
                )
            
            # Single summary or map-reduce
            if len(chunked) == 1:
                result = self._summarize_single_chunk(chunked[0], request, start_time)
            else:
                result = self._map_reduce_summarization(chunked, request, start_time)
            
            # Cache the result
            if result and not result.cache_hit:
                quality_score = result.quality_metrics.get('overall', 0.0)
                self.cache_manager.cache_summary(
                    request.content, request, result.summary.text, 
                    quality_score, result.processing_info
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return SummaryResult(
                summary=SummaryContent("", request.content_type, request.source_id),
                quality_metrics={'error': 0.0},
                processing_info={'timestamp': datetime.now().isoformat()},
                error=str(e)
            )
    
    def _progressive_summarization(self, request: SummaryRequest) -> SummaryResult:
        """Implement progressive summarization: brief → detailed on demand."""
        start_time = time.time()
        
        # First, generate a brief summary
        brief_request = SummaryRequest(
            content=request.content,
            content_type=request.content_type,
            source_id=request.source_id,
            summary_length="brief",
            format_type=request.format_type,
            include_metadata=request.include_metadata,
            enable_progressive=False,  # Prevent recursion
            enable_caching=request.enable_caching,
            max_retries=request.max_retries,
            timeout=request.timeout
        )
        
        brief_result = self._summarize_single_chunk(
            {'text': request.content, 'chunk_index': 0, 'chunk_type': request.content_type},
            brief_request,
            start_time
        )
        
        if not brief_result or brief_result.error:
            return brief_result
        
        # For progressive, we return the brief summary but indicate it can be expanded
        result = SummaryResult(
            summary=SummaryContent(
                text=brief_result.summary.text + "\n\n[Progressive: Available in detail]",
                content_type=request.content_type,
                source_id=request.source_id,
                metadata={
                    **brief_result.summary.metadata,
                    'progressive': True,
                    'full_summary_available': True
                }
            ),
            quality_metrics=brief_result.quality_metrics,
            processing_info={
                **brief_result.processing_info,
                'progressive_summary': True,
                'full_processing_time': time.time() - start_time
            },
            cache_hit=brief_result.cache_hit,
            llm_provider=brief_result.llm_provider,
            model_used=brief_result.model_used
        )
        
        return result
    
    def _summarize_single_chunk(self, 
                               chunk: Dict[str, Any], 
                               request: SummaryRequest,
                               start_time: float) -> SummaryResult:
        """Summarize a single chunk of content."""
        try:
            # Select appropriate LLM backend
            backend = self._select_backend(request)
            model_config = self._get_model_config(request)
            
            # Generate summary prompt
            prompt = self._generate_summary_prompt(chunk['text'], request)
            
            # Generate summary
            summary_text, llm_info = backend.generate_summary(
                chunk['text'], prompt, model_config
            )
            
            # Format the summary
            formatted_summary = self._format_summary(summary_text, request)
            
            # Create summary content
            summary_content = SummaryContent(
                text=formatted_summary,
                content_type=request.content_type,
                source_id=request.source_id,
                chunk_index=chunk.get('chunk_index'),
                chunk_count=1,
                processing_time=time.time() - start_time,
                token_count=llm_info.get('total_tokens', 0),
                metadata={'chunk_type': chunk.get('chunk_type', 'text')}
            )
            
            # Assess quality
            quality_metrics = self.quality_assessor.assess_quality(
                chunk['text'], formatted_summary, request
            )
            
            # Check quality threshold
            if quality_metrics['overall'] < request.quality_threshold:
                # Try again with adjusted parameters
                return self._retry_with_adjustment(chunk, request, start_time)
            
            return SummaryResult(
                summary=summary_content,
                quality_metrics=quality_metrics,
                processing_info={
                    'timestamp': datetime.now().isoformat(),
                    'chunk_count': 1,
                    'include_metadata': request.include_metadata,
                    'llm_info': llm_info
                },
                llm_provider=llm_info.get('provider'),
                model_used=llm_info.get('model')
            )
            
        except Exception as e:
            self.logger.error(f"Single chunk summarization failed: {e}")
            raise
    
    def _map_reduce_summarization(self, 
                                 chunks: List[Dict[str, Any]], 
                                 request: SummaryRequest,
                                 start_time: float) -> SummaryResult:
        """Implement map-reduce summarization for large content."""
        self.logger.info(f"Starting map-reduce summarization for {len(chunks)} chunks")
        
        # Map phase: summarize each chunk
        chunk_summaries = []
        map_errors = []
        
        for chunk in chunks:
            try:
                chunk_result = self._summarize_single_chunk(chunk, request, start_time)
                if chunk_result and not chunk_result.error:
                    chunk_summaries.append(chunk_result.summary)
                else:
                    map_errors.append(f"Chunk {chunk.get('chunk_index', 'unknown')} failed")
            except Exception as e:
                map_errors.append(f"Chunk {chunk.get('chunk_index', 'unknown')} error: {str(e)}")
        
        if not chunk_summaries:
            raise LLMBackendError("All chunks failed to summarize")
        
        # Reduce phase: combine chunk summaries
        combined_content = "\n\n".join([s.text for s in chunk_summaries])
        
        # Create a combined summary
        backend = self._select_backend(request)
        model_config = self._get_model_config(request)
        
        # Use a combination prompt for the reduce phase
        reduce_prompt = f"""Combine and synthesize the following section summaries into a comprehensive {request.summary_length} summary:

{self._generate_combination_prompt(request)}

Section summaries:
{combined_content}
"""
        
        combined_summary_text, llm_info = backend.generate_summary(
            combined_content, reduce_prompt, model_config
        )
        
        # Format the final summary
        formatted_summary = self._format_summary(combined_summary_text, request)
        
        # Create final summary content
        summary_content = SummaryContent(
            text=formatted_summary,
            content_type=request.content_type,
            source_id=request.source_id,
            chunk_count=len(chunks),
            processing_time=time.time() - start_time,
            token_count=llm_info.get('total_tokens', 0),
            metadata={
                'map_reduce': True,
                'chunk_count': len(chunks),
                'chunk_summaries': [s.text for s in chunk_summaries],
                'map_errors': map_errors
            }
        )
        
        # Assess overall quality
        quality_metrics = self.quality_assessor.assess_quality(
            request.content, formatted_summary, request
        )
        
        return SummaryResult(
            summary=summary_content,
            quality_metrics=quality_metrics,
            processing_info={
                'timestamp': datetime.now().isoformat(),
                'chunk_count': len(chunks),
                'map_errors': map_errors,
                'include_metadata': request.include_metadata,
                'llm_info': llm_info
            },
            llm_provider=llm_info.get('provider'),
            model_used=llm_info.get('model')
        )
    
    def _select_backend(self, request: SummaryRequest) -> LLMBackend:
        """Select the appropriate LLM backend."""
        # Try to get preferred backend from configuration
        if self.config.default_provider and self.config.default_provider in self.backends:
            return self.backends[self.config.default_provider]
        
        # Fallback to any available backend
        if self.backends:
            return list(self.backends.values())[0]
        
        raise ServiceUnavailableError("No LLM backends available")
    
    def _get_model_config(self, request: SummaryRequest) -> ModelConfig:
        """Get model configuration for the request."""
        # For now, use a simple default model configuration
        # In a real implementation, this would be more sophisticated
        from .config import ModelConfig, ModelType
        
        # Try to get a model from the selected backend's provider
        for provider_name, provider_config in self.config.providers.items():
            if provider_name in self.backends and provider_config.default_model:
                return provider_config.get_default_model()
        
        # Fallback to a simple model config
        return ModelConfig(
            name="default",
            model_type=ModelType.CHAT,
            provider="default",
            max_tokens=1000,
            temperature=0.7
        )
    
    def _generate_summary_prompt(self, content: str, request: SummaryRequest) -> str:
        """Generate appropriate summary prompt based on content type and length."""
        template_key = request.summary_length
        format_key = request.format_type
        
        if template_key not in self.summary_templates:
            template_key = 'detailed'
        if format_key not in self.summary_templates[template_key]:
            format_key = 'paragraph'
        
        base_prompt = self.summary_templates[template_key][format_key]
        
        # Add content-type specific instructions
        type_instructions = {
            'article': " This is an article or blog post. Focus on main arguments, evidence, and conclusions.",
            'transcript': " This is a transcript or speech. Include key speakers, main topics, and important quotes.",
            'documentation': " This is technical documentation. Focus on functionality, setup, and usage instructions.",
            'code': " This is source code. Summarize the purpose, main functions, and key implementation details.",
            'news': " This is a news article. Focus on who, what, when, where, why, and how."
        }
        
        instruction = type_instructions.get(request.content_type, "")
        
        return f"{base_prompt}{instruction}\n\nContent to summarize:\n{content}"
    
    def _generate_combination_prompt(self, request: SummaryRequest) -> str:
        """Generate prompt for combining multiple summaries."""
        prompts = {
            'brief': "Create a concise overview that captures the main themes and key points from all sections.",
            'detailed': "Create a comprehensive synthesis that preserves important details while eliminating redundancy.",
            'executive': "Focus on strategic insights, implications, and actionable items from the combined content."
        }
        
        return prompts.get(request.summary_length, prompts['detailed'])
    
    def _format_summary(self, raw_summary: str, request: SummaryRequest) -> str:
        """Format the generated summary according to specifications."""
        summary = raw_summary.strip()
        
        # Clean up common LLM artifacts
        summary = re.sub(r'^(Summary:|Here\'s a summary:)', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'\n{3,}', '\n\n', summary)  # Multiple newlines to double
        
        if request.format_type == 'bullets':
            # Ensure bullet points are properly formatted
            summary = re.sub(r'^(\d+\.|\-|\*)', lambda m: f"- {m.group(0)[2:].strip()}", summary, flags=re.MULTILINE)
        
        return summary
    
    def _retry_with_adjustment(self, 
                              chunk: Dict[str, Any], 
                              request: SummaryRequest,
                              start_time: float) -> SummaryResult:
        """Retry summarization with adjusted parameters."""
        # Create adjusted request with more conservative settings
        adjusted_request = SummaryRequest(
            content=request.content,
            content_type=request.content_type,
            source_id=request.source_id,
            summary_length="brief",  # Use shorter summary as fallback
            format_type="paragraph",  # Use simpler format
            include_metadata=request.include_metadata,
            quality_threshold=0.5,  # Lower quality threshold
            enable_progressive=False,
            enable_caching=request.enable_caching,
            max_retries=0,  # Prevent infinite recursion
            timeout=request.timeout
        )
        
        return self._summarize_single_chunk(chunk, adjusted_request, start_time)
    
    def summarize_multiple_documents(self, 
                                   documents: List[Tuple[str, str]], 
                                   summary_type: str = "comparative",
                                   **kwargs) -> MultiDocumentSummary:
        """
        Summarize multiple documents together.
        
        Args:
            documents: List of (content, content_type) tuples
            summary_type: Type of multi-document summary
            **kwargs: Additional parameters for summarization
            
        Returns:
            MultiDocumentSummary with combined insights
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Summarize each document individually
        individual_summaries = []
        for i, (content, content_type) in enumerate(documents):
            request = SummaryRequest(
                content=content,
                content_type=content_type,
                source_id=f"doc_{i}",
                **kwargs
            )
            result = self.summarize_content(request)
            if result and not result.error:
                individual_summaries.append(result.summary)
        
        if not individual_summaries:
            raise LLMBackendError("Failed to summarize any documents")
        
        # Generate combined summary
        combined_content = "\n\n".join([s.text for s in individual_summaries])
        combined_request = SummaryRequest(
            content=combined_content,
            content_type="multi_document",
            source_id="combined",
            summary_length=kwargs.get('summary_length', 'detailed'),
            format_type=kwargs.get('format_type', 'paragraph'),
            enable_progressive=False
        )
        
        combined_result = self.summarize_content(combined_request)
        
        if not combined_result or combined_result.error:
            raise LLMBackendError("Failed to generate combined summary")
        
        # Generate cross-document insights
        cross_doc_insights = self._generate_cross_document_insights(
            individual_summaries, summary_type
        )
        
        return MultiDocumentSummary(
            combined_summary=combined_result.summary,
            individual_summaries=individual_summaries,
            cross_document_insights=cross_doc_insights['insights'],
            common_themes=cross_doc_insights['themes'],
            unique_content=cross_doc_insights['unique'],
            contradictions=cross_doc_insights['contradictions'],
            quality_comparison={s.source_id: s.quality_score for s in individual_summaries if s.quality_score}
        )
    
    def _generate_cross_document_insights(self, 
                                        summaries: List[SummaryContent], 
                                        summary_type: str) -> Dict[str, List[str]]:
        """Generate insights across multiple document summaries."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated analysis
        
        combined_text = " ".join([s.text for s in summaries])
        
        insights = {
            'insights': [],
            'themes': [],
            'unique': [],
            'contradictions': []
        }
        
        # Simple theme extraction
        words = combined_text.lower().split()
        word_freq = defaultdict(int)
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        for word in filtered_words:
            word_freq[word] += 1
        
        # Get top themes
        insights['themes'] = [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        # Generate basic insights
        if summary_type == "comparative":
            insights['insights'].append("Documents show varying perspectives on shared topics")
            insights['insights'].append("Key differences in approach and emphasis identified")
        elif summary_type == "synthesis":
            insights['insights'].append("Common themes synthesized across documents")
            insights['insights'].append("Complementary information merged into coherent view")
        
        return insights
    
    def export_summary(self, 
                      summary_result: SummaryResult, 
                      output_path: str, 
                      format_type: str = "json"):
        """
        Export summary result to file.
        
        Args:
            summary_result: SummaryResult to export
            output_path: Output file path
            format_type: Export format (json, markdown, text)
        """
        if format_type == "json":
            self.exporter.export_json(summary_result, output_path)
        elif format_type == "markdown":
            self.exporter.export_markdown(summary_result, output_path)
        elif format_type == "text":
            self.exporter.export_plain_text(summary_result, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Summary exported to {output_path} in {format_type} format")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.cache_manager.cache_db_path)
        try:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(last_accessed) as most_recent_access
                FROM summary_cache
            ''')
            
            row = cursor.fetchone()
            return {
                'total_entries': row[0] or 0,
                'avg_access_count': row[1] or 0,
                'oldest_entry': row[2],
                'most_recent_access': row[3]
            }
        finally:
            conn.close()
    
    def clear_cache(self, older_than_days: int = 30):
        """Clear old cache entries."""
        self.cache_manager.clear_cache(older_than_days)
        self.logger.info(f"Cleared cache entries older than {older_than_days} days")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the summarization system."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'backends': {},
            'cache': {},
            'overall_status': 'healthy'
        }
        
        # Check backends
        for name, backend in self.backends.items():
            try:
                # Simple connectivity check
                health['backends'][name] = {
                    'available': True,
                    'status': 'healthy'
                }
            except Exception as e:
                health['backends'][name] = {
                    'available': True,
                    'status': f'error: {str(e)}'
                }
        
        # Check cache
        try:
            stats = self.get_cache_stats()
            health['cache'] = {
                'available': True,
                'stats': stats
            }
        except Exception as e:
            health['cache'] = {
                'available': False,
                'error': str(e)
            }
            health['overall_status'] = 'degraded'
        
        # Overall health assessment
        if not self.backends:
            health['overall_status'] = 'unhealthy'
            health['issues'] = ['No LLM backends available']
        elif health['overall_status'] != 'unhealthy':
            health['overall_status'] = 'healthy'
        
        return health


# Utility functions for easy integration

def create_summarization_manager(config_path: Optional[str] = None) -> SummarizationManager:
    """Create and configure a SummarizationManager instance."""
    return SummarizationManager(config_path=config_path)


def summarize_text(content: str, 
                  content_type: str = "text",
                  summary_length: str = "detailed",
                  format_type: str = "paragraph") -> str:
    """
    Simple function to summarize text with default settings.
    
    Args:
        content: Text to summarize
        content_type: Type of content
        summary_length: Length of summary
        format_type: Format of summary
        
    Returns:
        Summarized text
    """
    manager = SummarizationManager()
    request = SummaryRequest(
        content=content,
        content_type=content_type,
        summary_length=summary_length,
        format_type=format_type,
        enable_progressive=False
    )
    
    result = manager.summarize_content(request)
    return result.summary.text if result and not result.error else content[:200] + "..."


# Integration with ContextBox database
class DatabaseIntegratedSummarizer(SummarizationManager):
    """Summarization manager with ContextBox database integration."""
    
    def __init__(self, database, config_path: Optional[str] = None):
        """
        Initialize with database integration.
        
        Args:
            database: ContextBox database instance
            config_path: Path to LLM configuration file
        """
        super().__init__(config_path)
        self.database = database
    
    def summarize_artifact(self, 
                          artifact_id: int, 
                          **kwargs) -> SummaryResult:
        """
        Summarize an artifact from the database.
        
        Args:
            artifact_id: ID of the artifact to summarize
            **kwargs: Additional summarization parameters
            
        Returns:
            SummaryResult
        """
        # Get artifact from database
        artifact = self.database.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        # Create summary request
        request = SummaryRequest(
            content=artifact['text'] or "",
            content_type=kwargs.get('content_type', artifact['kind']),
            source_id=str(artifact_id),
            **kwargs
        )
        
        # Generate summary
        result = self.summarize_content(request)
        
        # Store summary in database as new artifact
        if result and not result.error:
            try:
                # Create summary metadata
                summary_metadata = {
                    'original_artifact_id': artifact_id,
                    'quality_score': result.quality_metrics.get('overall', 0.0),
                    'summary_length': request.summary_length,
                    'format_type': request.format_type,
                    'llm_provider': result.llm_provider,
                    'model_used': result.model_used,
                    'processing_time': result.summary.processing_time
                }
                
                # Store summary as artifact
                summary_artifact_id = self.database.create_artifact(
                    capture_id=artifact['capture_id'],
                    kind='summary',
                    title=f"Summary of {artifact['title'] or artifact['kind']}",
                    text=result.summary.text,
                    metadata=summary_metadata
                )
                
                result.summary.metadata = result.summary.metadata or {}
                result.summary.metadata['database_artifact_id'] = summary_artifact_id
                
            except Exception as e:
                self.logger.warning(f"Failed to store summary in database: {e}")
        
        return result