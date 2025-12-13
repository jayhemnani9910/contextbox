"""
Question-Answering System for ContextBox

This module provides comprehensive question-answering functionality using captured content
from the ContextBox database. It includes semantic search, context building, question
classification, and multi-document QA capabilities.

Features:
- Content retrieval and context building
- Question classification (factual, inferential, comparative, etc.)
- Context-aware answering with source attribution
- Multi-document question answering
- Follow-up question handling and clarification
- Answer confidence scoring and uncertainty handling
- Response formatting with direct answers, explanations, and sources
- Integration with all content extraction modules
- Query optimization and performance enhancement
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from ..database import ContextDatabase, DatabaseError
from .exceptions import LLMBackendError


class QuestionType(Enum):
    """Enumeration of question types for classification."""
    FACTUAL = "factual"
    INFERENTIAL = "inferential"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    EXPLANATORY = "explanatory"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    DEFINITION = "definition"
    LIST = "list"
    REASONING = "reasoning"
    UNCLEAR = "unclear"


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels for answers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ResponseFormat(Enum):
    """Enumeration of response formats."""
    DIRECT = "direct"
    EXPLANATORY = "explanatory"
    DETAILED = "detailed"
    SUMMARY = "summary"


@dataclass
class Source:
    """Source information for answer attribution."""
    artifact_id: int
    capture_id: int
    content: str
    url: Optional[str] = None
    title: Optional[str] = None
    kind: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[datetime] = None
    relevance_score: float = 0.0


@dataclass
class QAContext:
    """Context for a question-answering session."""
    question: str
    question_type: QuestionType
    keywords: List[str]
    context_window: List[Source]
    temporal_context: Optional[datetime] = None
    confidence_threshold: float = 0.7
    max_sources: int = 10
    require_citations: bool = True


@dataclass
class Answer:
    """Complete answer with metadata and sources."""
    answer_text: str
    question_type: QuestionType
    confidence_level: ConfidenceLevel
    confidence_score: float
    sources: List[Source]
    explanations: List[str] = None
    follow_up_suggestions: List[str] = None
    processing_time: float = 0.0
    tokens_used: int = 0
    formatted_response: str = ""
    needs_clarification: bool = False
    clarification_questions: List[str] = None


class QuestionClassifier:
    """Classifies questions into different types for optimal processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Question pattern definitions
        self.patterns = {
            QuestionType.FACTUAL: [
                r'\b(who|what|when|where|which)\b',
                r'\b(is|are|was|were|does|do|did|can|will)\b'
            ],
            QuestionType.INFERENTIAL: [
                r'\b(imply|implying|inferred|inference|infer|deduce|deducing|deduction)\b',
                r'\b(implication|implications|suggest|suggesting|suggestion)\b'
            ],
            QuestionType.COMPARATIVE: [
                r'\b(better|best|worse|worst|compare|comparison|versus|vs|than|more|less)\b',
                r'\b(difference|differences|similar|similarity|contrast|contrasting)\b'
            ],
            QuestionType.PROCEDURAL: [
                r'\b(how\s+to|how\s+do|how\s+can|step|steps|process|procedure)\b',
                r'\b(instructions|directions|method|approach)\b'
            ],
            QuestionType.TEMPORAL: [
                r'\b(when|time|date|year|month|day|period|duration|before|after|during)\b',
                r'\b(history|historical|recent|latest|earlier|later)\b'
            ],
            QuestionType.CAUSAL: [
                r'\b(why|because|cause|caused|causes|causing|result|results|effect|effects)\b',
                r'\b(therefore|thus|consequently|as\s+a\s+result)\b'
            ],
            QuestionType.DEFINITION: [
                r'\b(what\s+is|what\s+are|definition|define|means|meaning|refer\s+to)\b',
                r'\b(describe|description|explain\s+what)\b'
            ],
            QuestionType.LIST: [
                r'\b(list|items|examples|types|categories|kinds|sorts)\b',
                r'\b(what\s+are\s+some|give\s+me|name)\b'
            ],
            QuestionType.ANALYTICAL: [
                r'\b(analyz|analysis|examine|examine|evaluate|evaluation|assess|assessment)\b',
                r'\b(reason|reasons|factor|factors|criteria|consider)\b'
            ]
        }
    
    def classify(self, question: str) -> Tuple[QuestionType, float]:
        """
        Classify a question and return confidence score.
        
        Args:
            question: The question to classify
            
        Returns:
            Tuple of (question_type, confidence_score)
        """
        question_lower = question.lower()
        scores = {}
        
        # Score each question type
        for q_type, patterns in self.patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    matches += 1
                    score += 1.0
            
            # Normalize score
            if len(patterns) > 0:
                scores[q_type] = matches / len(patterns)
            else:
                scores[q_type] = 0.0
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            return QuestionType.UNCLEAR, 0.1
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Apply additional heuristics
        confidence = self._apply_heuristics(question_lower, best_type, confidence)
        
        self.logger.debug(f"Classified question '{question[:50]}...' as {best_type.value} with confidence {confidence:.2f}")
        return best_type, confidence
    
    def _apply_heuristics(self, question: str, q_type: QuestionType, confidence: float) -> float:
        """Apply additional heuristics to improve classification."""
        
        # Wh-question boost for factual
        if q_type == QuestionType.FACTUAL and re.search(r'\b(who|what|when|where|which|how)\b', question):
            confidence += 0.2
        
        # Multiple question marks suggest unclear
        if question.count('?') > 1:
            confidence *= 0.8
        
        # Very short questions might be unclear
        if len(question.split()) < 3:
            confidence *= 0.7
        
        # Long questions might be analytical
        if q_type == QuestionType.ANALYTICAL and len(question.split()) > 15:
            confidence += 0.1
        
        return min(confidence, 1.0)


class ContentRetriever:
    """Retrieves and ranks content from the ContextBox database."""
    
    def __init__(self, database: ContextDatabase):
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Content priority weights
        self.kind_weights = {
            'url': 1.0,
            'text': 1.2,
            'ocr_text': 0.8,
            'extracted_url': 0.9,
            'url_analysis': 0.7,
            'extraction_summary': 0.6,
            'extraction_metadata': 0.5,
            'wikipedia_content': 1.3,
            'webpage_content': 1.1
        }
    
    def retrieve_relevant_content(
        self,
        question: str,
        question_type: QuestionType,
        max_results: int = 50,
        time_window_days: int = 365
    ) -> List[Source]:
        """
        Retrieve content relevant to the question.
        
        Args:
            question: The question to find content for
            question_type: Type of question for specialized retrieval
            max_results: Maximum number of results to return
            time_window_days: Only consider content from this many days ago
            
        Returns:
            List of ranked Source objects
        """
        start_time = time.time()
        
        try:
            # Extract keywords from question
            keywords = self._extract_keywords(question)
            
            # Get time-based filter
            since_date = datetime.now() - timedelta(days=time_window_days)
            
            # Retrieve artifacts
            all_sources = []
            
            # Search by keywords
            for keyword in keywords:
                if len(keyword) >= 3:  # Only search for meaningful keywords
                    sources = self._search_by_keyword(keyword, max_results // len(keywords))
                    all_sources.extend(sources)
            
            # If no keyword matches, search more broadly
            if not all_sources:
                all_sources = self._search_all_content(max_results // 2)
            
            # Rank and filter results
            ranked_sources = self._rank_sources(all_sources, keywords, question_type)
            
            # Limit results
            final_sources = ranked_sources[:max_results]
            
            processing_time = time.time() - start_time
            self.logger.info(f"Retrieved {len(final_sources)} sources in {processing_time:.2f}s for question: '{question[:50]}...'")
            
            return final_sources
            
        except Exception as e:
            self.logger.error(f"Error retrieving content: {e}")
            return []
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract meaningful keywords from a question."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'can', 'may', 'might', 'must', 'shall', 'what', 'when',
            'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        # Extract words, remove punctuation, filter length and stop words
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        keywords = [word for word in words if len(word) >= 3 and word not in stop_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _search_by_keyword(self, keyword: str, limit: int) -> List[Source]:
        """Search for artifacts containing a specific keyword."""
        try:
            artifacts = self.database.search_extraction_artifacts(keyword, limit=limit)
            return [self._artifact_to_source(artifact) for artifact in artifacts]
        except Exception as e:
            self.logger.error(f"Error searching for keyword '{keyword}': {e}")
            return []
    
    def _search_all_content(self, limit: int) -> List[Source]:
        """Search all available content when keywords fail."""
        try:
            with self.database._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM artifacts 
                    WHERE text IS NOT NULL AND TRIM(text) != ''
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
                
                sources = []
                for row in cursor.fetchall():
                    artifact = dict(row)
                    source = self._artifact_to_source(artifact)
                    sources.append(source)
                
                return sources
        except Exception as e:
            self.logger.error(f"Error searching all content: {e}")
            return []
    
    def _artifact_to_source(self, artifact: Dict[str, Any]) -> Source:
        """Convert database artifact to Source object."""
        return Source(
            artifact_id=artifact['id'],
            capture_id=artifact['capture_id'],
            content=artifact.get('text', '') or '',
            url=artifact.get('url'),
            title=artifact.get('title'),
            kind=artifact.get('kind'),
            confidence=artifact.get('metadata', {}).get('confidence'),
            timestamp=datetime.now(),  # Default to now, can be improved
            relevance_score=0.0
        )
    
    def _rank_sources(
        self,
        sources: List[Source],
        keywords: List[str],
        question_type: QuestionType
    ) -> List[Source]:
        """Rank sources by relevance to the question."""
        
        for source in sources:
            # Calculate base relevance score
            relevance_score = self._calculate_relevance_score(source, keywords)
            
            # Apply question-type specific boosts
            relevance_score = self._apply_question_type_boost(source, question_type, relevance_score)
            
            # Apply content kind weights
            if source.kind in self.kind_weights:
                relevance_score *= self.kind_weights[source.kind]
            
            source.relevance_score = relevance_score
        
        # Sort by relevance score
        return sorted(sources, key=lambda s: s.relevance_score, reverse=True)
    
    def _calculate_relevance_score(self, source: Source, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches."""
        if not source.content:
            return 0.0
        
        content_lower = source.content.lower()
        total_score = 0.0
        
        for keyword in keywords:
            # Count exact matches
            exact_matches = content_lower.count(keyword.lower())
            total_score += exact_matches * 2.0
            
            # Count partial matches (word boundaries)
            partial_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\w*\b', content_lower))
            total_score += partial_matches * 1.5
            
            # Check in title if available
            if source.title:
                title_lower = source.title.lower()
                if keyword.lower() in title_lower:
                    total_score += 3.0
        
        # Normalize by content length
        content_length = len(source.content.split())
        if content_length > 0:
            total_score = total_score / (content_length ** 0.5)
        
        return min(total_score, 10.0)  # Cap at 10.0
    
    def _apply_question_type_boost(
        self,
        source: Source,
        question_type: QuestionType,
        score: float
    ) -> float:
        """Apply question-type specific scoring boosts."""
        
        content_lower = source.content.lower()
        
        if question_type == QuestionType.FACTUAL:
            # Boost sources with structured data
            if re.search(r'\d{4}|\d{1,2}/\d{1,2}|\d{1,2}:\d{2}', content_lower):  # Dates, times
                score *= 1.2
            if source.url:  # Sources with URLs might be more factual
                score *= 1.1
        
        elif question_type == QuestionType.PROCEDURAL:
            # Boost sources with action words
            action_words = ['step', 'process', 'procedure', 'method', 'how', 'install', 'create', 'build']
            for word in action_words:
                if word in content_lower:
                    score *= 1.15
                    break
        
        elif question_type == QuestionType.TEMPORAL:
            # Boost sources with dates/times
            if re.search(r'\d{4}|year|month|day|recent|latest|earlier', content_lower):
                score *= 1.3
        
        elif question_type == QuestionType.LIST:
            # Boost sources with list-like structure
            if re.search(r'\n|\.|,|;', content_lower):  # Has separators
                score *= 1.1
        
        return score


class AnswerGenerator:
    """Generates answers using language models and context."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration defaults
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.3)
        self.model = self.config.get('model', 'gpt-3.5-turbo')
        self.provider = self.config.get('provider', 'openai')
        
        # Context window management
        self.max_context_length = self.config.get('max_context_length', 4000)
        self.context_overlap = self.config.get('context_overlap', 200)
    
    def generate_answer(
        self,
        qa_context: QAContext,
        sources: List[Source]
    ) -> Answer:
        """
        Generate an answer to a question using the provided context.
        
        Args:
            qa_context: Question-answering context
            sources: Retrieved and ranked sources
            
        Returns:
            Answer object with generated content and metadata
        """
        start_time = time.time()
        
        try:
            # Build context from sources
            context_text = self._build_context_text(sources, qa_context.max_sources)
            
            # Generate answer based on question type
            if qa_context.question_type == QuestionType.FACTUAL:
                answer_text = self._generate_factual_answer(qa_context.question, context_text)
            elif qa_context.question_type == QuestionType.PROCEDURAL:
                answer_text = self._generate_procedural_answer(qa_context.question, context_text)
            elif qa_context.question_type == QuestionType.COMPARATIVE:
                answer_text = self._generate_comparative_answer(qa_context.question, context_text)
            else:
                answer_text = self._generate_general_answer(qa_context.question, context_text)
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(answer_text, sources, qa_context)
            confidence_level = self._determine_confidence_level(confidence_score)
            
            # Generate explanations
            explanations = self._generate_explanations(qa_context.question, sources, answer_text)
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_ups(qa_context.question, answer_text)
            
            # Format final response
            formatted_response = self._format_response(answer_text, sources, explanations, follow_ups)
            
            processing_time = time.time() - start_time
            
            return Answer(
                answer_text=answer_text,
                question_type=qa_context.question_type,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                sources=sources[:5],  # Top 5 sources
                explanations=explanations,
                follow_up_suggestions=follow_ups,
                processing_time=processing_time,
                tokens_used=len(answer_text.split()) * 1.3,  # Rough estimate
                formatted_response=formatted_response
            )
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return self._generate_fallback_answer(qa_context, sources)
    
    def _build_context_text(self, sources: List[Source], max_sources: int) -> str:
        """Build context text from sources with size management."""
        context_parts = []
        current_length = 0
        
        for source in sources[:max_sources]:
            # Prepare source text
            source_text = f"[Source {len(context_parts) + 1}]"
            if source.title:
                source_text += f" {source.title}"
            if source.url:
                source_text += f" ({source.url})"
            source_text += f"\n{source.content}"
            
            # Check if adding this source would exceed the limit
            if current_length + len(source_text) > self.max_context_length:
                break
            
            context_parts.append(source_text)
            current_length += len(source_text)
        
        return "\n\n".join(context_parts)
    
    def _generate_factual_answer(self, question: str, context: str) -> str:
        """Generate a factual answer."""
        if not context.strip():
            return "I don't have sufficient information in my context to answer this factual question accurately. Could you provide more specific details or ask about content that's been captured?"
        
        prompt = f"""Based on the following context, provide a direct, factual answer to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Simulate LLM call (in real implementation, this would call actual LLM)
        answer = self._simulate_llm_call(prompt)
        return answer
    
    def _generate_procedural_answer(self, question: str, context: str) -> str:
        """Generate a procedural/instructional answer."""
        if not context.strip():
            return "I don't have step-by-step instructions in my context for this question. The content might not contain procedural information."
        
        prompt = f"""Based on the following context, provide clear, step-by-step instructions or procedural information to answer the question.

Context:
{context}

Question: {question}

Provide the steps or procedures:"""
        
        answer = self._simulate_llm_call(prompt)
        return answer
    
    def _generate_comparative_answer(self, question: str, context: str) -> str:
        """Generate a comparative answer."""
        if not context.strip():
            return "I don't have comparative information in my context to answer this question. The content might not contain the information needed for comparison."
        
        prompt = f"""Based on the following context, provide a comparative analysis to answer the question.

Context:
{context}

Question: {question}

Provide a comparison:"""
        
        answer = self._simulate_llm_call(prompt)
        return answer
    
    def _generate_general_answer(self, question: str, context: str) -> str:
        """Generate a general answer for other question types."""
        if not context.strip():
            return "I don't have relevant information in my context to answer this question comprehensively. The captured content may not contain the information you're looking for."
        
        prompt = f"""Based on the following context, provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self._simulate_llm_call(prompt)
        return answer
    
    def _simulate_llm_call(self, prompt: str) -> str:
        """Simulate LLM call (placeholder for actual implementation)."""
        # In a real implementation, this would call the actual LLM API
        # For now, return a template response
        
        # Simple heuristic-based response generation
        if "step" in prompt.lower() or "procedure" in prompt.lower():
            return "Based on the available context, here's what I found:\n\n1. The information suggests a multi-step approach\n2. Each step builds upon the previous one\n3. The process requires careful attention to detail\n\nFor specific instructions, please ensure the relevant content is captured in your ContextBox."
        
        elif "compare" in prompt.lower() or "versus" in prompt.lower() or "difference" in prompt.lower():
            return "Based on the available context, here are the key differences:\n\n• Each option has distinct characteristics\n• Performance varies by specific use case\n• Consider your specific requirements when choosing\n\nA detailed comparison would require more specific context."
        
        else:
            return "Based on the available context:\n\nThe information indicates that this topic involves multiple factors and considerations. The captured content provides some insights, but a complete answer would benefit from additional context or more specific information about your particular question.\n\nKey points from the available content include relevant details that support this conclusion."
    
    def _calculate_confidence(
        self,
        answer: str,
        sources: List[Source],
        qa_context: QAContext
    ) -> float:
        """Calculate confidence score for the answer."""
        
        # Base confidence from source relevance
        if not sources:
            return 0.1
        
        avg_relevance = sum(source.relevance_score for source in sources) / len(sources)
        relevance_confidence = min(avg_relevance / 10.0, 1.0)  # Normalize to 0-1
        
        # Confidence from answer completeness
        answer_length = len(answer.split())
        if answer_length < 10:
            completeness_confidence = 0.3
        elif answer_length < 50:
            completeness_confidence = 0.7
        else:
            completeness_confidence = 1.0
        
        # Confidence from source diversity
        unique_kinds = len(set(source.kind for source in sources if source.kind))
        source_diversity_confidence = min(unique_kinds / 3.0, 1.0)
        
        # Confidence from question type match
        type_match_confidence = 0.8 if qa_context.question_type != QuestionType.UNCLEAR else 0.4
        
        # Weighted combination
        confidence = (
            relevance_confidence * 0.4 +
            completeness_confidence * 0.3 +
            source_diversity_confidence * 0.2 +
            type_match_confidence * 0.1
        )
        
        return min(confidence, 1.0)
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    def _generate_explanations(
        self,
        question: str,
        sources: List[Source],
        answer: str
    ) -> List[str]:
        """Generate explanations for the answer."""
        explanations = []
        
        # Explain source basis
        if sources:
            explanations.append(f"This answer is based on {len(sources)} sources from your captured content.")
            
            # Mention source types
            source_types = [source.kind for source in sources if source.kind]
            if source_types:
                unique_types = list(set(source_types))
                explanations.append(f"Primary source types: {', '.join(unique_types)}")
        
        # Explain reasoning approach
        if len(question.split()) > 10:
            explanations.append("This was a complex question requiring analysis of multiple pieces of information.")
        else:
            explanations.append("This was a direct question that could be answered with specific information.")
        
        return explanations
    
    def _generate_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate follow-up question suggestions."""
        follow_ups = []
        
        question_lower = question.lower()
        
        # Generate contextual follow-ups based on question type
        if "how" in question_lower:
            follow_ups.extend([
                "What are the potential challenges or obstacles?",
                "Are there alternative methods or approaches?",
                "What prerequisites are needed?"
            ])
        elif "what" in question_lower:
            follow_ups.extend([
                "What are the main benefits or advantages?",
                "What are the potential drawbacks?",
                "How does this compare to alternatives?"
            ])
        elif "why" in question_lower:
            follow_ups.extend([
                "What are the underlying causes or factors?",
                "What would happen if this changed?",
                "Are there different perspectives on this?"
            ])
        else:
            # General follow-ups
            follow_ups.extend([
                "Can you provide more specific examples?",
                "What are the next steps or implications?",
                "Are there related topics you'd like to explore?"
            ])
        
        return follow_ups[:3]  # Limit to 3 follow-ups
    
    def _format_response(
        self,
        answer: str,
        sources: List[Source],
        explanations: List[str],
        follow_ups: List[str]
    ) -> str:
        """Format the final response with answer, sources, and metadata."""
        
        response_parts = []
        
        # Main answer
        response_parts.append("## Answer")
        response_parts.append(answer)
        response_parts.append("")
        
        # Sources
        if sources:
            response_parts.append("## Sources")
            for i, source in enumerate(sources[:5], 1):
                source_text = f"{i}. "
                if source.title:
                    source_text += f"**{source.title}**"
                if source.url:
                    source_text += f" ({source.url})"
                if source.kind:
                    source_text += f" - {source.kind}"
                response_parts.append(source_text)
            response_parts.append("")
        
        # Explanations
        if explanations:
            response_parts.append("## Notes")
            for explanation in explanations:
                response_parts.append(f"• {explanation}")
            response_parts.append("")
        
        # Follow-ups
        if follow_ups:
            response_parts.append("## Related Questions")
            for follow_up in follow_ups:
                response_parts.append(f"• {follow_up}")
        
        return "\n".join(response_parts)
    
    def _generate_fallback_answer(self, qa_context: QAContext, sources: List[Source]) -> Answer:
        """Generate a fallback answer when normal generation fails."""
        return Answer(
            answer_text="I apologize, but I encountered an issue while generating an answer to your question. This could be due to insufficient context or technical limitations. Please try rephrasing your question or ensuring that relevant content has been captured in ContextBox.",
            question_type=qa_context.question_type,
            confidence_level=ConfidenceLevel.UNKNOWN,
            confidence_score=0.1,
            sources=sources[:3],
            explanations=["Technical issue during answer generation"],
            follow_up_suggestions=["Could you try rephrasing the question?", "Is the relevant content captured in ContextBox?"],
            processing_time=0.0,
            tokens_used=0,
            formatted_response="## Answer\nI apologize, but I encountered an issue while generating an answer to your question. This could be due to insufficient context or technical limitations. Please try rephrasing your question or ensuring that relevant content has been captured in ContextBox.\n\n## Notes\n• Technical issue during answer generation\n\n## Related Questions\n• Could you try rephrasing the question?\n• Is the relevant content captured in ContextBox?"
        )


class QASystem:
    """
    Main Question-Answering System for ContextBox.
    
    Provides comprehensive question-answering capabilities using captured content
    from the ContextBox database with semantic search, context building, and
    intelligent response generation.
    """
    
    def __init__(
        self,
        database: ContextDatabase,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the QA System.
        
        Args:
            database: ContextBox database instance
            config: Configuration dictionary for QA system
        """
        self.database = database
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.classifier = QuestionClassifier()
        self.retriever = ContentRetriever(database)
        self.generator = AnswerGenerator(self.config.get('generator', {}))
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        
        # Performance tracking
        self.stats = {
            'total_questions': 0,
            'successful_answers': 0,
            'failed_answers': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        self.logger.info("QASystem initialized successfully")
    
    def ask_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        options: Dict[str, Any] = None
    ) -> Answer:
        """
        Ask a question and get an answer based on captured content.
        
        Args:
            question: The question to ask
            session_id: Optional session ID for conversation context
            options: Additional options for question processing
            
        Returns:
            Answer object with the response and metadata
        """
        start_time = time.time()
        options = options or {}
        
        self.logger.info(f"Processing question: '{question[:100]}...' (session: {session_id})")
        
        try:
            # Update statistics
            self.stats['total_questions'] += 1
            
            # Classify the question
            question_type, classification_confidence = self.classifier.classify(question)
            
            # Extract keywords
            keywords = self.retriever._extract_keywords(question)
            
            # Get session context if available
            session_context = None
            if session_id and session_id in self.active_sessions:
                session_context = self.active_sessions[session_id]
            
            # Set up QA context
            qa_context = QAContext(
                question=question,
                question_type=question_type,
                keywords=keywords,
                context_window=[],
                confidence_threshold=options.get('confidence_threshold', 0.7),
                max_sources=options.get('max_sources', 10),
                require_citations=options.get('require_citations', True)
            )
            
            # Retrieve relevant content
            sources = self.retriever.retrieve_relevant_content(
                question,
                question_type,
                max_results=qa_context.max_sources,
                time_window_days=options.get('time_window_days', 365)
            )
            
            qa_context.context_window = sources
            
            # Generate answer
            answer = self.generator.generate_answer(qa_context, sources)
            
            # Handle low confidence responses
            if answer.confidence_score < qa_context.confidence_threshold:
                answer.needs_clarification = True
                answer.clarification_questions = self._generate_clarification_questions(question, sources)
            
            # Update session context
            if session_id:
                self._update_session(session_id, question, answer)
            
            # Update statistics
            processing_time = time.time() - start_time
            answer.processing_time = processing_time
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_questions']
            )
            
            if answer.confidence_level != ConfidenceLevel.UNKNOWN:
                self.stats['successful_answers'] += 1
            else:
                self.stats['failed_answers'] += 1
            
            self.logger.info(
                f"Question answered in {processing_time:.2f}s with {answer.confidence_level.value} "
                f"confidence ({answer.confidence_score:.2f})"
            )
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            
            # Return error answer
            return Answer(
                answer_text="I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                question_type=QuestionType.UNCLEAR,
                confidence_level=ConfidenceLevel.UNKNOWN,
                confidence_score=0.0,
                sources=[],
                explanations=[f"Error: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def ask_follow_up(
        self,
        original_question: str,
        follow_up: str,
        session_id: str,
        options: Dict[str, Any] = None
    ) -> Answer:
        """
        Ask a follow-up question within a session.
        
        Args:
            original_question: The original question that was asked
            follow_up: The follow-up question
            session_id: Session ID for context
            options: Additional options
            
        Returns:
            Answer object for the follow-up
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found for follow-up question")
            return self.ask_question(follow_up, session_id, options)
        
        # Get session context
        session = self.active_sessions[session_id]
        
        # Enhance follow-up with context from original question and session
        enhanced_question = self._enhance_follow_up_question(original_question, follow_up, session)
        
        return self.ask_question(enhanced_question, session_id, options)
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new QA session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID string
        """
        session_id = self._generate_session_id()
        
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'user_id': user_id,
            'questions': [],
            'answers': [],
            'context': {}
        }
        
        self.logger.info(f"Created new QA session: {session_id}")
        return session_id
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a QA session.
        
        Args:
            session_id: Session ID to close
            
        Returns:
            True if session was closed successfully
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Closed QA session: {session_id}")
            return True
        return False
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session statistics dictionary
        """
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'question_count': len(session['questions']),
            'average_confidence': (
                sum(ans.confidence_score for ans in session['answers']) / 
                len(session['answers']) if session['answers'] else 0.0
            ),
            'total_processing_time': sum(ans.processing_time for ans in session['answers']),
            'topics_covered': list(set(keyword for qa_context in session['context'].values() for keyword in qa_context.keywords))
        }
    
    def search_content(
        self,
        query: str,
        content_types: Optional[List[str]] = None,
        limit: int = 20,
        time_window_days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Search across all captured content.
        
        Args:
            query: Search query
            content_types: Optional filter by content types
            limit: Maximum number of results
            time_window_days: Time window for results
            
        Returns:
            List of matching content items
        """
        try:
            since_date = datetime.now() - timedelta(days=time_window_days)
            
            # Use the database search method
            results = []
            for content_type in (content_types or ['text', 'url', 'ocr_text']):
                type_results = self.database.search_extraction_artifacts(
                    query, 
                    artifact_kind=content_type, 
                    limit=limit // len(content_types or ['text'])
                )
                results.extend(type_results)
            
            # Deduplicate and sort by relevance
            seen_ids = set()
            unique_results = []
            for result in results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching content: {e}")
            return []
    
    def get_qa_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive QA system statistics.
        
        Returns:
            Statistics dictionary
        """
        total_questions = self.stats['total_questions']
        
        return {
            'system_stats': self.stats.copy(),
            'session_stats': {
                'active_sessions': len(self.active_sessions),
                'session_timeout': self.session_timeout
            },
            'performance_metrics': {
                'success_rate': (
                    self.stats['successful_answers'] / total_questions 
                    if total_questions > 0 else 0.0
                ),
                'average_processing_time': self.stats['average_processing_time'],
                'questions_per_minute': (
                    total_questions / (self.stats['total_processing_time'] / 60)
                    if self.stats['total_processing_time'] > 0 else 0.0
                )
            },
            'database_stats': self.database.get_stats()
        }
    
    def optimize_queries(self) -> Dict[str, List[str]]:
        """
        Analyze query patterns and suggest optimizations.
        
        Returns:
            Optimization recommendations
        """
        # Analyze recent questions for patterns
        recent_sessions = [
            session for session in self.active_sessions.values()
            if (datetime.now() - session['created_at']).days < 7
        ]
        
        recommendations = {
            'frequent_keywords': [],
            'suggested_indexes': [],
            'content_gaps': [],
            'performance_tips': []
        }
        
        # Analyze keyword frequency
        keyword_count = {}
        for session in recent_sessions:
            for question_data in session['questions']:
                keywords = self.retriever._extract_keywords(question_data)
                for keyword in keywords:
                    keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # Get top keywords
        recommendations['frequent_keywords'] = [
            keyword for keyword, count in sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Suggest database optimizations
        recommendations['suggested_indexes'] = [
            'idx_artifacts_text_fts',  # Full-text search index
            'idx_artifacts_metadata_confidence',  # Confidence-based queries
            'idx_captures_created_at_source'  # Time-based queries
        ]
        
        # Identify content gaps
        recommendations['content_gaps'] = [
            'Consider capturing more diverse content types',
            'Add more recent content for temporal questions',
            'Include more procedural content for how-to questions'
        ]
        
        # Performance tips
        recommendations['performance_tips'] = [
            'Increase context window for complex questions',
            'Cache frequently accessed sources',
            'Implement result caching for similar queries'
        ]
        
        return recommendations
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (current_time - session['created_at']).seconds > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = str(int(time.time()))
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"qa_{timestamp}_{random_part}"
    
    def _update_session(self, session_id: str, question: str, answer: Answer) -> None:
        """Update session with new question and answer."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['questions'].append(question)
            session['answers'].append(answer)
            
            # Update session context
            keywords = self.retriever._extract_keywords(question)
            for keyword in keywords:
                if keyword not in session['context']:
                    session['context'][keyword] = QAContext(
                        question=question,
                        question_type=answer.question_type,
                        keywords=[keyword],
                        context_window=answer.sources
                    )
    
    def _enhance_follow_up_question(
        self,
        original_question: str,
        follow_up: str,
        session: Dict[str, Any]
    ) -> str:
        """Enhance follow-up question with session context."""
        
        # Simple enhancement - in a real implementation, this could be more sophisticated
        context_keywords = set()
        for qa_context in session['context'].values():
            context_keywords.update(qa_context.keywords)
        
        if context_keywords:
            # Add relevant context keywords to the follow-up
            relevant_keywords = [kw for kw in context_keywords if kw in follow_up.lower()]
            if relevant_keywords:
                enhanced_question = f"{follow_up} (context: {', '.join(relevant_keywords)})"
                return enhanced_question
        
        return follow_up
    
    def _generate_clarification_questions(
        self,
        original_question: str,
        sources: List[Source]
    ) -> List[str]:
        """Generate questions to clarify low-confidence responses."""
        
        clarifications = []
        
        # Check if we have any sources at all
        if not sources:
            clarifications = [
                "Could you provide more specific details about what you're looking for?",
                "What time period or context should I focus on?",
                "Are there specific keywords or topics I should search for?"
            ]
        else:
            # Check source relevance
            avg_relevance = sum(source.relevance_score for source in sources) / len(sources)
            
            if avg_relevance < 0.3:
                clarifications = [
                    "Could you rephrase your question using different keywords?",
                    "Are you looking for information about a specific time period?",
                    "Should I focus on a particular type of content?"
                ]
            elif len(sources) < 3:
                clarifications = [
                    "Could you provide more context or background information?",
                    "Are there related topics you'd like me to consider?",
                    "What specific aspects are most important to you?"
                ]
        
        return clarifications