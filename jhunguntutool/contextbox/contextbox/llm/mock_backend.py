"""
Mock LLM backend for demonstration purposes.

This module provides a simple rule-based summarization backend that can be used
when actual LLM backends are not available for testing and demonstration.
"""

import re
import random
from typing import Dict, Any, Tuple

from .exceptions import LLMBackendError
from .config import ModelConfig


class MockLLMBackend:
    """Mock LLM backend for demonstration purposes."""
    
    def __init__(self):
        self.name = "mock_llm"
        self.processed_requests = 0
    
    def generate_summary(self, content: str, prompt: str, config: ModelConfig) -> Tuple[str, Dict[str, Any]]:
        """Generate a mock summary using rule-based approach."""
        try:
            self.processed_requests += 1
            
            # Extract key elements from content
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Analyze content type from prompt
            content_type = self._detect_content_type(prompt)
            
            # Generate summary based on content type and length
            if "brief" in prompt.lower() or "concise" in prompt.lower():
                summary = self._generate_brief_summary(sentences, content_type)
            elif "executive" in prompt.lower():
                summary = self._generate_executive_summary(sentences, content_type)
            else:
                summary = self._generate_detailed_summary(sentences, content_type)
            
            # Add some randomness to simulate real LLM behavior
            if random.random() < 0.1:  # 10% chance of minor issues
                summary = self._add_minor_error(summary)
            
            return summary, {
                'provider': 'mock_llm',
                'model': config.name,
                'total_tokens': len(content.split()) + len(summary.split()),
                'mock_generated': True
            }
            
        except Exception as e:
            raise LLMBackendError(
                f"Mock LLM generation failed: {str(e)}",
                provider="mock_llm",
                model=config.name,
                details={'error': str(e)}
            )
    
    def _detect_content_type(self, prompt: str) -> str:
        """Detect content type from prompt."""
        prompt_lower = prompt.lower()
        
        if "article" in prompt_lower or "blog" in prompt_lower:
            return "article"
        elif "transcript" in prompt_lower or "speech" in prompt_lower:
            return "transcript"
        elif "documentation" in prompt_lower or "guide" in prompt_lower:
            return "documentation"
        elif "news" in prompt_lower:
            return "news"
        elif "code" in prompt_lower:
            return "code"
        else:
            return "general"
    
    def _generate_brief_summary(self, sentences: list, content_type: str) -> str:
        """Generate a brief summary."""
        if not sentences:
            return "No content available for summarization."
        
        # Take first 1-2 most important sentences
        if content_type == "news":
            # For news, focus on first sentence (typically the lead)
            key_sentence = sentences[0] if sentences else ""
            return f"Key development: {key_sentence}"
        elif content_type == "documentation":
            # For documentation, focus on first meaningful sentence
            key_sentence = next((s for s in sentences if len(s) > 20), sentences[0])
            return f"This documentation covers: {key_sentence}"
        else:
            # General case - combine first sentence with key terms
            first_sentence = sentences[0]
            return f"Main points: {first_sentence}"
    
    def _generate_executive_summary(self, sentences: list, content_type: str) -> str:
        """Generate an executive summary."""
        if not sentences:
            return "Executive summary not available."
        
        # Identify key concepts and actions
        key_points = []
        
        for sentence in sentences[:3]:  # Use first 3 sentences
            if any(word in sentence.lower() for word in ["important", "key", "critical", "significant"]):
                key_points.append(sentence)
        
        if not key_points:
            key_points = sentences[:2]
        
        summary_parts = ["Executive Summary:"]
        for point in key_points[:3]:
            summary_parts.append(f"• {point}")
        
        summary_parts.append("\nStrategic Implications:")
        summary_parts.append("• Requires executive attention and resource allocation")
        summary_parts.append("• Potential impact on organizational objectives")
        
        return "\n".join(summary_parts)
    
    def _generate_detailed_summary(self, sentences: list, content_type: str) -> str:
        """Generate a detailed summary."""
        if not sentences:
            return "Detailed summary not available."
        
        # Organize content into logical sections
        sections = {
            "main_points": [],
            "supporting_details": [],
            "implications": []
        }
        
        # Categorize sentences
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["main", "primary", "key", "important", "central"]):
                sections["main_points"].append(sentence)
            elif any(word in sentence.lower() for word in ["however", "but", "although", "despite"]):
                sections["supporting_details"].append(sentence)
            else:
                sections["implications"].append(sentence)
        
        # Build detailed summary
        summary_parts = ["Detailed Summary:\n"]
        
        if sections["main_points"]:
            summary_parts.append("Main Points:")
            for point in sections["main_points"][:3]:
                summary_parts.append(f"• {point}")
            summary_parts.append("")
        
        if sections["supporting_details"]:
            summary_parts.append("Supporting Information:")
            for detail in sections["supporting_details"][:2]:
                summary_parts.append(f"• {detail}")
            summary_parts.append("")
        
        if sections["implications"]:
            summary_parts.append("Key Implications:")
            for implication in sections["implications"][:2]:
                summary_parts.append(f"• {implication}")
        
        return "\n".join(summary_parts)
    
    def _add_minor_error(self, summary: str) -> str:
        """Add a minor formatting error for realism."""
        if random.random() < 0.5:
            # Remove period at the end
            return summary.rstrip('.') + '.'
        else:
            # Add extra space
            return re.sub(r'\s+', ' ', summary)


# Example usage function
def create_mock_summarization_system():
    """Create a mock summarization system for demonstration."""
    from .summarization import SummarizationManager
    
    # Create manager with mock backend
    manager = SummarizationManager()
    
    # Add mock backend
    mock_backend = MockLLMBackend()
    manager.backends['mock'] = mock_backend
    
    # Make mock the default backend
    manager.config.default_provider = 'mock'
    
    return manager


if __name__ == "__main__":
    # Quick test
    backend = MockLLMBackend()
    
    test_content = """
    Artificial Intelligence is transforming industries worldwide. Machine learning algorithms 
    can now process vast amounts of data efficiently. Deep learning has enabled breakthroughs 
    in computer vision and natural language processing. However, there are concerns about 
    algorithmic bias and job displacement. The technology continues to evolve rapidly.
    """
    
    test_prompt = "Provide a detailed summary of this AI article."
    
    from .config import ModelConfig, ModelType
    
    config = ModelConfig(
        name="mock_model",
        model_type=ModelType.CHAT,
        provider="mock"
    )
    
    summary, info = backend.generate_summary(test_content, test_prompt, config)
    
    print("Mock Summary Test:")
    print(f"Summary: {summary}")
    print(f"Info: {info}")