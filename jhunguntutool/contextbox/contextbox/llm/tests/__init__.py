"""
Test suite for ContextBox LLM integration.

This package contains comprehensive tests for all LLM backends
and integration components.
"""

# Make test modules available
try:
    from .test_ollama_integration import *
    __all__ = [
        'TestOllamaBackend',
        'TestOllamaIntegration', 
        'TestEdgeCases',
        'test_ollama_model_dataclass',
        'test_create_chat_message'
    ]
except ImportError:
    __all__ = []

__version__ = "1.0.0"
__description__ = "Tests for ContextBox LLM integration"