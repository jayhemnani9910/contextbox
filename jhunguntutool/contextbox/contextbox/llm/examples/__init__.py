"""
ContextBox LLM Integration Examples

This package contains comprehensive examples demonstrating how to use
the ContextBox LLM integration, including the Ollama backend.
"""

from .ollama_examples import (
    basic_chat_example,
    streaming_chat_example, 
    model_management_example,
    token_counting_example,
    performance_monitoring_example,
    contextbox_integration_example,
    custom_configuration_example,
    error_handling_example,
    main
)

__all__ = [
    "basic_chat_example",
    "streaming_chat_example",
    "model_management_example", 
    "token_counting_example",
    "performance_monitoring_example",
    "contextbox_integration_example",
    "custom_configuration_example",
    "error_handling_example",
    "main"
]

__version__ = "1.0.0"
__description__ = "Examples for ContextBox LLM integration"