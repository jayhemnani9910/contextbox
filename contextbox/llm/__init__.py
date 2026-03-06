"""
LLM Backend Architecture for ContextBox.

This package provides a pluggable LLM backend architecture that supports
multiple providers with consistent interfaces and comprehensive functionality.
"""

# Core interfaces and types
from .base import (
    BaseLLMBackend,
    LLMResponse,
    ChatMessage,
    ChatRequest,
    CompletionRequest,
    EmbeddingsRequest,
    TokenUsage,
    CostInfo,
    create_chat_message,
    format_chat_messages,
    extract_text_from_response,
    TokenCounter,
    RateLimiter,
    CostTracker
)

# Configuration management
from .config import (
    LLMBackendConfig,
    ProviderConfig,
    ModelConfig,
    RateLimitConfig,
    CostConfig,
    ModelType,
    ConfigManager
)

# Custom exceptions
from .exceptions import (
    LLMBackendError,
    ConfigurationError,
    AuthenticationError,
    RateLimitError,
    TokenLimitError,
    ModelNotFoundError,
    ServiceUnavailableError,
    ResponseParsingError,
    CostTrackingError,
    RateLimiterError
)

# Backend implementations - GitHub Models (primary)
from .github_models import (
    GitHubModelsBackend,
    create_github_models_backend,
    quick_chat,
    GITHUB_MODELS,
    DEFAULT_MODEL,
)

# LLM Systems
from .qa import (
    QASystem,
    QuestionClassifier,
    ContentRetriever,
    AnswerGenerator,
    QuestionType,
    ConfidenceLevel,
    ResponseFormat,
    Source,
    QAContext,
    Answer,
)
from .summarization import (
    SummarizationManager,
    DatabaseIntegratedSummarizer,
    SummaryContent,
    SummaryRequest,
    SummaryResult,
    MultiDocumentSummary
)

__all__ = [
    # Core interfaces
    "BaseLLMBackend",
    "LLMResponse",
    "ChatMessage",
    "ChatRequest",
    "CompletionRequest",
    "EmbeddingsRequest",
    "TokenUsage",
    "CostInfo",
    "create_chat_message",
    "format_chat_messages",
    "extract_text_from_response",
    "TokenCounter",
    "RateLimiter",
    "CostTracker",

    # Configuration
    "LLMBackendConfig",
    "ProviderConfig",
    "ModelConfig",
    "RateLimitConfig",
    "CostConfig",
    "ModelType",
    "ConfigManager",

    # Exceptions
    "LLMBackendError",
    "ConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "TokenLimitError",
    "ModelNotFoundError",
    "ServiceUnavailableError",
    "ResponseParsingError",
    "CostTrackingError",
    "RateLimiterError",

    # Backend implementations
    "GitHubModelsBackend",
    "create_github_models_backend",
    "quick_chat",
    "GITHUB_MODELS",
    "DEFAULT_MODEL",

    # LLM Systems
    "QASystem",
    "QuestionClassifier",
    "ContentRetriever",
    "AnswerGenerator",
    "QuestionType",
    "ConfidenceLevel",
    "ResponseFormat",
    "Source",
    "QAContext",
    "Answer",
    "SummarizationManager",
    "DatabaseIntegratedSummarizer",
    "SummaryContent",
    "SummaryRequest",
    "SummaryResult",
    "MultiDocumentSummary"
]

__version__ = "1.0.0"
__author__ = "ContextBox Team"
