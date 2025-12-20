"""
Custom exceptions for LLM backend operations.
"""

from typing import Optional, Dict, Any


class LLMBackendError(Exception):
    """Base exception for all LLM backend errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(LLMBackendError):
    """Raised when there's a configuration issue."""
    pass


class AuthenticationError(LLMBackendError):
    """Raised when authentication fails."""
    pass


class RateLimitError(LLMBackendError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TokenLimitError(LLMBackendError):
    """Raised when token limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        input_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens


class ModelNotFoundError(LLMBackendError):
    """Raised when the specified model is not available."""
    pass


class ServiceUnavailableError(LLMBackendError):
    """Raised when the LLM service is unavailable."""
    pass


class ResponseParsingError(LLMBackendError):
    """Raised when response parsing fails."""
    pass


class CostTrackingError(LLMBackendError):
    """Raised when cost tracking operations fail."""
    pass


class RateLimiterError(LLMBackendError):
    """Raised when rate limiting operations fail."""
    pass