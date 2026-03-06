"""
Base LLM backend interface and utilities.
"""

import abc
import asyncio
import logging
import time
from typing import (
    Dict, Any, Optional, List, Union, AsyncIterator, Iterator,
    Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import LLMBackendConfig, ProviderConfig, ModelConfig, ModelType
from .exceptions import (
    LLMBackendError, RateLimitError, TokenLimitError, 
    ModelNotFoundError, ResponseParsingError, RateLimiterError
)


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add_prompt_tokens(self, tokens: int):
        """Add prompt tokens to usage."""
        self.prompt_tokens += tokens
        self.total_tokens += tokens
    
    def add_completion_tokens(self, tokens: int):
        """Add completion tokens to usage."""
        self.completion_tokens += tokens
        self.total_tokens += tokens


@dataclass
class CostInfo:
    """Cost information for a request."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    
    def add_input_cost(self, cost: float):
        """Add input cost."""
        self.input_cost += cost
        self.total_cost += cost
    
    def add_output_cost(self, cost: float):
        """Add output cost."""
        self.output_cost += cost
        self.total_cost += cost


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    provider: str
    usage: TokenUsage
    cost: Optional[CostInfo] = None
    finish_reason: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure created_at is timezone-aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)


@dataclass
class ChatMessage:
    """Chat message format."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ChatRequest:
    """Chat completion request."""
    messages: List[ChatMessage]
    model: str
    provider: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    @classmethod
    def from_text(
        cls, 
        text: str, 
        model: str, 
        provider: str,
        system_prompt: Optional[str] = None
    ) -> "ChatRequest":
        """Create chat request from simple text."""
        messages = []
        
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        messages.append(ChatMessage(role="user", content=text))
        
        return cls(
            messages=messages,
            model=model,
            provider=provider
        )


@dataclass
class CompletionRequest:
    """Text completion request."""
    prompt: str
    model: str
    provider: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class EmbeddingsRequest:
    """Embeddings request."""
    input_texts: List[str]
    model: str
    provider: str
    encoding_format: str = "float"


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting."""
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for specific model."""
        ...
    
    def count_messages(self, messages: List[ChatMessage], model: str) -> int:
        """Count tokens in chat messages."""
        ...


@runtime_checkable
class RateLimiter(Protocol):
    """Protocol for rate limiting."""
    
    async def acquire(self, model: str, estimated_tokens: int = 0):
        """Acquire rate limit token."""
        ...
    
    def reset_if_needed(self):
        """Reset rate limiter if needed."""
        ...
    
    def get_rate_limit_status(self, model: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        ...


@runtime_checkable
class CostTracker(Protocol):
    """Protocol for cost tracking."""
    
    def track_request(
        self, 
        model: str, 
        usage: TokenUsage, 
        cost_info: CostInfo
    ):
        """Track request cost."""
        ...
    
    def get_total_cost(self, provider: Optional[str] = None) -> float:
        """Get total cost."""
        ...
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        ...


class BaseLLMBackend(abc.ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(
        self,
        config: LLMBackendConfig,
        provider_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize LLM backend."""
        self.config = config
        self.provider_name = provider_name
        self.provider_config = config.get_provider(provider_name)
        self.logger = logger or self._setup_logger()
        
        # Initialize components
        self._token_counter: Optional[TokenCounter] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._cost_tracker: Optional[CostTracker] = None
        self._usage_stats: Dict[str, Any] = {}
        
        # Internal state
        self._is_initialized = False
        self._last_request_time = 0.0
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the backend."""
        logger = logging.getLogger(
            f"contextbox.llm.backend.{self.provider_name}"
        )
        
        if self.config.enable_logging:
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    if self.config.log_format == 'text'
                    else '%(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(getattr(logging, self.config.log_level))
        
        return logger
    
    def _validate_config(self):
        """Validate provider configuration."""
        try:
            self.provider_config.validate()
        except Exception as e:
            raise LLMBackendError(f"Invalid configuration: {e}")
    
    async def initialize(self):
        """Initialize the backend (async)."""
        if self._is_initialized:
            return
        
        try:
            await self._do_initialize()
            self._is_initialized = True
            self.logger.info(f"Initialized LLM backend for provider '{self.provider_name}'")
        except LLMBackendError as e:
            self.logger.error(f"Failed to initialize backend: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize backend: {e}")
            raise LLMBackendError(f"Initialization failed: {e}")
    
    async def _do_initialize(self):
        """Perform actual initialization (override in subclasses)."""
        # Default implementation does nothing
        pass
    
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._is_initialized
    
    # Abstract methods that must be implemented by subclasses
    
    @abc.abstractmethod
    async def chat_completion(
        self, 
        request: ChatRequest
    ) -> LLMResponse:
        """Perform chat completion."""
        pass
    
    @abc.abstractmethod
    async def text_completion(
        self, 
        request: CompletionRequest
    ) -> LLMResponse:
        """Perform text completion."""
        pass
    
    @abc.abstractmethod
    async def create_embeddings(
        self, 
        request: EmbeddingsRequest
    ) -> List[List[float]]:
        """Create embeddings for input texts."""
        pass
    
    # Optional abstract methods
    
    async def stream_chat_completion(
        self, 
        request: ChatRequest
    ) -> AsyncIterator[str]:
        """Stream chat completion (optional)."""
        raise NotImplementedError("Streaming not supported")
    
    async def stream_text_completion(
        self, 
        request: CompletionRequest
    ) -> AsyncIterator[str]:
        """Stream text completion (optional)."""
        raise NotImplementedError("Streaming not supported")
    
    # Utility methods for subclasses
    
    def _estimate_tokens(
        self, 
        request: Union[ChatRequest, CompletionRequest, EmbeddingsRequest]
    ) -> int:
        """Estimate tokens for a request."""
        if self._token_counter:
            if isinstance(request, ChatRequest):
                return self._token_counter.count_messages(request.messages, request.model)
            elif isinstance(request, CompletionRequest):
                return self._token_counter.count_tokens(request.prompt, request.model)
            elif isinstance(request, EmbeddingsRequest):
                return sum(
                    self._token_counter.count_tokens(text, request.model)
                    for text in request.input_texts
                )
        
        # Fallback: estimate based on character count
        if isinstance(request, ChatRequest):
            text = " ".join(msg.content for msg in request.messages)
        elif isinstance(request, CompletionRequest):
            text = request.prompt
        else:  # EmbeddingsRequest
            text = " ".join(request.input_texts)
        
        # Rough estimate: 4 characters per token
        return len(text) // 4
    
    async def _enforce_rate_limit(
        self, 
        model: str, 
        estimated_tokens: int = 0
    ):
        """Enforce rate limiting."""
        if not self._rate_limiter:
            return
        
        try:
            await self._rate_limiter.acquire(model, estimated_tokens)
        except Exception as e:
            raise RateLimiterError(f"Rate limiting failed: {e}")
    
    def _calculate_cost(
        self, 
        usage: TokenUsage, 
        model_config: ModelConfig
    ) -> Optional[CostInfo]:
        """Calculate cost for usage."""
        if not self.provider_config.cost_config.track_costs:
            return None
        
        cost_info = CostInfo()
        
        if model_config.cost_per_input_token:
            cost_info.add_input_cost(
                usage.prompt_tokens * model_config.cost_per_input_token
            )
        
        if model_config.cost_per_output_token:
            cost_info.add_output_cost(
                usage.completion_tokens * model_config.cost_per_output_token
            )
        
        return cost_info
    
    def _track_usage(
        self, 
        model: str, 
        usage: TokenUsage, 
        cost_info: Optional[CostInfo] = None
    ):
        """Track usage statistics."""
        if self._cost_tracker and cost_info:
            self._cost_tracker.track_request(model, usage, cost_info)
        
        # Update internal stats
        self._usage_stats.setdefault(model, {
            "requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        })
        
        stats = self._usage_stats[model]
        stats["requests"] += 1
        stats["total_tokens"] += usage.total_tokens
        if cost_info:
            stats["total_cost"] += cost_info.total_cost
    
    def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        model: str
    ) -> LLMResponse:
        """Parse provider response into standardized format."""
        try:
            # Extract content based on response structure
            content = self._extract_content(response_data)
            
            # Extract usage
            usage_data = response_data.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
            
            # Extract metadata
            metadata = {
                "raw_response": response_data,
                "finish_reason": usage_data.get("finish_reason")
            }
            
            # Calculate cost if possible
            model_config = self.provider_config.get_model(model)
            cost_info = self._calculate_cost(usage, model_config)
            
            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                cost=cost_info,
                finish_reason=usage_data.get("finish_reason"),
                metadata=metadata
            )
            
        except Exception as e:
            raise ResponseParsingError(f"Failed to parse response: {e}")
    
    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract content from response data (override in subclasses)."""
        # Default implementation assumes OpenAI-style response
        choices = response_data.get("choices", [])
        if not choices:
            return ""
        
        choice = choices[0]
        return choice.get("message", {}).get("content", "")
    
    # Configuration and lifecycle management
    
    def set_token_counter(self, counter: TokenCounter):
        """Set token counter implementation."""
        self._token_counter = counter
    
    def set_rate_limiter(self, limiter: RateLimiter):
        """Set rate limiter implementation."""
        self._rate_limiter = limiter
    
    def set_cost_tracker(self, tracker: CostTracker):
        """Set cost tracker implementation."""
        self._cost_tracker = tracker
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        if not self._is_initialized:
            return {"status": "not_initialized", "healthy": False}
        
        try:
            # Try a simple request
            request = ChatRequest.from_text(
                "Hello", 
                self.provider_config.default_model or "test",
                self.provider_name
            )
            await self.chat_completion(request)
            
            return {
                "status": "healthy",
                "healthy": True,
                "provider": self.provider_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "provider": self.provider_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self):
        """Cleanup resources."""
        self.logger.info(f"Closing LLM backend for provider '{self.provider_name}'")
        self._is_initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Utility functions

def create_chat_message(role: str, content: str, name: Optional[str] = None) -> ChatMessage:
    """Create a chat message."""
    return ChatMessage(role=role, content=content, name=name)


def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages as a single string."""
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


def extract_text_from_response(response: LLMResponse) -> str:
    """Extract text content from LLM response."""
    return response.content
