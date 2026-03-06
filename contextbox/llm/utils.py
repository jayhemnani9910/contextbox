"""
Utility implementations for LLM backends.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .config import ModelConfig, ModelType, RateLimitConfig, CostConfig
from .base import (
    TokenCounter, RateLimiter, CostTracker, ChatMessage, 
    TokenUsage, CostInfo
)
from .exceptions import (
    RateLimiterError, CostTrackingError, ServiceUnavailableError, 
    AuthenticationError
)


class SimpleTokenCounter(TokenCounter):
    """Simple token counter using character-based estimation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def count_tokens(self, text: str, model: str) -> int:
        """Estimate tokens using character count."""
        # Simple approximation: 4 characters per token
        return max(1, len(text) // 4)
    
    def count_messages(self, messages: List[ChatMessage], model: str) -> int:
        """Estimate tokens in chat messages."""
        total = 0
        for message in messages:
            # Add role token overhead
            total += 4  # Role token
            total += self.count_tokens(message.content, model)
            if message.name:
                total += 4  # Name token
        
        # Add final overhead
        total += 3
        
        return total


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    def __init__(self, config: "RateLimitConfig"):
        self.config = config
        self.requests: deque = deque()
        self.tokens: deque = deque()
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def acquire(self, model: str, estimated_tokens: int = 0):
        """Acquire rate limit permission."""
        async with self._lock:
            now = time.time()
            
            # Clean old requests (older than 1 minute)
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # Clean old tokens (older than 1 minute)
            while self.tokens and now - self.tokens[0] > 60:
                self.tokens.popleft()
            
            # Check rate limits
            current_requests = len(self.requests)
            current_tokens = sum(self.tokens)
            
            if current_requests >= self.config.requests_per_minute:
                raise RateLimitError(
                    f"Rate limit exceeded: {current_requests} requests/minute"
                )
            
            if current_tokens + estimated_tokens >= self.config.tokens_per_minute:
                raise RateLimitError(
                    f"Token rate limit exceeded: {current_tokens + estimated_tokens} tokens/minute"
                )
            
            # Add current request
            self.requests.append(now)
            self.tokens.append(estimated_tokens)
    
    def reset_if_needed(self):
        """Reset rate limiter (no-op for sliding window)."""
        pass
    
    def get_rate_limit_status(self, model: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        now = time.time()
        
        # Clean old data
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        while self.tokens and now - self.tokens[0] > 60:
            self.tokens.popleft()
        
        return {
            "requests_per_minute": {
                "current": len(self.requests),
                "limit": self.config.requests_per_minute,
                "available": max(0, self.config.requests_per_minute - len(self.requests))
            },
            "tokens_per_minute": {
                "current": sum(self.tokens),
                "limit": self.config.tokens_per_minute,
                "available": max(0, self.config.tokens_per_minute - sum(self.tokens))
            }
        }


class LeakyBucketRateLimiter(RateLimiter):
    """Leaky bucket rate limiter."""
    
    def __init__(self, config: "RateLimitConfig"):
        self.config = config
        self.bucket = 0.0
        self.last_refill = time.time()
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        
        # Initialize with full bucket
        self.bucket = config.burst_limit
    
    async def acquire(self, model: str, estimated_tokens: int = 0):
        """Acquire rate limit permission."""
        async with self._lock:
            self._refill_bucket()
            
            # Check if we have enough tokens
            if self.bucket < 1:
                raise RateLimitError(
                    f"Rate limit exceeded: bucket empty"
                )
            
            self.bucket -= 1
    
    def _refill_bucket(self):
        """Refill the bucket based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add based on refill rate
        # Refill rate = requests_per_minute / 60
        refill_rate = self.config.requests_per_minute / 60.0
        tokens_to_add = elapsed * refill_rate
        
        self.bucket = min(
            self.config.burst_limit,
            self.bucket + tokens_to_add
        )
        self.last_refill = now
    
    def reset_if_needed(self):
        """Reset rate limiter if needed."""
        self.bucket = self.config.burst_limit
        self.last_refill = time.time()
    
    def get_rate_limit_status(self, model: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        self._refill_bucket()
        
        return {
            "bucket": {
                "current": self.bucket,
                "capacity": self.config.burst_limit,
                "refill_rate": self.config.requests_per_minute / 60.0
            }
        }


class SimpleCostTracker(CostTracker):
    """Simple cost tracker implementation."""
    
    def __init__(self, cost_config: "CostConfig"):
        self.cost_config = cost_config
        self.usage_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "input_cost": 0.0,
                "output_cost": 0.0
            }
        )
        self.total_cost = 0.0
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def track_request(
        self, 
        model: str, 
        usage: TokenUsage, 
        cost_info: CostInfo
    ):
        """Track request cost and usage."""
        async with self._lock:
            stats = self.usage_stats[model]
            
            stats["requests"] += 1
            stats["prompt_tokens"] += usage.prompt_tokens
            stats["completion_tokens"] += usage.completion_tokens
            stats["total_tokens"] += usage.total_tokens
            
            if cost_info:
                stats["input_cost"] += cost_info.input_cost
                stats["output_cost"] += cost_info.output_cost
                stats["total_cost"] += cost_info.total_cost
                self.total_cost += cost_info.total_cost
    
    def get_total_cost(self, provider: Optional[str] = None) -> float:
        """Get total cost."""
        if provider:
            return sum(
                stats["total_cost"] 
                for model, stats in self.usage_stats.items()
                # In a real implementation, you'd filter by provider
            )
        
        return self.total_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return dict(self.usage_stats)
    
    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        return self.usage_stats.get(model, {})
    
    def reset_stats(self, model: Optional[str] = None):
        """Reset statistics."""
        if model:
            if model in self.usage_stats:
                del self.usage_stats[model]
        else:
            self.usage_stats.clear()
            self.total_cost = 0.0


# Retry utilities

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


async def with_retry(
    func,
    *args,
    retry_config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    **kwargs
):
    """Execute function with retry logic."""
    if retry_config is None:
        retry_config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt == retry_config.max_retries:
                raise
            
            # Calculate delay with exponential backoff
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            
            # Add jitter if enabled
            if retry_config.jitter:
                import random
                delay *= (0.5 + random.random() * 0.5)
            
            await asyncio.sleep(delay)
    
    # Should never reach here
    raise last_exception


# Response parsing utilities

def extract_text_from_openai_response(response_data: Dict[str, Any]) -> str:
    """Extract text from OpenAI-style response."""
    choices = response_data.get("choices", [])
    if not choices:
        return ""
    
    choice = choices[0]
    
    # Try different response formats
    if "message" in choice:
        return choice["message"].get("content", "")
    elif "text" in choice:
        return choice["text"]
    else:
        # Fallback to choice itself
        return str(choice)


def extract_usage_from_openai_response(response_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract usage information from OpenAI-style response."""
    usage = response_data.get("usage", {})
    
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": usage.get("finish_reason")
    }


# Error handling utilities

def is_retriable_error(error: Exception) -> bool:
    """Check if an error is retriable."""
    if isinstance(error, RateLimitError):
        return True
    elif isinstance(error, ServiceUnavailableError):
        return True
    elif isinstance(error, AuthenticationError):
        # Don't retry authentication errors
        return False
    elif isinstance(error, TokenLimitError):
        # Don't retry token limit errors
        return False
    elif isinstance(error, ModelNotFoundError):
        # Don't retry model not found errors
        return False
    else:
        # Generic retry for network errors and timeouts
        error_name = type(error).__name__.lower()
        return any(keyword in error_name for keyword in [
            'timeout', 'connection', 'network', 'temporary', 'unavailable'
        ])


def format_error_for_logging(error: Exception, context: Dict[str, Any]) -> str:
    """Format error information for logging."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add error-specific details
    if isinstance(error, RateLimitError):
        error_info["retry_after"] = getattr(error, "retry_after", None)
    elif isinstance(error, TokenLimitError):
        error_info["input_tokens"] = getattr(error, "input_tokens", None)
        error_info["max_tokens"] = getattr(error, "max_tokens", None)
    
    import json
    return json.dumps(error_info, default=str)

