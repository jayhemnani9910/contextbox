"""
OpenAI backend implementation for ContextBox.

This module provides a comprehensive OpenAI API integration with support for:
- Multiple models (GPT-4, GPT-3.5, etc.)
- Streaming responses
- Function calling and structured outputs
- Token counting and cost tracking
- Rate limiting with exponential backoff
- Error handling for all OpenAI API response scenarios
- Secure API key management
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, AsyncIterator, Union
from datetime import datetime, timezone

# OpenAI client - optional dependency
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None
    ChatCompletion = None
    ChatCompletionChunk = None

# ContextBox imports
from .base import BaseLLMBackend, ChatRequest, CompletionRequest, EmbeddingsRequest
from .base import LLMResponse, TokenUsage, CostInfo, ChatMessage
from .config import LLMBackendConfig, ProviderConfig, ModelConfig, ModelType
from .exceptions import (
    LLMBackendError, AuthenticationError, RateLimitError, 
    TokenLimitError, ModelNotFoundError, ServiceUnavailableError,
    ResponseParsingError, ConfigurationError
)
from .utils import (
    SimpleTokenCounter, SlidingWindowRateLimiter, SimpleCostTracker,
    with_retry, extract_text_from_openai_response,
    extract_usage_from_openai_response, is_retriable_error
)


class OpenAIBackend(BaseLLMBackend):
    """OpenAI backend implementation."""
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "input_cost_per_1k": 0.0015,
            "output_cost_per_1k": 0.002,
            "supports_function_calling": True,
            "supports_vision": False,
            "supports_streaming": True
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.004,
            "supports_function_calling": True,
            "supports_vision": False,
            "supports_streaming": True
        },
        "gpt-4": {
            "max_tokens": 8192,
            "input_cost_per_1k": 0.03,
            "output_cost_per_1k": 0.06,
            "supports_function_calling": True,
            "supports_vision": False,
            "supports_streaming": True
        },
        "gpt-4-32k": {
            "max_tokens": 32768,
            "input_cost_per_1k": 0.06,
            "output_cost_per_1k": 0.12,
            "supports_function_calling": True,
            "supports_vision": False,
            "supports_streaming": True
        },
        "gpt-4-turbo": {
            "max_tokens": 128000,
            "input_cost_per_1k": 0.01,
            "output_cost_per_1k": 0.03,
            "supports_function_calling": True,
            "supports_vision": False,
            "supports_streaming": True
        },
        "gpt-4o": {
            "max_tokens": 128000,
            "input_cost_per_1k": 0.005,
            "output_cost_per_1k": 0.015,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_streaming": True
        },
        "gpt-4o-mini": {
            "max_tokens": 128000,
            "input_cost_per_1k": 0.00015,
            "output_cost_per_1k": 0.0006,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_streaming": True
        },
        "text-embedding-3-small": {
            "max_tokens": 8192,
            "input_cost_per_1k": 0.00002,
            "output_cost_per_1k": 0.0,
            "supports_function_calling": False,
            "supports_vision": False,
            "supports_streaming": False
        },
        "text-embedding-3-large": {
            "max_tokens": 8192,
            "input_cost_per_1k": 0.00013,
            "output_cost_per_1k": 0.0,
            "supports_function_calling": False,
            "supports_vision": False,
            "supports_streaming": False
        }
    }
    
    def __init__(
        self,
        config: LLMBackendConfig,
        provider_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize OpenAI backend."""
        if not OPENAI_AVAILABLE:
            raise LLMBackendError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        super().__init__(config, provider_name, logger)
        
        # OpenAI client
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        
        # Initialize utilities
        self._token_counter = SimpleTokenCounter()
        self._rate_limiter = SlidingWindowRateLimiter(
            self.provider_config.rate_limit
        )
        self._cost_tracker = SimpleCostTracker(
            self.provider_config.cost_config
        )
        
        # Retry configuration
        self._retry_config = RetryConfig(
            max_retries=self.provider_config.max_retries,
            base_delay=self.provider_config.retry_delay,
            max_delay=60.0,
            exponential_base=2.0
        )
        
        self.logger.info(f"OpenAI backend initialized for provider '{provider_name}'")
    
    async def _do_initialize(self):
        """Initialize OpenAI client."""
        if not self.provider_config.api_key:
            raise AuthenticationError(
                "OpenAI API key not provided",
                provider=self.provider_name
            )
        
        # Create clients
        self._client = OpenAI(
            api_key=self.provider_config.api_key,
            timeout=self.provider_config.timeout
        )
        
        self._async_client = AsyncOpenAI(
            api_key=self.provider_config.api_key,
            timeout=self.provider_config.timeout
        )
        
        # Test connection
        await self.test_connection()
    
    async def test_connection(self) -> bool:
        """Test connection to OpenAI API."""
        try:
            if not self._async_client:
                return False
            
            # Try a simple models list request
            models = await self._async_client.models.list()
            return len(models.data) > 0
            
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            raise ServiceUnavailableError(
                f"Failed to connect to OpenAI: {e}",
                provider=self.provider_name
            )
    
    def validate_api_key(self) -> bool:
        """Validate the API key."""
        try:
            if not self._client:
                return False
            
            # Try a simple request to validate key
            # Use a lightweight models list request
            models = self._client.models.list()
            return len(models.data) > 0
            
        except Exception as e:
            self.logger.warning(f"API key validation failed: {e}")
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about an OpenAI model."""
        return OpenAIModelInfo.get_model_info(model)
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for specific model."""
        return self._token_counter.count_tokens(text, model)
    
    async def chat_completion(self, request: ChatRequest) -> LLMResponse:
        """Perform chat completion."""
        await self.initialize()
        
        # Validate request
        self._validate_chat_request(request)
        
        # Estimate tokens and check limits
        estimated_tokens = self._estimate_tokens(request)
        model_config = OpenAIModelInfo.get_model_info(request.model)
        max_allowed = model_config["context_length"]
        
        if estimated_tokens > max_allowed:
            raise TokenLimitError(
                f"Request exceeds token limit: {estimated_tokens} > {max_allowed}",
                input_tokens=estimated_tokens,
                max_tokens=max_allowed,
                model=request.model,
                provider=self.provider_name
            )
        
        # Enforce rate limiting
        await self._enforce_rate_limit(request.model, estimated_tokens)
        
        try:
            # Convert to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]
            
            # Build request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": request.stream
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            # Make request with retry
            response = await with_retry(
                self._make_chat_completion_request,
                params,
                retry_config=self._retry_config
            )
            
            # Parse response
            return self._parse_chat_response(response, request.model)
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise self._handle_openai_error(e)
    
    async def stream_chat_completion(
        self, 
        request: ChatRequest
    ) -> AsyncIterator[str]:
        """Perform streaming chat completion."""
        await self.initialize()
        
        # Validate request
        self._validate_chat_request(request)
        
        # Set stream to True
        request.stream = True
        
        # Estimate tokens and check limits
        estimated_tokens = self._estimate_tokens(request)
        model_config = OpenAIModelInfo.get_model_info(request.model)
        max_allowed = model_config["context_length"]
        
        if estimated_tokens > max_allowed:
            raise TokenLimitError(
                f"Request exceeds token limit: {estimated_tokens} > {max_allowed}",
                input_tokens=estimated_tokens,
                max_tokens=max_allowed,
                model=request.model,
                provider=self.provider_name
            )
        
        # Enforce rate limiting
        await self._enforce_rate_limit(request.model, estimated_tokens)
        
        try:
            # Convert to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]
            
            # Build request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": True
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            # Make streaming request
            stream = await self._async_client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise self._handle_openai_error(e)
    
    async def text_completion(self, request: CompletionRequest) -> LLMResponse:
        """Perform text completion."""
        await self.initialize()
        
        # For now, convert to chat completion
        messages = [
            ChatMessage(role="user", content=request.prompt)
        ]
        
        chat_request = ChatRequest(
            messages=messages,
            model=request.model,
            provider=request.provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop_sequences=request.stop_sequences,
            stream=request.stream
        )
        
        return await self.chat_completion(chat_request)
    
    async def create_embeddings(self, request: EmbeddingsRequest) -> List[List[float]]:
        """Create embeddings for input texts."""
        await self.initialize()
        
        try:
            response = await self._async_client.embeddings.create(
                model=request.model,
                input=request.input_texts,
                encoding_format=request.encoding_format
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            self.logger.error(f"Embeddings creation failed: {e}")
            raise self._handle_openai_error(e)
    
    def _validate_chat_request(self, request: ChatRequest):
        """Validate chat request."""
        if not request.messages:
            raise LLMBackendError("No messages provided")
        
        # Validate model
        if not self.is_model_available(request.model):
            raise ModelNotFoundError(
                f"Model '{request.model}' not available",
                model=request.model,
                provider=self.provider_name
            )
        
        # Validate parameters
        model_config = self.get_model_info(request.model)
        
        if not 0 <= request.temperature <= 2:
            raise LLMBackendError("Temperature must be between 0 and 2")
        
        if not 0 <= request.top_p <= 1:
            raise LLMBackendError("Top-p must be between 0 and 1")
        
        if request.max_tokens and request.max_tokens > model_config["max_tokens"]:
            raise LLMBackendError(
                f"max_tokens exceeds model limit: {request.max_tokens} > {model_config['max_tokens']}"
            )
    
    async def _make_chat_completion_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make actual chat completion request."""
        response = await self._async_client.chat.completions.create(**params)
        return response.dict()
    
    def _parse_chat_response(
        self, 
        response_data: Dict[str, Any], 
        model: str
    ) -> LLMResponse:
        """Parse OpenAI chat completion response."""
        try:
            # Extract content
            content = extract_text_from_openai_response(response_data)
            
            # Extract usage
            usage_data = extract_usage_from_openai_response(response_data)
            usage = TokenUsage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"]
            )
            
            # Calculate cost
            model_config = OpenAIModelInfo.get_model_info(model)
            cost_info = CostInfo()
            
            if model_config["input_price_per_1k"] > 0:
                cost_info.input_cost = (
                    usage.prompt_tokens / 1000 * model_config["input_price_per_1k"]
                )
                cost_info.total_cost += cost_info.input_cost
            
            if model_config["output_price_per_1k"] > 0:
                cost_info.output_cost = (
                    usage.completion_tokens / 1000 * model_config["output_price_per_1k"]
                )
                cost_info.total_cost += cost_info.output_cost
            
            # Create response
            response = LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                cost=cost_info if cost_info.total_cost > 0 else None,
                finish_reason=usage_data.get("finish_reason"),
                metadata={
                    "raw_response": response_data,
                    "model_config": model_config
                }
            )
            
            # Track usage
            self._track_usage(model, usage, cost_info)
            
            return response
            
        except Exception as e:
            raise ResponseParsingError(f"Failed to parse OpenAI response: {e}")
    
    def _handle_openai_error(self, error: Exception) -> LLMBackendError:
        """Handle OpenAI API errors."""
        error_message = str(error)
        
        # Handle specific OpenAI errors
        if "401" in error_message or "unauthorized" in error_message.lower():
            return AuthenticationError(
                "Invalid OpenAI API key or unauthorized access",
                provider=self.provider_name,
                error_code="401"
            )
        
        elif "429" in error_message or "rate limit" in error_message.lower():
            # Extract retry-after if available
            retry_after = None
            if hasattr(error, 'response') and error.response:
                retry_after = error.response.headers.get('Retry-After')
                if retry_after:
                    retry_after = int(retry_after)
            
            return RateLimitError(
                f"OpenAI rate limit exceeded: {error_message}",
                retry_after=retry_after,
                provider=self.provider_name
            )
        
        elif "quota" in error_message.lower():
            return LLMBackendError(
                f"OpenAI quota exceeded: {error_message}",
                provider=self.provider_name,
                error_code="quota_exceeded"
            )
        
        elif "model not found" in error_message.lower():
            return ModelNotFoundError(
                f"OpenAI model not found: {error_message}",
                provider=self.provider_name
            )
        
        elif "503" in error_message or "unavailable" in error_message.lower():
            return ServiceUnavailableError(
                f"OpenAI service unavailable: {error_message}",
                provider=self.provider_name
            )
        
        else:
            # Generic error
            return LLMBackendError(
                f"OpenAI API error: {error_message}",
                provider=self.provider_name,
                details={"original_error": str(error)}
            )
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        return model in OpenAIModelInfo.MODEL_DATA
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return OpenAIModelInfo.get_supported_models()
    
    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return self._rate_limiter.get_rate_limit_status("default")
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self._cost_tracker.get_usage_stats()
        rate_limit_status = await self.get_rate_limit_status()
        
        return {
            "cost_tracking": stats,
            "rate_limits": rate_limit_status,
            "provider": self.provider_name
        }
    
    async def close(self):
        """Cleanup resources."""
        if self._async_client:
            await self._async_client.close()
        
        await super().close()


# Factory function for easy backend creation

def create_openai_backend(
    api_key: str,
    provider_name: str = "openai",
    **kwargs
) -> OpenAIBackend:
    """Create an OpenAI backend with default configuration."""
    from .config import RateLimitConfig, CostConfig
    
    # Default configurations
    rate_limit = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        tokens_per_minute=100000,
        burst_limit=10
    )
    
    cost_config = CostConfig(
        track_usage=True,
        track_costs=True,
        budget_limit=None,
        alert_threshold=0.8
    )
    
    # Create provider config
    provider_config = ProviderConfig(
        name=provider_name,
        api_key=api_key,
        models={
            model: ModelConfig(
                name=model,
                model_type=ModelType.CHAT,
                provider=provider_name,
                max_tokens=config["max_tokens"],
                cost_per_input_token=config["input_cost_per_1k"] / 1000,
                cost_per_output_token=config["output_cost_per_1k"] / 1000
            )
            for model, config in OpenAIBackend.MODEL_CONFIGS.items()
        },
        default_model="gpt-4",
        timeout=30,
        max_retries=3,
        retry_delay=1.0,
        rate_limit=rate_limit,
        cost_config=cost_config
    )
    
    # Create main config
    config = LLMBackendConfig(
        providers={provider_name: provider_config},
        default_provider=provider_name,
        enable_logging=True,
        log_level="INFO",
        log_format="json",
        enable_monitoring=True,
        monitoring_interval=60
    )
    
    return OpenAIBackend(config, provider_name)


# Utility functions

def get_supported_models() -> List[str]:
    """Get list of supported OpenAI models."""
    return list(OpenAIBackend.MODEL_CONFIGS.keys())


def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing information for a model."""
    if model not in OpenAIBackend.MODEL_CONFIGS:
        return {"input": 0.0, "output": 0.0}
    
    config = OpenAIBackend.MODEL_CONFIGS[model]
    return {
        "input": config["input_cost_per_1k"],
        "output": config["output_cost_per_1k"]
    }


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for given tokens."""
    if model not in OpenAIBackend.MODEL_CONFIGS:
        return 0.0
    
    config = OpenAIBackend.MODEL_CONFIGS[model]
    input_cost = (input_tokens / 1000) * config["input_cost_per_1k"]
    output_cost = (output_tokens / 1000) * config["output_cost_per_1k"]
    
    return input_cost + output_cost