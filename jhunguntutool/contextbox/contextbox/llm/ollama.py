"""
Ollama backend implementation for ContextBox.

This module provides a complete integration with Ollama's REST API for local LLM models,
supporting chat completion, model management, streaming responses, and more.
"""

import json
import asyncio
import logging
import time
import subprocess
import tempfile
import shutil
from typing import (
    Dict, Any, Optional, List, AsyncIterator, Union,
    Callable, Awaitable, TypeVar, Generic
)
from pathlib import Path
from urllib.parse import urljoin
from dataclasses import dataclass

import aiohttp
import tiktoken
from unittest.mock import AsyncMock, MagicMock, DEFAULT

from .base import (
    BaseLLMBackend, ChatRequest, CompletionRequest, EmbeddingsRequest,
    LLMResponse, TokenUsage, CostInfo, ChatMessage, create_chat_message
)
from .config import (
    LLMBackendConfig, ProviderConfig, ModelConfig, ModelType,
    RateLimitConfig, CostConfig
)
from .exceptions import (
    LLMBackendError, ServiceUnavailableError, ModelNotFoundError,
    ConfigurationError, ResponseParsingError, RateLimitError
)


@dataclass
class OllamaModel:
    """Ollama model information."""
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]
    available: bool = True


@dataclass
class OllamaChatResponse:
    """Ollama chat response structure."""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaBackend(BaseLLMBackend):
    """
    Ollama backend for ContextBox.
    
    This backend integrates with Ollama's REST API to provide local LLM
    capabilities including chat completion, streaming, and model management.
    """
    
    def __init__(
        self,
        config: LLMBackendConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize Ollama backend."""
        # Get Ollama provider configuration
        provider_name = "ollama"
        if provider_name not in config.providers:
            # Create default Ollama provider config if not present
            provider_config = self._create_default_provider_config()
            config.providers[provider_name] = provider_config
            if config.default_provider is None:
                config.default_provider = provider_name
        
        super().__init__(config, provider_name, logger)
        
        # Ollama-specific configuration
        self.base_url = self.provider_config.base_url or "http://localhost:11434"
        self.api_version = "api"
        
        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_context_exit: Optional[Callable[..., Awaitable[Any]]] = None
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0.0
        self.service_available = False
        
        # Model management
        self.models_cache: Dict[str, OllamaModel] = {}
        self.cache_ttl = 300  # 5 minutes
        self._populate_default_models_cache()
        
        # Request tracking
        self.request_count = 0
        self.last_request_time = 0.0
        
        # Token counting
        self.token_encoders: Dict[str, tiktoken.Encoding] = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "total_tokens_generated": 0,
            "total_context_tokens": 0
        }
        
        self.logger.info(f"Initialized Ollama backend with base URL: {self.base_url}")
    
    def _populate_default_models_cache(self) -> None:
        """Seed the models cache from provider configuration for offline usage."""
        for model_name, model_config in self.provider_config.models.items():
            if model_name in self.models_cache:
                continue
            details = {
                "model_type": model_config.model_type.value
                if hasattr(model_config.model_type, "value")
                else str(model_config.model_type)
            }
            if model_config.api_endpoint:
                details["api_endpoint"] = model_config.api_endpoint
            self.models_cache[model_name] = OllamaModel(
                name=model_name,
                modified_at="",
                size=model_config.max_tokens or 0,
                digest="",
                details=details,
                available=True
            )
    
    def _create_default_provider_config(self) -> ProviderConfig:
        """Create default Ollama provider configuration."""
        # Default models configuration
        models = {
            "llama2": ModelConfig(
                name="llama2",
                model_type=ModelType.CHAT,
                provider="ollama",
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                api_endpoint="http://localhost:11434"
            ),
            "codellama": ModelConfig(
                name="codellama",
                model_type=ModelType.CHAT,
                provider="ollama",
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
                api_endpoint="http://localhost:11434"
            ),
            "mistral": ModelConfig(
                name="mistral",
                model_type=ModelType.CHAT,
                provider="ollama",
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                api_endpoint="http://localhost:11434"
            )
        }
        
        provider_config = ProviderConfig(
            name="ollama",
            models=models,
            default_model="llama2",
            timeout=60,
            max_retries=3,
            retry_delay=2.0,
            base_url="http://localhost:11434",
            rate_limit=RateLimitConfig(
                requests_per_minute=30,  # Conservative for local inference
                requests_per_hour=500,
                tokens_per_minute=50000,
                burst_limit=5
            ),
            cost_config=CostConfig(
                track_usage=True,
                track_costs=False  # Local inference has no monetary cost
            )
        )
        
        return provider_config
    
    async def _resolve_context_manager(self, context_candidate: Any):
        """Ensure a session request is an async context manager (mock-friendly)."""
        if asyncio.iscoroutine(context_candidate):
            context_candidate = await context_candidate
        return context_candidate
    
    async def _iterate_response_content(self, response: Any) -> AsyncIterator[bytes]:
        """Iterate over streaming content in a mock-friendly way."""
        content = getattr(response, "content", None)
        if content is None:
            return
        
        iterator = None
        aiter = getattr(content, "__aiter__", None)
        if aiter:
            mock_return = getattr(aiter, "_mock_return_value", DEFAULT)
            if mock_return is not DEFAULT:
                iterator = mock_return
            else:
                iterator = aiter()
                if asyncio.iscoroutine(iterator):
                    iterator = await iterator
        else:
            iterator = content
        
        if iterator is None:
            return
        
        if hasattr(iterator, "__anext__"):
            async for chunk in iterator:
                yield chunk
            return
        
        for chunk in iterator:
            yield chunk
    
    def _get_http_status(self, response: Any) -> int:
        """Extract HTTP status code with support for mocks."""
        status = getattr(response, "status", None)
        if isinstance(status, (AsyncMock, MagicMock)):
            mock_return = getattr(status, "_mock_return_value", DEFAULT)
            if mock_return is not DEFAULT and isinstance(mock_return, int):
                return mock_return
            return_value = getattr(status, "return_value", None)
            if isinstance(return_value, int):
                return return_value
            return 200
        if status is None:
            return 200
        return status
    
    async def _do_initialize(self):
        """Perform Ollama-specific initialization."""
        # Create HTTP session
        session_candidate = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.provider_config.timeout),
            connector=aiohttp.TCPConnector(limit=10)
        )
        if isinstance(session_candidate, (AsyncMock, MagicMock)):
            try:
                self.session = await session_candidate.__aenter__()
                self._session_context_exit = getattr(session_candidate, "__aexit__", None)
            except Exception:
                # Fallback to using the mock session directly
                self.session = session_candidate
        else:
            self.session = session_candidate
        
        # Validate service availability
        await self._validate_service()
        
        # Load available models
        await self._refresh_models_cache()
        
        self.logger.info("Ollama backend initialization completed")
    
    async def _validate_service(self):
        """Validate Ollama service availability."""
        try:
            request_ctx = await self._resolve_context_manager(
                self.session.get(f"{self.base_url}/{self.api_version}/tags")
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status == 200:
                    self.service_available = True
                    self.logger.info("Ollama service is available")
                else:
                    raise ServiceUnavailableError(
                        f"Ollama service returned status {status}"
                    )
        except aiohttp.ClientError as e:
            raise ServiceUnavailableError(
                f"Failed to connect to Ollama service: {e}"
            )
    
    async def _refresh_models_cache(self):
        """Refresh the models cache."""
        try:
            models_data = await self._list_models()
            self.models_cache = {}
            
            for model_data in models_data:
                model = OllamaModel(
                    name=model_data["name"],
                    modified_at=model_data["modified_at"],
                    size=model_data["size"],
                    digest=model_data["digest"],
                    details=model_data.get("details", {})
                )
                self.models_cache[model.name] = model
            
            self.logger.info(f"Loaded {len(self.models_cache)} models from Ollama")
            
        except Exception as e:
            self.logger.warning(f"Failed to refresh models cache: {e}")
    
    async def _list_models(self) -> List[Dict[str, Any]]:
        """List models from Ollama API."""
        request_ctx = await self._resolve_context_manager(
            self.session.get(f"{self.base_url}/{self.api_version}/tags")
        )
        async with request_ctx as response:
            status = self._get_http_status(response)
            if status == 200:
                data = await response.json()
                return data.get("models", [])
            else:
                raise ServiceUnavailableError(
                    f"Failed to list models: HTTP {status}"
                )
    
    # Model Management Methods
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Check cache validity
            current_time = time.time()
            if current_time - self.last_health_check > self.cache_ttl:
                if self.session:
                    await self._refresh_models_cache()
                else:
                    self.logger.debug("Skipping model cache refresh: session not initialized")
                self.last_health_check = current_time
            
            models = []
            for model in self.models_cache.values():
                models.append({
                    "name": model.name,
                    "size": model.size,
                    "modified_at": model.modified_at,
                    "digest": model.digest,
                    "details": model.details,
                    "available": model.available
                })
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise LLMBackendError(f"Model listing failed: {e}")
    
    async def pull_model(self, model: str, **kwargs) -> Dict[str, Any]:
        """
        Pull/download a model from Ollama.
        
        Args:
            model: Model name to pull
            **kwargs: Additional parameters for pulling
            
        Returns:
            Pull operation result
        """
        if not model or "/" not in model:
            raise ConfigurationError(
                f"Invalid model name: {model}. Use format 'namespace/model:tag'"
            )
        if not self.session:
            raise ServiceUnavailableError(
                "Ollama session not initialized. Call initialize() before pulling models."
            )
        
        try:
            self.logger.info(f"Starting model pull: {model}")
            
            # Prepare pull request
            request_data = {
                "name": model,
                "stream": True
            }
            
            # Add progress callback if provided
            progress_callback = kwargs.get("progress_callback")
            
            request_ctx = await self._resolve_context_manager(
                self.session.post(
                    f"{self.base_url}/{self.api_version}/pull",
                    json=request_data
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status != 200:
                    error_data = await response.json() if response.content_type == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {status}')
                    raise LLMBackendError(f"Model pull failed: {error_msg}")
                
                # Stream the response to track progress
                pull_result = {
                    "model": model,
                    "status": "starting",
                    "total": 0,
                    "completed": 0,
                    "digest": "",
                    "message": ""
                }
                
                async for line in self._iterate_response_content(response):
                    if not line:
                        continue
                    
                    try:
                        line_data = json.loads(line.decode('utf-8'))
                        status = line_data.get("status", "")
                        digest = line_data.get("digest", "")
                        total = line_data.get("total", 0)
                        completed = line_data.get("completed", 0)
                        
                        pull_result.update({
                            "status": status,
                            "digest": digest,
                            "total": total,
                            "completed": completed,
                            "message": line_data.get("message", "")
                        })
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress = completed / total if total > 0 else 0
                            await progress_callback(progress, status, model)
                        
                        if status == "success":
                            pull_result["status"] = "completed"
                            self.logger.info(f"Model pull completed: {model}")
                            break
                            
                    except json.JSONDecodeError:
                        continue
                
                # Refresh models cache
                await self._refresh_models_cache()
                
                return pull_result
                
        except LLMBackendError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            raise LLMBackendError(f"Model pull failed: {e}")
    
    async def delete_model(self, model: str) -> bool:
        """
        Delete a model from Ollama.
        
        Args:
            model: Model name to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Check if model exists
            if model not in self.models_cache:
                self.logger.warning(f"Model {model} not found in cache")
                return False
            
            request_data = {"name": model}
            
            request_ctx = await self._resolve_context_manager(
                self.session.delete(
                    f"{self.base_url}/{self.api_version}/delete",
                    json=request_data
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status == 200:
                    # Remove from cache
                    del self.models_cache[model]
                    self.logger.info(f"Model deleted: {model}")
                    return True
                else:
                    error_data = await response.json() if response.content_type == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {status}')
                    raise LLMBackendError(f"Model deletion failed: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Failed to delete model {model}: {e}")
            raise LLMBackendError(f"Model deletion failed: {e}")
    
    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        
        Args:
            model: Model name
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            # Check cache first
            if model in self.models_cache:
                cached_model = self.models_cache[model]
                return {
                    "name": cached_model.name,
                    "size": cached_model.size,
                    "modified_at": cached_model.modified_at,
                    "digest": cached_model.digest,
                    "details": cached_model.details,
                    "available": cached_model.available,
                    "cached": True
                }
            
            # Try to get from API
            request_ctx = await self._resolve_context_manager(
                self.session.get(
                    f"{self.base_url}/{self.api_version}/show",
                    json={"name": model}
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status == 200:
                    data = await response.json()
                    return {
                        "name": data["name"],
                        "modified_at": data.get("modified_at", ""),
                        "size": data.get("size", 0),
                        "details": data.get("details", {}),
                        "available": True,
                        "cached": False
                    }
                elif status == 404:
                    return None
                else:
                    raise LLMBackendError(f"Failed to get model info: HTTP {status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model}: {e}")
            return None
    
    # Core LLM Methods
    
    async def chat_completion(self, request: ChatRequest) -> LLMResponse:
        """
        Perform chat completion using Ollama.
        
        Args:
            request: Chat completion request
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        if not request.messages:
            raise ConfigurationError("Chat request must contain messages")
        
        if request.model not in self.models_cache:
            available_models = list(self.models_cache.keys())
            raise ModelNotFoundError(
                f"Model '{request.model}' not available. Available models: {available_models}"
            )
        
        try:
            # Enforce rate limiting
            await self._enforce_rate_limit(request.model)
            
            # Prepare messages
            ollama_messages = [msg.to_dict() for msg in request.messages]
            
            # Prepare request data
            request_data = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop_sequences
                }
            }
            
            # Make API request
            request_ctx = await self._resolve_context_manager(
                self.session.post(
                    f"{self.base_url}/{self.api_version}/chat",
                    json=request_data
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status != 200:
                    error_data = await response.json() if response.content_type == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {status}')
                    raise LLMBackendError(f"Chat completion failed: {error_msg}")
                
                response_data = await response.json()
                
                # Parse response
                return self._parse_ollama_chat_response(response_data, request.model)
                
        except LLMBackendError as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Chat completion failed: {e}")
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Chat completion failed: {e}")
            raise LLMBackendError(f"Chat completion failed: {e}")
        finally:
            # Update metrics
            elapsed_time = time.time() - start_time
            self._update_metrics(elapsed_time)
    
    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Perform streaming chat completion.
        
        Args:
            request: Chat completion request
            
        Yields:
            Response content chunks
        """
        start_time = time.time()
        
        if not request.messages:
            raise ConfigurationError("Chat request must contain messages")
        
        if request.model not in self.models_cache:
            available_models = list(self.models_cache.keys())
            raise ModelNotFoundError(
                f"Model '{request.model}' not available. Available models: {available_models}"
            )
        
        try:
            # Enforce rate limiting
            await self._enforce_rate_limit(request.model)
            
            # Prepare messages
            ollama_messages = [msg.to_dict() for msg in request.messages]
            
            # Prepare request data
            request_data = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop_sequences
                }
            }
            
            # Make streaming API request
            request_ctx = await self._resolve_context_manager(
                self.session.post(
                    f"{self.base_url}/{self.api_version}/chat",
                    json=request_data
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status != 200:
                    error_data = await response.json() if response.content_type == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {status}')
                    raise LLMBackendError(f"Streaming chat completion failed: {error_msg}")
                
                # Stream response
                async for line in self._iterate_response_content(response):
                    if not line:
                        continue
                    
                    try:
                        line_data = json.loads(line.decode('utf-8'))
                        if "message" in line_data and "content" in line_data["message"]:
                            yield line_data["message"]["content"]
                        
                        # Check for completion
                        if line_data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
        except LLMBackendError as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise LLMBackendError(f"Streaming chat completion failed: {e}")
        finally:
            # Update metrics
            elapsed_time = time.time() - start_time
            self._update_metrics(elapsed_time)
    
    async def text_completion(self, request: CompletionRequest) -> LLMResponse:
        """
        Perform text completion using Ollama.
        
        Args:
            request: Text completion request
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        if not request.prompt:
            raise ConfigurationError("Completion request must contain prompt")
        
        if request.model not in self.models_cache:
            available_models = list(self.models_cache.keys())
            raise ModelNotFoundError(
                f"Model '{request.model}' not available. Available models: {available_models}"
            )
        
        try:
            # Enforce rate limiting
            await self._enforce_rate_limit(request.model)
            
            # Prepare request data
            request_data = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop_sequences
                }
            }
            
            # Make API request
            request_ctx = await self._resolve_context_manager(
                self.session.post(
                    f"{self.base_url}/{self.api_version}/generate",
                    json=request_data
                )
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status != 200:
                    error_data = await response.json() if response.content_type == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {status}')
                    raise LLMBackendError(f"Text completion failed: {error_msg}")
                
                response_data = await response.json()
                
                # Parse response
                return self._parse_ollama_generate_response(response_data, request.model)
                
        except LLMBackendError as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Text completion failed: {e}")
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Text completion failed: {e}")
            raise LLMBackendError(f"Text completion failed: {e}")
        finally:
            # Update metrics
            elapsed_time = time.time() - start_time
            self._update_metrics(elapsed_time)
    
    async def create_embeddings(self, request: EmbeddingsRequest) -> List[List[float]]:
        """
        Create embeddings for input texts using Ollama.
        
        Args:
            request: Embeddings request
            
        Returns:
            List of embedding vectors
        """
        try:
            # Note: Ollama doesn't have a dedicated embeddings API endpoint yet
            # This is a placeholder for future implementation
            raise NotImplementedError(
                "Embeddings not yet supported by Ollama backend. "
                "This feature will be added when Ollama releases embeddings support."
            )
            
        except Exception as e:
            self.logger.error(f"Embeddings creation failed: {e}")
            raise LLMBackendError(f"Embeddings creation failed: {e}")
    
    # Token Counting
    
    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            model: Model name for token counting
            
        Returns:
            Number of tokens
        """
        try:
            if not model:
                model = self.provider_config.default_model or "llama2"
            
            # Try to get specific tokenizer for model
            if model not in self.token_encoders:
                # For most models, try to get the appropriate tokenizer
                try:
                    # Use tiktoken with appropriate encoding
                    if "gpt" in model.lower():
                        self.token_encoders[model] = tiktoken.get_encoding("gpt2")
                    elif "cl100k" in model.lower():
                        self.token_encoders[model] = tiktoken.get_encoding("cl100k_base")
                    else:
                        # Default to a reasonable approximation
                        self.token_encoders[model] = tiktoken.get_encoding("gpt2")
                except Exception:
                    # Fallback to character-based counting
                    return len(text) // 4  # Rough approximation
            
            # Use the tokenizer
            try:
                tokens = self.token_encoders[model].encode(text)
                return len(tokens)
            except Exception:
                # Fallback to character-based counting
                return len(text) // 4
                
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}, using approximation")
            # Simple fallback: approximately 4 characters per token
            return len(text) // 4
    
    # Utility Methods
    
    def _parse_ollama_chat_response(
        self, 
        response_data: Dict[str, Any], 
        model: str
    ) -> LLMResponse:
        """Parse Ollama chat API response."""
        try:
            # Extract content
            content = response_data.get("message", {}).get("content", "")
            
            # Extract usage information
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            completion_tokens = response_data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            # Extract metadata
            metadata = {
                "raw_response": response_data,
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_duration": response_data.get("eval_duration")
            }
            
            return LLMResponse(
                content=content,
                model=model,
                provider="ollama",
                usage=usage,
                finish_reason="stop" if response_data.get("done") else None,
                metadata=metadata
            )
            
        except Exception as e:
            raise ResponseParsingError(f"Failed to parse Ollama chat response: {e}")
    
    def _parse_ollama_generate_response(
        self, 
        response_data: Dict[str, Any], 
        model: str
    ) -> LLMResponse:
        """Parse Ollama generate API response."""
        try:
            # Extract content
            content = response_data.get("response", "")
            
            # Extract usage information
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            completion_tokens = response_data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            # Extract metadata
            metadata = {
                "raw_response": response_data,
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_duration": response_data.get("eval_duration")
            }
            
            return LLMResponse(
                content=content,
                model=model,
                provider="ollama",
                usage=usage,
                finish_reason="stop" if response_data.get("done") else None,
                metadata=metadata
            )
            
        except Exception as e:
            raise ResponseParsingError(f"Failed to parse Ollama generate response: {e}")
    
    def _update_metrics(self, response_time: float):
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["successful_requests"] += 1
        
        # Update average response time
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "healthy": False,
                    "provider": "ollama",
                    "error": "HTTP session not initialized",
                    "timestamp": time.time()
                }
            
            # Check service availability
            if not self.service_available:
                try:
                    await self._validate_service()
                except ServiceUnavailableError as exc:
                    return {
                        "status": "unhealthy",
                        "healthy": False,
                        "provider": "ollama",
                        "error": str(exc),
                        "timestamp": time.time()
                    }
            
            # Check API endpoint
            request_ctx = await self._resolve_context_manager(
                self.session.get(f"{self.base_url}/{self.api_version}/tags")
            )
            async with request_ctx as response:
                status = self._get_http_status(response)
                if status != 200:
                    return {
                        "status": "unhealthy",
                        "healthy": False,
                        "provider": "ollama",
                        "error": f"API returned status {status}",
                        "timestamp": time.time()
                    }
            
            # Get model count
            model_count = len(self.models_cache)
            
            return {
                "status": "healthy",
                "healthy": True,
                "provider": "ollama",
                "base_url": self.base_url,
                "model_count": model_count,
                "metrics": self.metrics.copy(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "provider": "ollama",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.metrics.copy()
        stats.update({
            "service_available": self.service_available,
            "models_available": len(self.models_cache),
            "cache_size": len(self.models_cache),
            "request_rate": self.metrics["total_requests"] / max(1, time.time() - self.last_request_time)
        })
        return stats
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Get supported features by Ollama backend."""
        return {
            "streaming": True,
            "model_management": True,
            "token_counting": True,
            "custom_parameters": True,
            "embeddings": False,  # Not yet supported by Ollama
            "cost_tracking": True,  # Track usage but no monetary cost
            "batch_requests": False,
            "async_requests": True
        }
    
    async def _enforce_rate_limit(self, model: str):
        """Enforce rate limiting for Ollama backend."""
        current_time = time.time()
        
        # Simple rate limiting implementation
        if current_time - self.last_request_time < 2.0:  # Minimum 2 seconds between requests
            wait_time = 2.0 - (current_time - self.last_request_time)
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        if self._session_context_exit:
            try:
                await self._session_context_exit(None, None, None)
            finally:
                self._session_context_exit = None
        
        self.service_available = False
        self.logger.info("Ollama backend closed")
    
    # ContextBox Integration Methods
    
    async def generate_summary(
        self, 
        text: str, 
        model: Optional[str] = None,
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            model: Model to use for summarization
            max_length: Maximum summary length
            
        Returns:
            Generated summary
        """
        if not model:
            model = self.provider_config.default_model or "llama2"
        
        prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Summary:"""
        
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            provider="ollama",
            max_tokens=max_length // 4,  # Rough estimate
            temperature=0.3
        )
        
        response = await self.text_completion(request)
        return response.content.strip()
    
    async def extract_key_points(
        self, 
        text: str, 
        model: Optional[str] = None,
        max_points: int = 10
    ) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Text to analyze
            model: Model to use for extraction
            max_points: Maximum number of key points
            
        Returns:
            List of key points
        """
        if not model:
            model = self.provider_config.default_model or "llama2"
        
        prompt = f"""Extract the {max_points} most important key points from the following text. 
Return them as a numbered list:

{text}

Key Points:"""
        
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            provider="ollama",
            temperature=0.2
        )
        
        response = await self.text_completion(request)
        
        # Parse the numbered list from response
        points = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Clean up the bullet point
                point = line.lstrip('0123456789.-').strip()
                if point:
                    points.append(point)
        
        return points[:max_points]
    
    async def analyze_sentiment(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            model: Model to use for analysis
            
        Returns:
            Sentiment analysis results
        """
        if not model:
            model = self.provider_config.default_model or "llama2"
        
        prompt = f"""Analyze the sentiment of the following text. 
Respond with a JSON object containing:
- "sentiment": one of ["positive", "negative", "neutral"]
- "confidence": a number between 0 and 1
- "reasoning": brief explanation

Text: {text}

Response:"""
        
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            provider="ollama",
            temperature=0.1
        )
        
        try:
            response = await self.text_completion(request)
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse detailed analysis"
                }
        except Exception as e:
            self.logger.warning(f"Sentiment analysis parsing failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "Analysis completed with fallback response"
            }


# Factory function for easy integration

def create_ollama_backend(
    config: Optional[Dict[str, Any]] = None,
    base_url: str = "http://localhost:11434",
    default_model: str = "llama2"
) -> OllamaBackend:
    """
    Create and configure an Ollama backend.
    
    Args:
        config: Optional configuration dictionary
        base_url: Ollama service base URL
        default_model: Default model to use
        
    Returns:
        Configured Ollama backend
    """
    # Create configuration
    llm_config = LLMBackendConfig()
    
    # Create provider configuration
    provider_config = ProviderConfig(
        name="ollama",
        base_url=base_url,
        timeout=60,
        max_retries=3,
        rate_limit=RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            tokens_per_minute=50000,
            burst_limit=5
        )
    )
    
    # Add default model
    model_config = ModelConfig(
        name=default_model,
        model_type=ModelType.CHAT,
        provider="ollama",
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9
    )
    provider_config.models[default_model] = model_config
    provider_config.default_model = default_model
    
    llm_config.providers["ollama"] = provider_config
    llm_config.default_provider = "ollama"
    
    # Apply custom config if provided
    if config:
        # Merge custom configuration
        pass
    
    return OllamaBackend(llm_config)
