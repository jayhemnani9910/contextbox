"""
GitHub Models LLM Backend for ContextBox.

Uses GitHub's free AI models API (Azure OpenAI compatible endpoint).
Supports: GPT-4o, GPT-4o-mini, Llama, Mistral, and more.
"""

import os
import logging
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from .base import (
    BaseLLMBackend,
    LLMResponse,
    ChatMessage,
    ChatRequest,
    CompletionRequest,
    EmbeddingsRequest,
    TokenUsage,
    CostInfo,
)
from .config import LLMBackendConfig, ModelConfig
from .exceptions import (
    LLMBackendError,
    AuthenticationError,
    ModelNotFoundError,
)


# GitHub Models endpoint
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Available models on GitHub Models (free tier)
GITHUB_MODELS = {
    "gpt-4o": {
        "name": "gpt-4o",
        "context_length": 128000,
        "description": "Most capable GPT-4 model",
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "context_length": 128000,
        "description": "Fast and efficient GPT-4 variant",
    },
    "meta-llama-3.1-405b-instruct": {
        "name": "meta-llama-3.1-405b-instruct",
        "context_length": 128000,
        "description": "Meta's largest Llama model",
    },
    "meta-llama-3.1-70b-instruct": {
        "name": "meta-llama-3.1-70b-instruct",
        "context_length": 128000,
        "description": "Meta's Llama 70B model",
    },
    "mistral-large": {
        "name": "mistral-large",
        "context_length": 32000,
        "description": "Mistral's flagship model",
    },
}

DEFAULT_MODEL = "gpt-4o-mini"


class GitHubModelsBackend(BaseLLMBackend):
    """
    LLM backend for GitHub Models.

    Uses GitHub token for authentication and the Azure OpenAI-compatible endpoint.
    Free tier available with GitHub account.
    """

    def __init__(
        self,
        config: Optional[LLMBackendConfig] = None,
        github_token: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize GitHub Models backend.

        Args:
            config: LLM backend configuration
            github_token: GitHub personal access token (or set GITHUB_TOKEN env var)
            default_model: Default model to use
            logger: Optional logger instance
        """
        if AsyncOpenAI is None:
            raise LLMBackendError(
                "openai package required. Install with: pip install openai"
            )

        # Get token from parameter, env var, or config
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

        if not self.github_token:
            raise AuthenticationError(
                "GitHub token required. Set GITHUB_TOKEN environment variable "
                "or pass github_token parameter."
            )

        self.default_model = default_model
        self._client: Optional[AsyncOpenAI] = None
        self._logger = logger or logging.getLogger(__name__)

        # Create minimal config if none provided
        if config is None:
            from .config import LLMBackendConfig, ProviderConfig
            config = LLMBackendConfig(
                providers={
                    "github_models": ProviderConfig(
                        provider_type="github_models",
                        enabled=True,
                        default_model=default_model,
                    )
                }
            )

        super().__init__(config, "github_models", self._logger)

    async def _do_initialize(self):
        """Initialize the OpenAI client for GitHub Models."""
        self._client = AsyncOpenAI(
            base_url=GITHUB_MODELS_ENDPOINT,
            api_key=self.github_token,
        )
        self._logger.info("GitHub Models backend initialized")

    async def chat_completion(self, request: ChatRequest) -> LLMResponse:
        """
        Perform chat completion using GitHub Models.

        Args:
            request: Chat completion request

        Returns:
            LLM response with generated content
        """
        if not self._is_initialized:
            await self.initialize()

        model = request.model or self.default_model

        if model not in GITHUB_MODELS:
            self._logger.warning(
                f"Model '{model}' not in known GitHub Models list. "
                f"Available: {list(GITHUB_MODELS.keys())}"
            )

        try:
            # Convert messages to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]

            # Make API call
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences,
            )

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""

            # Create usage info
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            # GitHub Models are free, but track for monitoring
            cost = CostInfo(
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                currency="USD",
            )

            return LLMResponse(
                content=content,
                model=model,
                provider="github_models",
                usage=usage,
                cost=cost,
                finish_reason=choice.finish_reason,
                metadata={
                    "response_id": response.id,
                    "created": response.created,
                },
            )

        except Exception as e:
            self._logger.error(f"GitHub Models chat completion failed: {e}")
            raise LLMBackendError(f"Chat completion failed: {e}")

    async def text_completion(self, request: CompletionRequest) -> LLMResponse:
        """
        Perform text completion using GitHub Models.

        Note: Converts to chat format since GitHub Models uses chat API.
        """
        # Convert to chat request
        chat_request = ChatRequest.from_text(
            text=request.prompt,
            model=request.model or self.default_model,
            provider="github_models",
        )
        chat_request.max_tokens = request.max_tokens
        chat_request.temperature = request.temperature
        chat_request.top_p = request.top_p
        chat_request.stop_sequences = request.stop_sequences

        return await self.chat_completion(chat_request)

    async def create_embeddings(self, request: EmbeddingsRequest) -> List[List[float]]:
        """
        Create embeddings using GitHub Models.

        Note: Uses text-embedding-3-small model by default.
        """
        if not self._is_initialized:
            await self.initialize()

        model = request.model or "text-embedding-3-small"

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=request.input_texts,
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            self._logger.error(f"GitHub Models embeddings failed: {e}")
            raise LLMBackendError(f"Embeddings creation failed: {e}")

    async def stream_chat_completion(
        self,
        request: ChatRequest
    ) -> AsyncIterator[str]:
        """Stream chat completion responses."""
        if not self._is_initialized:
            await self.initialize()

        model = request.model or self.default_model
        messages = [msg.to_dict() for msg in request.messages]

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._logger.error(f"GitHub Models streaming failed: {e}")
            raise LLMBackendError(f"Streaming failed: {e}")

    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.close()
        await super().close()

    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        """List available models on GitHub Models."""
        return GITHUB_MODELS.copy()

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return GITHUB_MODELS.get(model_name)


def create_github_models_backend(
    github_token: Optional[str] = None,
    default_model: str = DEFAULT_MODEL,
    **kwargs,
) -> GitHubModelsBackend:
    """
    Factory function to create a GitHub Models backend.

    Args:
        github_token: GitHub personal access token
        default_model: Default model to use
        **kwargs: Additional arguments passed to backend

    Returns:
        Configured GitHubModelsBackend instance
    """
    return GitHubModelsBackend(
        github_token=github_token,
        default_model=default_model,
        **kwargs,
    )


# Quick helper for simple use cases
async def quick_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    github_token: Optional[str] = None,
) -> str:
    """
    Quick helper for simple chat completions.

    Args:
        prompt: User prompt
        model: Model to use
        system_prompt: Optional system prompt
        github_token: GitHub token (or set GITHUB_TOKEN env var)

    Returns:
        Generated text response
    """
    backend = create_github_models_backend(
        github_token=github_token,
        default_model=model,
    )

    async with backend:
        request = ChatRequest.from_text(
            text=prompt,
            model=model,
            provider="github_models",
            system_prompt=system_prompt,
        )
        response = await backend.chat_completion(request)
        return response.content
