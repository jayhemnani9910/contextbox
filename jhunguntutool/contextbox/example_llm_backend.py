"""
Example usage of the ContextBox LLM Backend Architecture.

This script demonstrates how to use the pluggable LLM backend architecture
to interact with different LLM providers.
"""

import asyncio
import os
import json
from typing import Dict, Any

# Import the LLM backend components
from contextbox.llm import (
    LLMBackendConfig,
    ConfigManager,
    BaseLLMBackend,
    LLMResponse,
    ChatRequest,
    CompletionRequest,
    EmbeddingsRequest,
    create_chat_message,
    TokenUsage,
    CostInfo,
    ModelType
)
from contextbox.llm.config import ProviderConfig, ModelConfig
from contextbox.llm.utils import (
    SimpleTokenCounter, 
    SlidingWindowRateLimiter, 
    SimpleCostTracker
)


class ExampleLLMBackend(BaseLLMBackend):
    """Example implementation of a simple LLM backend."""
    
    def __init__(self, config: LLMBackendConfig, provider_name: str):
        super().__init__(config, provider_name)
        self.api_responses = {
            "test": "Hello! I'm an example LLM backend response.",
            "help": "I'd be happy to help you with that question.",
            "what": "That's an interesting question about what you're asking."
        }
    
    async def _do_initialize(self):
        """Initialize the backend."""
        # In a real implementation, this would test the API connection
        print(f"Initializing {self.provider_name} backend...")
        await asyncio.sleep(0.1)  # Simulate initialization delay
    
    async def chat_completion(self, request: ChatRequest) -> LLMResponse:
        """Perform chat completion (simplified example)."""
        # Extract user message
        user_message = None
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise ValueError("No user message found")
        
        # Generate response based on message content
        response_text = ""
        for keyword, response in self.api_responses.items():
            if keyword in user_message.lower():
                response_text = response
                break
        
        if not response_text:
            response_text = f"I received your message: '{user_message}'. This is an example response."
        
        # Create token usage
        usage = TokenUsage(
            prompt_tokens=len(user_message.split()) * 1.3,  # Rough estimate
            completion_tokens=len(response_text.split()) * 1.3,
            total_tokens=0
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        
        # Create cost info
        cost_info = CostInfo(
            input_cost=usage.prompt_tokens * 0.001,
            output_cost=usage.completion_tokens * 0.002,
            total_cost=usage.prompt_tokens * 0.001 + usage.completion_tokens * 0.002
        )
        
        # Track usage
        self._track_usage(request.model, usage, cost_info)
        
        # Create response
        return LLMResponse(
            content=response_text,
            model=request.model,
            provider=self.provider_name,
            usage=usage,
            cost=cost_info,
            finish_reason="stop"
        )
    
    async def text_completion(self, request: CompletionRequest) -> LLMResponse:
        """Perform text completion (simplified example)."""
        prompt = request.prompt
        
        # Generate response
        response_text = f"Completion for: {prompt[:50]}..."
        
        # Create token usage
        usage = TokenUsage(
            prompt_tokens=len(prompt.split()) * 1.3,
            completion_tokens=len(response_text.split()) * 1.3,
            total_tokens=0
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        
        # Create cost info
        cost_info = CostInfo(
            input_cost=usage.prompt_tokens * 0.001,
            output_cost=usage.completion_tokens * 0.002,
            total_cost=usage.prompt_tokens * 0.001 + usage.completion_tokens * 0.002
        )
        
        # Track usage
        self._track_usage(request.model, usage, cost_info)
        
        return LLMResponse(
            content=response_text,
            model=request.model,
            provider=self.provider_name,
            usage=usage,
            cost=cost_info,
            finish_reason="stop"
        )
    
    async def create_embeddings(self, request: EmbeddingsRequest) -> list:
        """Create embeddings (simplified example)."""
        embeddings = []
        for text in request.input_texts:
            # Generate fake embeddings (768 dimensions)
            import random
            embedding = [random.uniform(-1, 1) for _ in range(768)]
            embeddings.append(embedding)
        
        return embeddings


async def example_basic_usage():
    """Example of basic LLM backend usage."""
    print("=== Basic LLM Backend Usage Example ===")
    
    # Create custom configuration for the example
    config = LLMBackendConfig()
    
    # Add example provider
    provider_config = ProviderConfig(
        name="example",
        api_key="example_key",
        base_url="https://api.example.com"
    )
    
    # Add example model
    model_config = ModelConfig(
        name="example-model",
        model_type=ModelType.CHAT,
        provider="example",
        max_tokens=4096,
        temperature=0.7,
        cost_per_input_token=0.001,
        cost_per_output_token=0.002
    )
    model_config.validate()
    provider_config.models["example-model"] = model_config
    provider_config.default_model = "example-model"
    
    config.providers["example"] = provider_config
    config.default_provider = "example"
    
    # Create backend
    backend = ExampleLLMBackend(config, "example")
    
    # Initialize backend
    await backend.initialize()
    
    # Create chat request
    request = ChatRequest.from_text(
        "Hello, can you help me with something?",
        model="example-model",
        provider="example",
        system_prompt="You are a helpful assistant."
    )
    
    # Make request
    response = await backend.chat_completion(request)
    
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Token usage: {response.usage.total_tokens} tokens")
    if response.cost:
        print(f"Cost: ${response.cost.total_cost:.4f}")
    
    await backend.close()


async def example_configuration():
    """Example of configuring multiple providers."""
    print("\n=== Configuration Example ===")
    
    # Create custom configuration
    config = LLMBackendConfig()
    
    # Add a provider
    provider_config = ProviderConfig(
        name="custom_provider",
        api_key="test_key",
        base_url="https://api.example.com",
        default_model="test-model"
    )
    
    # Add a model
    model_config = ModelConfig(
        name="test-model",
        model_type=ModelType.CHAT,
        provider="custom_provider",
        max_tokens=4096,
        temperature=0.7,
        cost_per_input_token=0.001,
        cost_per_output_token=0.002
    )
    model_config.validate()
    provider_config.models["test-model"] = model_config
    
    config.providers["custom_provider"] = provider_config
    config.default_provider = "custom_provider"
    
    # Validate configuration
    config_manager = ConfigManager()
    config_manager.validate_config(config)
    
    print("Configuration created successfully!")
    print(f"Default provider: {config.default_provider}")
    print(f"Models available: {list(provider_config.models.keys())}")


async def example_token_counting():
    """Example of token counting."""
    print("\n=== Token Counting Example ===")
    
    # Create token counter
    counter = SimpleTokenCounter()
    
    # Count tokens in text
    text = "This is a test sentence for token counting."
    token_count = counter.count_tokens(text, "test-model")
    print(f"Text: '{text}'")
    print(f"Estimated tokens: {token_count}")
    
    # Count tokens in chat messages
    messages = [
        create_chat_message("system", "You are helpful."),
        create_chat_message("user", "Hello, how are you?"),
        create_chat_message("assistant", "I'm doing well, thank you!")
    ]
    
    message_tokens = counter.count_messages(messages, "test-model")
    print(f"Messages token count: {message_tokens}")


async def example_rate_limiting():
    """Example of rate limiting."""
    print("\n=== Rate Limiting Example ===")
    
    from contextbox.llm.config import RateLimitConfig
    from contextbox.llm.utils import SlidingWindowRateLimiter
    
    # Create rate limiter
    rate_config = RateLimitConfig(
        requests_per_minute=2,  # Very low for demo
        tokens_per_minute=1000,
        burst_limit=1
    )
    rate_limiter = SlidingWindowRateLimiter(rate_config)
    
    print("Testing rate limiting with 2 requests per minute...")
    
    # Try to make requests
    for i in range(3):
        try:
            await rate_limiter.acquire("test-model", 100)
            print(f"Request {i+1}: Allowed")
        except Exception as e:
            print(f"Request {i+1}: Rate limited - {e}")
    
    # Check rate limit status
    status = rate_limiter.get_rate_limit_status("test-model")
    print(f"Rate limit status: {status}")


async def example_cost_tracking():
    """Example of cost tracking."""
    print("\n=== Cost Tracking Example ===")
    
    from contextbox.llm.config import CostConfig
    from contextbox.llm.utils import SimpleCostTracker
    
    # Create cost tracker
    cost_config = CostConfig(
        track_usage=True,
        track_costs=True,
        budget_limit=10.0
    )
    cost_tracker = SimpleCostTracker(cost_config)
    
    # Simulate some requests
    usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    cost1 = CostInfo(input_cost=0.01, output_cost=0.02, total_cost=0.03)
    
    usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)
    cost2 = CostInfo(input_cost=0.015, output_cost=0.025, total_cost=0.04)
    
    # Track requests
    await cost_tracker.track_request("model1", usage1, cost1)
    await cost_tracker.track_request("model1", usage2, cost2)
    
    print(f"Total cost: ${cost_tracker.get_total_cost():.4f}")
    
    stats = cost_tracker.get_usage_stats()
    print(f"Usage stats: {stats}")


async def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    from contextbox.llm.exceptions import ModelNotFoundError, RateLimitError
    
    try:
        # Simulate model not found error
        raise ModelNotFoundError("Model 'nonexistent' not found", model="nonexistent")
    except ModelNotFoundError as e:
        print(f"Caught ModelNotFoundError: {e.message}")
        print(f"Model: {e.model}")
    
    try:
        # Simulate rate limit error
        raise RateLimitError("Rate limit exceeded", retry_after=60)
    except RateLimitError as e:
        print(f"Caught RateLimitError: {e.message}")
        print(f"Retry after: {e.retry_after} seconds")


async def main():
    """Run all examples."""
    print("ContextBox LLM Backend Architecture Examples")
    print("=" * 50)
    
    await example_basic_usage()
    await example_configuration()
    await example_token_counting()
    await example_rate_limiting()
    await example_cost_tracking()
    await example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())