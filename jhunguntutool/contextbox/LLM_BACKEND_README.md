# ContextBox LLM Backend Architecture

A comprehensive, pluggable LLM backend architecture designed for ContextBox that supports multiple providers with consistent interfaces and comprehensive functionality.

## Features

### Core Capabilities
- **Pluggable Architecture**: Support for multiple LLM providers through a unified interface
- **Async/Sync Support**: Both synchronous and asynchronous operation support
- **Configuration Management**: Flexible JSON-based configuration system
- **Token Counting**: Built-in token counting with model-specific implementations
- **Rate Limiting**: Multiple rate limiting strategies (sliding window, leaky bucket)
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Cost Tracking**: Built-in cost tracking and budget management
- **Usage Analytics**: Detailed usage statistics and monitoring
- **Response Parsing**: Standardized response format across all providers
- **Logging**: Comprehensive logging and monitoring capabilities

### Supported Operations
- **Chat Completions**: Interactive conversational AI
- **Text Completions**: Traditional text generation
- **Embeddings**: Vector embeddings for semantic search and analysis

### Model Support
- **Chat Models**: GPT, Claude, and other chat-focused models
- **Completion Models**: Text generation and completion models  
- **Embedding Models**: Text embedding and vector generation

## Architecture Overview

```
contextbox/llm/
├── __init__.py              # Package initialization
├── base.py                  # Abstract base class and core interfaces
├── config.py                # Configuration management system
├── exceptions.py            # Custom exception classes
├── utils.py                 # Utility implementations
└── llm_config_example.json  # Example configuration file
```

## Quick Start

### 1. Basic Setup

```python
from contextbox.llm import (
    ConfigManager, BaseLLMBackend, ChatRequest, create_chat_message
)

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Initialize your backend (implement BaseLLMBackend)
backend = YourLLMBackend(config, provider_name="your_provider")
await backend.initialize()
```

### 2. Chat Completion

```python
# Create chat request
request = ChatRequest.from_text(
    "Hello, how are you?",
    model="gpt-3.5-turbo", 
    provider="openai",
    system_prompt="You are a helpful assistant."
)

# Make request
response = await backend.chat_completion(request)

print(f"Response: {response.content}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Cost: ${response.cost.total_cost:.4f}")
```

### 3. Configuration

Create a `llm_config.json` file:

```json
{
  "default_provider": "openai",
  "enable_logging": true,
  "log_level": "INFO",
  "providers": {
    "openai": {
      "api_key": "your_openai_api_key",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-3.5-turbo",
      "models": {
        "gpt-3.5-turbo": {
          "model_type": "chat",
          "max_tokens": 4096,
          "temperature": 0.7,
          "cost_per_input_token": 0.0015,
          "cost_per_output_token": 0.002
        }
      }
    }
  }
}
```

## Implementation Guide

### Creating a Custom Backend

1. **Extend the Base Class**:

```python
from contextbox.llm import BaseLLMBackend, ChatRequest, LLMResponse

class YourLLMBackend(BaseLLMBackend):
    async def _do_initialize(self):
        # Initialize your provider connection
        pass
    
    async def chat_completion(self, request: ChatRequest) -> LLMResponse:
        # Implement chat completion
        pass
    
    async def text_completion(self, request) -> LLMResponse:
        # Implement text completion  
        pass
    
    async def create_embeddings(self, request) -> list:
        # Implement embeddings
        pass
```

2. **Implement Core Methods**:

- `_do_initialize()`: Initialize provider connection
- `chat_completion()`: Handle chat completion requests
- `text_completion()`: Handle text completion requests  
- `create_embeddings()`: Handle embedding requests
- `_extract_content()`: Parse provider-specific responses

3. **Handle Provider-Specific Details**:

```python
def _extract_content(self, response_data: Dict[str, Any]) -> str:
    # Extract content from your provider's response format
    return response_data["choices"][0]["message"]["content"]
```

### Configuration System

#### Provider Configuration

```python
from contextbox.llm.config import ProviderConfig, ModelConfig, ModelType

# Create provider
provider = ProviderConfig(
    name="your_provider",
    api_key="your_api_key",
    base_url="https://api.yourprovider.com",
    default_model="your-model",
    timeout=30,
    max_retries=3
)

# Add models
model = ModelConfig(
    name="your-model",
    model_type=ModelType.CHAT,
    provider="your_provider",
    max_tokens=4096,
    temperature=0.7,
    cost_per_input_token=0.001,
    cost_per_output_token=0.002
)

provider.models["your-model"] = model
```

#### Rate Limiting

```python
from contextbox.llm.config import RateLimitConfig
from contextbox.llm.utils import SlidingWindowRateLimiter

# Configure rate limits
rate_config = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000, 
    tokens_per_minute=100000,
    burst_limit=10
)

# Create rate limiter
rate_limiter = SlidingWindowRateLimiter(rate_config)
backend.set_rate_limiter(rate_limiter)
```

#### Cost Tracking

```python
from contextbox.llm.config import CostConfig
from contextbox.llm.utils import SimpleCostTracker

# Configure cost tracking
cost_config = CostConfig(
    track_usage=True,
    track_costs=True,
    budget_limit=100.0,
    alert_threshold=0.8
)

# Create cost tracker  
cost_tracker = SimpleCostTracker(cost_config)
backend.set_cost_tracker(cost_tracker)
```

### Utility Components

#### Token Counting

```python
from contextbox.llm.utils import SimpleTokenCounter

counter = SimpleTokenCounter()
tokens = counter.count_tokens("Your text here", "model-name")
```

#### Retry Logic

```python
from contextbox.llm.utils import with_retry, RetryConfig

retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    exponential_base=2.0,
    jitter=True
)

result = await with_retry(
    your_function,
    retry_config=retry_config,
    exceptions=(Exception,)
)
```

## Error Handling

### Exception Hierarchy

```python
from contextbox.llm.exceptions import (
    LLMBackendError,           # Base exception
    ConfigurationError,        # Configuration issues
    AuthenticationError,       # Authentication failures
    RateLimitError,           # Rate limit exceeded
    TokenLimitError,          # Token limit exceeded
    ModelNotFoundError,       # Model not available
    ServiceUnavailableError,  # Service down
    ResponseParsingError      # Response parsing issues
)
```

### Error Handling Example

```python
try:
    response = await backend.chat_completion(request)
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ModelNotFoundError as e:
    print(f"Model {e.model} not available")
except LLMBackendError as e:
    print(f"LLM Error: {e.message}")
```

## Monitoring and Analytics

### Usage Statistics

```python
# Get usage stats
stats = backend.get_usage_stats()
print(f"Total requests: {stats['model_name']['requests']}")
print(f"Total tokens: {stats['model_name']['total_tokens']}")
print(f"Total cost: ${stats['model_name']['total_cost']:.4f}")
```

### Health Checks

```python
# Perform health check
health = await backend.health_check()
if health["healthy"]:
    print("Backend is healthy")
else:
    print(f"Backend unhealthy: {health['error']}")
```

### Cost Tracking

```python
# Get cost information
cost_info = response.cost
if cost_info:
    print(f"Input cost: ${cost_info.input_cost:.4f}")
    print(f"Output cost: ${cost_info.output_cost:.4f}")
    print(f"Total cost: ${cost_info.total_cost:.4f}")

# Get total costs
total_cost = cost_tracker.get_total_cost()
print(f"Total spending: ${total_cost:.4f}")
```

## Best Practices

### 1. Configuration Management
- Use environment variables for API keys
- Validate configuration on startup
- Set reasonable rate limits and timeouts
- Configure cost tracking for budget management

### 2. Error Handling
- Implement proper retry logic for transient errors
- Handle rate limits gracefully
- Validate responses before processing
- Log errors with sufficient context

### 3. Performance
- Use async operations where possible
- Implement proper rate limiting
- Cache results when appropriate
- Monitor token usage and costs

### 4. Security
- Never log API keys or sensitive data
- Use environment variables for configuration
- Validate input data
- Implement proper authentication

## Examples

See `example_llm_backend.py` for comprehensive usage examples including:
- Basic backend usage
- Configuration management
- Token counting
- Rate limiting
- Cost tracking
- Error handling

## Testing

The architecture includes comprehensive test coverage for:
- Configuration validation
- Rate limiting behavior
- Cost tracking accuracy
- Error handling
- Response parsing

Run tests with:
```bash
python -m pytest tests/llm/
```

## Contributing

1. Follow the existing architecture patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility
5. Add type hints for all public APIs

## License

This LLM backend architecture is part of the ContextBox project and follows the same licensing terms.