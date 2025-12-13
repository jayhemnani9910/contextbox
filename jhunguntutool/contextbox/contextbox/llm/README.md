# ContextBox Ollama Integration

The ContextBox Ollama integration provides comprehensive support for local LLM models through Ollama's REST API. This implementation enables users to run and interact with various language models locally, providing privacy, cost savings, and full control over their AI workflows.

## Features

### Core Functionality
- **Chat Completion**: Generate responses to conversational prompts
- **Text Completion**: Complete partial text with context-aware generation
- **Streaming Responses**: Real-time streaming of generated text for interactive applications
- **Model Management**: List, pull, delete, and inspect local models
- **Token Counting**: Accurate token counting for cost estimation and context management
- **Performance Monitoring**: Track usage metrics, response times, and system health

### Advanced Features
- **Multi-Model Support**: Works with various model formats (Llama, Mistral, CodeLlama, etc.)
- **Custom Parameters**: Fine-tune generation with temperature, top_p, max_tokens, etc.
- **Error Handling**: Comprehensive error handling for service unavailability, model issues, etc.
- **ContextBox Integration**: Built-in methods for content analysis, summarization, and sentiment analysis
- **Configuration Management**: Flexible configuration system with validation and defaults
- **Asynchronous Operations**: Full async/await support for high-performance applications

### Supported Model Types
- **Chat Models**: Conversational AI with message history support
- **Code Models**: Programming-focused models with enhanced code generation
- **General Models**: Multi-purpose models for various text generation tasks

## Installation

### Prerequisites
1. **Ollama Installation**: Install Ollama following [official documentation](https://ollama.ai)
2. **Python Dependencies**: Install required packages

```bash
# Install ContextBox with LLM dependencies
pip install -e /path/to/contextbox[llm]

# Or install dependencies manually
pip install aiohttp tiktoken
```

### Quick Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (e.g., Llama 2)
ollama pull llama2

# Test the integration
python -m contextbox.llm.examples.ollama_examples
```

## Basic Usage

### Simple Chat Completion

```python
import asyncio
from contextbox.llm import create_ollama_backend, ChatRequest, create_chat_message

async def basic_chat():
    # Create backend
    backend = create_ollama_backend(
        base_url="http://localhost:11434",
        default_model="llama2"
    )
    
    await backend.initialize()
    
    # Create chat request
    messages = [
        create_chat_message("system", "You are a helpful assistant."),
        create_chat_message("user", "Explain quantum computing in simple terms.")
    ]
    
    request = ChatRequest(
        messages=messages,
        model="llama2",
        provider="ollama"
    )
    
    # Get response
    response = await backend.chat_completion(request)
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    
    await backend.close()

asyncio.run(basic_chat())
```

### Streaming Chat

```python
async def streaming_chat():
    backend = create_ollama_backend()
    await backend.initialize()
    
    messages = [create_chat_message("user", "Tell me a story about AI.")]
    request = ChatRequest(
        messages=messages,
        model="llama2",
        provider="ollama",
        stream=True
    )
    
    # Stream response
    async for chunk in backend.stream_chat_completion(request):
        print(chunk, end="", flush=True)
    
    await backend.close()
```

### Model Management

```python
async def manage_models():
    backend = create_ollama_backend()
    await backend.initialize()
    
    # List available models
    models = await backend.list_models()
    for model in models:
        print(f"Model: {model['name']}, Size: {model['size']:,} bytes")
    
    # Pull a new model
    result = await backend.pull_model("mistral:latest")
    print(f"Pull status: {result['status']}")
    
    # Get model information
    info = await backend.get_model_info("mistral:latest")
    print(f"Model details: {info}")
    
    await backend.close()
```

## ContextBox Integration Features

The Ollama backend includes specialized methods for ContextBox integration:

### Content Summarization

```python
async def summarize_content():
    backend = create_ollama_backend()
    await backend.initialize()
    
    text = "Long document content to summarize..."
    summary = await backend.generate_summary(text, max_length=200)
    print(f"Summary: {summary}")
    
    await backend.close()
```

### Key Points Extraction

```python
async def extract_key_points():
    backend = create_ollama_backend()
    await backend.initialize()
    
    text = "Document with multiple key points..."
    points = await backend.extract_key_points(text, max_points=5)
    
    for i, point in enumerate(points, 1):
        print(f"{i}. {point}")
    
    await backend.close()
```

### Sentiment Analysis

```python
async def analyze_sentiment():
    backend = create_ollama_backend()
    await backend.initialize()
    
    text = "This product is absolutely amazing and wonderful!"
    sentiment = await backend.analyze_sentiment(text)
    
    print(f"Sentiment: {sentiment['sentiment']}")
    print(f"Confidence: {sentiment['confidence']:.2f}")
    print(f"Reasoning: {sentiment['reasoning']}")
    
    await backend.close()
```

## Configuration

### Basic Configuration

```python
from contextbox.llm import LLMBackendConfig, ProviderConfig, ModelConfig, ModelType

# Create configuration
config = LLMBackendConfig(
    default_provider="ollama",
    enable_logging=True,
    log_level="INFO"
)

# Configure Ollama provider
provider_config = ProviderConfig(
    name="ollama",
    base_url="http://localhost:11434",
    timeout=60,
    max_retries=3
)

# Add models
provider_config.models["llama2"] = ModelConfig(
    name="llama2",
    model_type=ModelType.CHAT,
    provider="ollama",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9
)

provider_config.default_model = "llama2"
config.providers["ollama"] = provider_config

# Create backend with custom config
backend = OllamaBackend(config)
```

### Advanced Configuration

```python
# Custom rate limiting
from contextbox.llm.config import RateLimitConfig, CostConfig

provider_config.rate_limit = RateLimitConfig(
    requests_per_minute=30,      # Limit requests per minute
    requests_per_hour=500,       # Limit requests per hour
    tokens_per_minute=50000,     # Limit tokens per minute
    burst_limit=5                # Allow bursts
)

# Cost tracking (for usage analytics, not monetary)
provider_config.cost_config = CostConfig(
    track_usage=True,            # Track token usage
    track_costs=False,           # No monetary cost for local inference
    budget_limit=None            # No budget limit
)

# Model-specific settings
llama_config = ModelConfig(
    name="llama2",
    model_type=ModelType.CHAT,
    provider="ollama",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop_sequences=["Human:", "###"]  # Stop generation at these sequences
)
```

## Error Handling

The integration provides comprehensive error handling:

```python
try:
    response = await backend.chat_completion(request)
except ServiceUnavailableError:
    print("Ollama service is not available. Check if Ollama is running.")
except ModelNotFoundError as e:
    print(f"Model not found: {e}. Available models: {available_models}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except LLMBackendError as e:
    print(f"General LLM backend error: {e}")
```

## Performance Optimization

### Connection Management
- **Session Reuse**: HTTP sessions are reused across requests
- **Connection Pooling**: Efficient connection management with aiohttp
- **Timeout Configuration**: Configurable timeouts for different operations

### Rate Limiting
- **Built-in Rate Limiting**: Prevents overwhelming the local service
- **Configurable Limits**: Adjustable based on hardware capabilities
- **Burst Handling**: Allows short bursts while maintaining overall limits

### Caching
- **Model Cache**: Caches model information to reduce API calls
- **Configurable TTL**: Cache expiration for fresh data
- **Automatic Refresh**: Cache updates when models change

## Health Monitoring

```python
async def monitor_health():
    backend = create_ollama_backend()
    await backend.initialize()
    
    # Check overall health
    health = await backend.health_check()
    print(f"Status: {health['status']}")
    print(f"Healthy: {health['healthy']}")
    
    # Get performance statistics
    stats = await backend.get_performance_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Average response time: {stats['avg_response_time']:.2f}s")
    print(f"Success rate: {stats['successful_requests']}/{stats['total_requests']}")
    
    await backend.close()
```

## Supported Models

### Popular Models
- **Llama 2**: Meta's general-purpose language model
- **Code Llama**: Code-focused variant for programming tasks
- **Mistral**: Efficient and capable general-purpose model
- **Mixtral**: Mixture of experts model with strong performance
- **Phi**: Microsoft's small but capable models
- **Qwen**: Alibaba's multilingual models
- **Gemma**: Google's open models

### Model Requirements
- **Format**: Models must be in Ollama-compatible format (.gguf)
- **Quantization**: Various quantization levels supported (4-bit, 8-bit, etc.)
- **Memory**: Requirements vary by model size (2GB - 100GB+)

### Adding Custom Models

```python
# Pull a custom model
await backend.pull_model("username/custom-model:latest")

# Use in chat completion
messages = [create_chat_message("user", "Hello!")]
request = ChatRequest(
    messages=messages,
    model="username/custom-model:latest",
    provider="ollama"
)
response = await backend.chat_completion(request)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest contextbox/llm/tests/test_ollama_integration.py -v

# Run with coverage
python -m pytest contextbox/llm/tests/test_ollama_integration.py --cov=contextbox.llm.ollama

# Run specific test class
python -m pytest contextbox/llm/tests/test_ollama_integration.py::TestOllamaBackend -v
```

## Examples

The `contextbox/llm/examples/` directory contains comprehensive examples:

- `basic_chat_example()`: Simple chat completion
- `streaming_chat_example()`: Streaming responses
- `model_management_example()`: Managing models
- `token_counting_example()`: Token counting features
- `performance_monitoring_example()`: Performance tracking
- `contextbox_integration_example()`: ContextBox-specific features
- `custom_configuration_example()`: Advanced configuration
- `error_handling_example()`: Error handling patterns

## Troubleshooting

### Common Issues

**Ollama Service Not Running**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

**Model Not Found**
```python
# List available models first
models = await backend.list_models()
print([m['name'] for m in models])

# Then use an available model
request = ChatRequest(..., model=models[0]['name'])
```

**Out of Memory**
- Use smaller models (7B instead of 70B)
- Enable quantization (4-bit instead of 16-bit)
- Close other applications to free RAM

**Slow Responses**
- Check system resources (CPU, RAM)
- Use more efficient models
- Adjust generation parameters (lower max_tokens)

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

backend = create_ollama_backend()
await backend.initialize()
```

## Architecture

### Component Structure
```
contextbox/llm/
├── __init__.py              # Package exports and convenience functions
├── base.py                  # Base classes and interfaces
├── config.py                # Configuration management
├── exceptions.py            # Custom exception classes
├── ollama.py               # Ollama backend implementation
├── examples/               # Usage examples
│   ├── __init__.py
│   └── ollama_examples.py
└── tests/                  # Test suite
    ├── __init__.py
    └── test_ollama_integration.py
```

### Integration Points
- **ContextBox Core**: Seamless integration with ContextBox main application
- **Database Storage**: Compatible with ContextBox SQLite database
- **Content Extraction**: Works with ContextBox extractors
- **Configuration System**: Uses ContextBox configuration patterns

## Performance Benchmarks

### Typical Performance (Hardware Dependent)
- **7B Models**: 5-15 tokens/second on modern CPUs
- **13B Models**: 2-8 tokens/second 
- **70B Models**: 0.5-3 tokens/second
- **Memory Usage**: 4GB-64GB RAM depending on model size

### Optimization Tips
- Use quantized models for faster inference
- Adjust temperature based on use case (lower for precise tasks)
- Set appropriate max_tokens to prevent excessive generation
- Enable streaming for better user experience

## Contributing

The Ollama integration is designed to be extensible:

1. **New Features**: Add methods to `OllamaBackend` class
2. **Custom Models**: Extend model support in configuration
3. **Error Handling**: Enhance error scenarios and recovery
4. **Performance**: Optimize for specific hardware or use cases

## License

This integration is part of ContextBox and follows the same licensing terms.

## Support

For issues, questions, or contributions:
- Check the examples and test suite
- Review error handling and logging
- Consult Ollama documentation for model-specific issues
- Submit issues to the ContextBox repository

---

*Last updated: November 2024*