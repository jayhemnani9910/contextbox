"""
ContextBox Ollama Integration Examples

This file demonstrates how to use the Ollama backend integration for ContextBox,
including model management, chat completion, streaming, and advanced features.
"""

import asyncio
import logging
from typing import List, Dict, Any

from contextbox.llm import (
    OllamaBackend, 
    create_ollama_backend,
    ChatRequest, 
    CompletionRequest, 
    ChatMessage,
    create_chat_message
)


async def basic_chat_example():
    """Example of basic chat completion with Ollama."""
    print("=== Basic Chat Example ===")
    
    # Create backend
    backend = create_ollama_backend(
        base_url="http://localhost:11434",
        default_model="llama2"
    )
    
    try:
        await backend.initialize()
        
        # Health check
        health = await backend.health_check()
        print(f"Health check: {health}")
        
        if not health["healthy"]:
            print("Ollama service not available. Please start Ollama service.")
            return
        
        # List available models
        models = await backend.list_models()
        print(f"Available models: {len(models)}")
        for model in models[:3]:  # Show first 3 models
            print(f"  - {model['name']} ({model['size']} bytes)")
        
        # Create chat request
        messages = [
            create_chat_message("system", "You are a helpful assistant."),
            create_chat_message("user", "Explain what AI is in simple terms.")
        ]
        
        chat_request = ChatRequest(
            messages=messages,
            model=models[0]["name"] if models else "llama2",
            provider="ollama",
            temperature=0.7,
            max_tokens=500
        )
        
        # Perform chat completion
        response = await backend.chat_completion(chat_request)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage.total_tokens} tokens")
        print(f"Provider: {response.provider}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def streaming_chat_example():
    """Example of streaming chat completion."""
    print("\n=== Streaming Chat Example ===")
    
    backend = create_ollama_backend(default_model="llama2")
    
    try:
        await backend.initialize()
        
        # Check if streaming is supported
        features = backend.get_supported_features()
        if not features["streaming"]:
            print("Streaming not supported by this backend")
            return
        
        # Create streaming chat request
        messages = [
            create_chat_message("system", "You are a creative storyteller."),
            create_chat_message("user", "Tell me a short story about a robot learning to paint.")
        ]
        
        chat_request = ChatRequest(
            messages=messages,
            model="llama2",
            provider="ollama",
            temperature=0.8,
            max_tokens=300,
            stream=True
        )
        
        # Stream the response
        print("Streaming response:")
        async for chunk in backend.stream_chat_completion(chat_request):
            print(chunk, end="", flush=True)
        
        print("\n")  # New line after streaming
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def model_management_example():
    """Example of model management operations."""
    print("\n=== Model Management Example ===")
    
    backend = create_ollama_backend()
    
    try:
        await backend.initialize()
        
        # List all models
        models = await backend.list_models()
        print(f"Currently installed models: {len(models)}")
        
        for model in models:
            print(f"  - {model['name']} ({model['size']:,} bytes)")
        
        # Get detailed info for first model
        if models:
            model_name = models[0]["name"]
            model_info = await backend.get_model_info(model_name)
            print(f"\nModel info for {model_name}:")
            print(f"  Modified: {model_info.get('modified_at', 'Unknown')}")
            print(f"  Size: {model_info.get('size', 0):,} bytes")
            print(f"  Details: {model_info.get('details', {})}")
        
        # Example of pulling a new model (commented out to avoid accidental downloads)
        print("\nTo pull a new model, use:")
        print("await backend.pull_model('namespace/model:tag', progress_callback=callback)")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def token_counting_example():
    """Example of token counting functionality."""
    print("\n=== Token Counting Example ===")
    
    backend = create_ollama_backend()
    
    try:
        await backend.initialize()
        
        texts = [
            "Hello, world! This is a simple test.",
            "Artificial intelligence is revolutionizing many industries.",
            "ContextBox is a powerful tool for content extraction and analysis."
        ]
        
        print("Token counts for different texts:")
        for text in texts:
            token_count = await backend.count_tokens(text, "llama2")
            char_count = len(text)
            print(f"Text: '{text}'")
            print(f"  Characters: {char_count}")
            print(f"  Tokens: {token_count}")
            print(f"  Ratio: {token_count / char_count:.3f} tokens/char")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def performance_monitoring_example():
    """Example of performance monitoring."""
    print("\n=== Performance Monitoring Example ===")
    
    backend = create_ollama_backend()
    
    try:
        await backend.initialize()
        
        # Perform a few requests to generate metrics
        messages = [create_chat_message("user", "Hello!")]
        
        for i in range(3):
            request = ChatRequest(
                messages=messages,
                model="llama2",
                provider="ollama",
                temperature=0.7
            )
            try:
                response = await backend.chat_completion(request)
                print(f"Request {i+1}: {response.usage.total_tokens} tokens")
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        # Get performance statistics
        stats = await backend.get_performance_stats()
        print("\nPerformance statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def contextbox_integration_example():
    """Example of ContextBox integration features."""
    print("\n=== ContextBox Integration Example ===")
    
    backend = create_ollama_backend()
    
    try:
        await backend.initialize()
        
        # Example text for analysis
        text = """
        ContextBox is a powerful desktop application that captures and analyzes 
        screen content. It can extract text, URLs, and other information from 
        screenshots, web pages, and documents. The application supports various 
        content extraction methods including OCR, web scraping, and AI-powered 
        analysis. Users can organize their captured information in a SQLite 
        database and search through their content history.
        """
        
        # Generate summary
        print("Generating summary...")
        summary = await backend.generate_summary(text, max_length=100)
        print(f"Summary: {summary}")
        
        # Extract key points
        print("\nExtracting key points...")
        key_points = await backend.extract_key_points(text, max_points=5)
        print("Key Points:")
        for i, point in enumerate(key_points, 1):
            print(f"  {i}. {point}")
        
        # Analyze sentiment
        print("\nAnalyzing sentiment...")
        sentiment = await backend.analyze_sentiment(text)
        print(f"Sentiment: {sentiment}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def custom_configuration_example():
    """Example of custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    from contextbox.llm.config import LLMBackendConfig, ProviderConfig, ModelConfig, ModelType
    
    # Create custom configuration
    config = LLMBackendConfig(
        default_provider="ollama",
        enable_logging=True,
        log_level="INFO"
    )
    
    # Create custom provider config
    provider_config = ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434",
        timeout=120,  # Longer timeout for local inference
        max_retries=5,
        models={
            "my-custom-model": ModelConfig(
                name="my-custom-model",
                model_type=ModelType.CHAT,
                provider="ollama",
                max_tokens=2048,
                temperature=0.5,
                top_p=0.8
            )
        },
        default_model="my-custom-model"
    )
    
    config.providers["ollama"] = provider_config
    
    # Create backend with custom config
    backend = OllamaBackend(config)
    
    try:
        await backend.initialize()
        print("Backend initialized with custom configuration")
        
        # Check configuration
        health = await backend.health_check()
        print(f"Health status: {health.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def error_handling_example():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    backend = create_ollama_backend()
    
    try:
        await backend.initialize()
        
        # Test with non-existent model
        messages = [create_chat_message("user", "Hello")]
        request = ChatRequest(
            messages=messages,
            model="non-existent-model",
            provider="ollama"
        )
        
        try:
            response = await backend.chat_completion(request)
            print("Unexpected success with non-existent model")
        except Exception as e:
            print(f"Expected error with non-existent model: {type(e).__name__}")
        
        # Test with empty messages
        empty_request = ChatRequest(
            messages=[],
            model="llama2",
            provider="ollama"
        )
        
        try:
            response = await backend.chat_completion(empty_request)
            print("Unexpected success with empty messages")
        except Exception as e:
            print(f"Expected error with empty messages: {type(e).__name__}")
        
        print("Error handling examples completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await backend.close()


async def main():
    """Run all examples."""
    print("ContextBox Ollama Integration Examples")
    print("=====================================")
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    examples = [
        basic_chat_example,
        streaming_chat_example,
        model_management_example,
        token_counting_example,
        performance_monitoring_example,
        contextbox_integration_example,
        custom_configuration_example,
        error_handling_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example {example.__name__} failed: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Check if Ollama service is running
    print("Checking if Ollama service is available...")
    
    # Try to create backend and check health
    try:
        backend = create_ollama_backend()
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to start examples: {e}")
        print("\nMake sure Ollama is installed and running:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        print("4. Run this script again")