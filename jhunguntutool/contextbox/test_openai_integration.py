#!/usr/bin/env python3
"""
OpenAI Integration Test for ContextBox

This script demonstrates the complete OpenAI integration with:
- Multiple models (GPT-4, GPT-3.5, etc.)
- Chat completion with structured responses
- Streaming responses
- Token counting and cost tracking
- Rate limiting and error handling
- Error handling for all OpenAI API response scenarios

Requirements:
- OpenAI API key
- Install dependencies: pip install openai tiktoken
"""

import asyncio
import os
import sys
import pytest
from typing import Dict, Any

# Add contextbox to path
sys.path.insert(0, '/workspace/contextbox')

from contextbox.llm.openai import OpenAIBackend, create_openai_backend, get_supported_models
from contextbox.llm import ChatMessage, ChatRequest, LLMResponse

# Ensure async tests run with pytest-asyncio even when not explicitly marked
pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_openai_backend():
    """Test OpenAI backend functionality."""
    
    # Get API key from environment or user
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print("🚀 Testing ContextBox OpenAI Integration")
    print("=" * 50)
    
    try:
        # Create backend using factory function
        backend = create_openai_backend(api_key, provider_name="test_openai")
        
        # Test 1: List available models
        print("\n📋 Test 1: Available Models")
        models = backend.get_available_models()
        print(f"   Available models: {len(models)}")
        for i, model in enumerate(models[:5], 1):  # Show first 5
            model_info = backend.get_model_info(model)
            print(f"   {i}. {model} - {model_info.get('description', 'N/A')}")
        
        # Test 2: Simple chat completion
        print("\n💬 Test 2: Chat Completion")
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain what artificial intelligence is in simple terms.")
        ]
        
        request = ChatRequest(
            messages=messages,
            model="gpt-4o-mini",  # Using cost-effective model for testing
            provider="test_openai",
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"   Model: {request.model}")
        print(f"   Temperature: {request.temperature}")
        print(f"   Max tokens: {request.max_tokens}")
        
        response = await backend.chat_completion(request)
        print(f"   ✅ Response received!")
        print(f"   Content: {response.content[:100]}...")
        print(f"   Usage: {response.usage.total_tokens} tokens")
        if response.cost:
            print(f"   Cost: ${response.cost.total_cost:.6f}")
        
        # Test 3: Streaming chat completion
        print("\n🌊 Test 3: Streaming Chat Completion")
        messages = [
            ChatMessage(role="user", content="Write a short story about a robot learning to paint. Make it 3-4 sentences.")
        ]
        
        request.stream = True
        request.model = "gpt-4o-mini"
        
        print(f"   Streaming from model: {request.model}")
        print("   Response: ", end="")
        
        async for chunk in await backend.stream_chat_completion(request):
            print(chunk, end="", flush=True)
        
        print("\n   ✅ Streaming completed!")
        
        # Test 4: Error handling - Invalid model
        print("\n🚫 Test 4: Error Handling (Invalid Model)")
        try:
            bad_request = ChatRequest(
                messages=[ChatMessage(role="user", content="Hello")],
                model="gpt-non-existent-model",
                provider="test_openai"
            )
            await backend.chat_completion(bad_request)
        except Exception as e:
            print(f"   ✅ Caught expected error: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}...")
        
        # Test 5: Rate limiting
        print("\n⏱️  Test 5: Rate Limit Status")
        try:
            status = await backend.get_rate_limit_status()
            print(f"   Rate limit status retrieved successfully")
            for limit_type, info in status.items():
                print(f"   {limit_type}: {info}")
        except Exception as e:
            print(f"   ⚠️  Rate limit check failed: {e}")
        
        # Test 6: Usage statistics
        print("\n📊 Test 6: Usage Statistics")
        try:
            stats = await backend.get_usage_stats()
            print(f"   Usage stats retrieved successfully")
            if "cost_tracking" in stats:
                for model, model_stats in stats["cost_tracking"].items():
                    print(f"   {model}: {model_stats['requests']} requests, {model_stats['total_tokens']} tokens")
        except Exception as e:
            print(f"   ⚠️  Usage stats failed: {e}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_token_counting():
    """Test token counting functionality."""
    print("\n🔢 Testing Token Counting")
    print("-" * 30)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("   ⚠️  Skipping token counting test (no API key)")
        return
    
    try:
        backend = create_openai_backend(api_key)
        await backend.initialize()
        
        # Test different text lengths
        texts = [
            "Hello",
            "This is a longer sentence with more words.",
            "This is a much longer paragraph that contains multiple sentences and should have significantly more tokens than the previous examples."
        ]
        
        for text in texts:
            tokens = backend.count_tokens(text, "gpt-4o-mini")
            print(f"   Text: '{text[:50]}...'")
            print(f"   Tokens: {tokens}")
            print()
        
        # Test message counting
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, how are you today?"),
            ChatMessage(role="assistant", content="I'm doing well, thank you! How can I help you?")
        ]
        
        token_counter = backend._token_counter
        message_tokens = token_counter.count_messages(messages, "gpt-4o-mini")
        print(f"   Messages ({len(messages)} messages): {message_tokens} tokens")
        
    except Exception as e:
        print(f"   ❌ Token counting test failed: {e}")


async def demonstrate_cost_tracking():
    """Demonstrate cost tracking features."""
    print("\n💰 Demonstrating Cost Tracking")
    print("-" * 35)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("   ⚠️  Skipping cost tracking demo (no API key)")
        return
    
    try:
        backend = create_openai_backend(api_key)
        await backend.initialize()
        
        # Get model pricing information
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
        
        print("   Model Pricing (per 1K tokens):")
        for model in models:
            if backend.is_model_available(model):
                info = backend.get_model_info(model)
                input_price = info["input_price_per_1k"]
                output_price = info["output_price_per_1k"]
                print(f"   {model}:")
                print(f"     Input: ${input_price:.6f}/1K tokens")
                print(f"     Output: ${output_price:.6f}/1K tokens")
        
        # Calculate example costs
        print("\n   Example Cost Calculations:")
        test_scenarios = [
            ("gpt-4o-mini", 1000, 500, "Short query and response"),
            ("gpt-4o", 2000, 1000, "Medium conversation"),
            ("gpt-4", 5000, 2000, "Long document processing")
        ]
        
        for model, input_tokens, output_tokens, description in test_scenarios:
            if backend.is_model_available(model):
                cost = backend.get_model_info(model)
                input_cost = (input_tokens / 1000) * cost["input_price_per_1k"]
                output_cost = (output_tokens / 1000) * cost["output_price_per_1k"]
                total_cost = input_cost + output_cost
                
                print(f"   {description}:")
                print(f"     Model: {model}")
                print(f"     Tokens: {input_tokens} input, {output_tokens} output")
                print(f"     Cost: ${total_cost:.6f}")
        
    except Exception as e:
        print(f"   ❌ Cost tracking demo failed: {e}")


async def main():
    """Main test function."""
    print("ContextBox OpenAI Integration Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    success = await test_openai_backend()
    
    # Test additional features
    await test_token_counting()
    await demonstrate_cost_tracking()
    
    if success:
        print("\n✅ All core functionality tests passed!")
        print("\nNext steps:")
        print("1. Set up your OPENAI_API_KEY environment variable")
        print("2. Install required dependencies: pip install openai tiktoken")
        print("3. Integrate OpenAI backend into your ContextBox workflow")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
