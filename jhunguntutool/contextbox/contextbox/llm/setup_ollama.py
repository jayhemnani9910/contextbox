#!/usr/bin/env python3
"""
ContextBox Ollama Integration Setup Script

This script helps users set up the Ollama integration for ContextBox,
including installing dependencies, configuring settings, and running tests.
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_ollama_installation():
    """Check if Ollama is installed and running."""
    try:
        # Check if Ollama command exists
        result = subprocess.run(
            ["which", "ollama"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode != 0:
            print("❌ Ollama not found in PATH")
            return False
        
        print("✅ Ollama installed")
        
        # Check if Ollama service is running
        try:
            import aiohttp
            
            async def check_service():
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get("http://localhost:11434/api/tags") as response:
                            return response.status == 200
                    except:
                        return False
            
            if asyncio.run(check_service()):
                print("✅ Ollama service is running")
                return True
            else:
                print("⚠️  Ollama service is not running")
                print("   Start it with: ollama serve")
                return False
                
        except ImportError:
            print("⚠️  aiohttp not available - cannot check service status")
            return True  # Assume it's ok
            
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def install_dependencies():
    """Install required Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    dependencies = [
        "aiohttp>=3.8.0",
        "tiktoken>=0.4.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL)
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True


def setup_configuration():
    """Set up Ollama configuration."""
    print("\n⚙️  Setting up configuration...")
    
    # Create default configuration
    config_data = {
        "default_provider": "ollama",
        "enable_logging": True,
        "log_level": "INFO",
        "providers": {
            "ollama": {
                "name": "ollama",
                "base_url": "http://localhost:11434",
                "timeout": 60,
                "max_retries": 3,
                "retry_delay": 2.0,
                "rate_limit": {
                    "requests_per_minute": 30,
                    "requests_per_hour": 500,
                    "tokens_per_minute": 50000,
                    "burst_limit": 5
                },
                "cost_config": {
                    "track_usage": True,
                    "track_costs": False,
                    "budget_limit": None,
                    "alert_threshold": 0.8
                },
                "models": {
                    "llama2": {
                        "name": "llama2",
                        "model_type": "chat",
                        "provider": "ollama",
                        "max_tokens": 4096,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0
                    },
                    "codellama": {
                        "name": "codellama",
                        "model_type": "chat", 
                        "provider": "ollama",
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0
                    },
                    "mistral": {
                        "name": "mistral",
                        "model_type": "chat",
                        "provider": "ollama", 
                        "max_tokens": 4096,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0
                    }
                },
                "default_model": "llama2"
            }
        }
    }
    
    # Create config directory
    config_dir = Path.home() / ".contextbox"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "ollama_config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        return False


async def test_integration():
    """Test the Ollama integration."""
    print("\n🧪 Testing Ollama integration...")
    
    try:
        # Import modules
        from contextbox.llm import create_ollama_backend, ChatRequest, create_chat_message
        
        # Create backend
        backend = create_ollama_backend()
        
        try:
            await backend.initialize()
            
            # Test health check
            health = await backend.health_check()
            if health["healthy"]:
                print("✅ Health check passed")
            else:
                print(f"⚠️  Health check failed: {health.get('error', 'Unknown error')}")
            
            # Test model listing
            try:
                models = await backend.list_models()
                print(f"✅ Found {len(models)} models")
                
                if models:
                    print("📋 Available models:")
                    for model in models[:3]:  # Show first 3
                        print(f"   - {model['name']} ({model['size']:,} bytes)")
                else:
                    print("⚠️  No models found")
                    print("   Pull a model with: ollama pull llama2")
                    
            except Exception as e:
                print(f"⚠️  Model listing failed: {e}")
            
            # Test basic chat (if models available)
            if models:
                try:
                    messages = [create_chat_message("user", "Hello!")]
                    request = ChatRequest(
                        messages=messages,
                        model=models[0]["name"],
                        provider="ollama"
                    )
                    
                    response = await backend.chat_completion(request)
                    print(f"✅ Chat test successful - {response.usage.total_tokens} tokens")
                    
                except Exception as e:
                    print(f"⚠️  Chat test failed: {e}")
            else:
                print("⏭️  Skipping chat test (no models available)")
            
            await backend.close()
            return True
            
        except Exception as e:
            print(f"❌ Backend initialization failed: {e}")
            await backend.close()
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. Start Ollama service: ollama serve")
    print("2. Pull a model: ollama pull llama2")
    print("3. Run examples: python -m contextbox.llm.examples.ollama_examples")
    print("4. Check configuration: cat ~/.contextbox/ollama_config.json")
    
    print("\n📚 Documentation:")
    print("- Full docs: contextbox/llm/README.md")
    print("- Examples: contextbox/llm/examples/ollama_examples.py")
    print("- Tests: contextbox/llm/tests/test_ollama_integration.py")
    
    print("\n🔧 Configuration:")
    print("- Config file: ~/.contextbox/ollama_config.json")
    print("- Modify models, parameters, and settings as needed")
    
    print("\n💡 Tips:")
    print("- Start with smaller models (7B) for testing")
    print("- Monitor memory usage with larger models")
    print("- Check logs for troubleshooting")


def main():
    """Main setup function."""
    print("🚀 ContextBox Ollama Integration Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_ollama_installation():
        print("\n❌ Setup cannot continue without Ollama")
        print("Install Ollama: https://ollama.ai")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        sys.exit(1)
    
    # Setup configuration
    if not setup_configuration():
        print("\n❌ Configuration setup failed")
        sys.exit(1)
    
    # Test integration
    success = asyncio.run(test_integration())
    
    if success:
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Some features may not work correctly")
        print("Check the error messages above for details")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)