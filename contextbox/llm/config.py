"""
Configuration management for LLM backends.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .exceptions import ConfigurationError


class ModelType(Enum):
    """Supported model types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDINGS = "embeddings"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000
    burst_limit: int = 10
    
    def validate(self):
        """Validate rate limit configuration."""
        if self.requests_per_minute <= 0:
            raise ConfigurationError("requests_per_minute must be positive")
        if self.requests_per_hour <= 0:
            raise ConfigurationError("requests_per_hour must be positive")
        if self.tokens_per_minute <= 0:
            raise ConfigurationError("tokens_per_minute must be positive")
        if self.burst_limit <= 0:
            raise ConfigurationError("burst_limit must be positive")


@dataclass
class CostConfig:
    """Cost tracking configuration."""
    track_usage: bool = True
    track_costs: bool = True
    budget_limit: Optional[float] = None
    alert_threshold: float = 0.8  # Alert when 80% of budget used
    
    def validate(self):
        """Validate cost configuration."""
        if self.budget_limit is not None and self.budget_limit <= 0:
            raise ConfigurationError("budget_limit must be positive")
        if not 0 <= self.alert_threshold <= 1:
            raise ConfigurationError("alert_threshold must be between 0 and 1")


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_type: ModelType
    provider: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    api_endpoint: Optional[str] = None
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ConfigurationError("Model name cannot be empty")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ConfigurationError("top_p must be between 0 and 1")
        if not -2 <= self.frequency_penalty <= 2:
            raise ConfigurationError("frequency_penalty must be between -2 and 2")
        if not -2 <= self.presence_penalty <= 2:
            raise ConfigurationError("presence_penalty must be between -2 and 2")
    
    def validate(self):
        """Validate model configuration."""
        self.__post_init__()


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    default_model: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cost_config: CostConfig = field(default_factory=CostConfig)
    
    def validate(self):
        """Validate configuration."""
        if not self.name:
            raise ConfigurationError("Provider name cannot be empty")
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be positive")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries cannot be negative")
        if self.retry_delay < 0:
            raise ConfigurationError("retry_delay cannot be negative")
        
        # Validate rate limit and cost config
        self.rate_limit.validate()
        self.cost_config.validate()
        
        # Validate default model
        if self.default_model and self.default_model not in self.models:
            raise ConfigurationError(
                f"Default model '{self.default_model}' not found in models"
            )
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def get_model(self, model_name: str) -> ModelConfig:
        """Get a model configuration by name."""
        if model_name not in self.models:
            raise ConfigurationError(
                f"Model '{model_name}' not found in provider '{self.name}'"
            )
        return self.models[model_name]
    
    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        if not self.default_model:
            return None
        return self.get_model(self.default_model)


@dataclass
class LLMBackendConfig:
    """Main configuration for LLM backend."""
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: Optional[str] = None
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # seconds
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.default_provider and self.default_provider not in self.providers:
            raise ConfigurationError(
                f"Default provider '{self.default_provider}' not found"
            )
    
    def get_provider(self, provider_name: str) -> ProviderConfig:
        """Get a provider configuration by name."""
        if provider_name not in self.providers:
            raise ConfigurationError(
                f"Provider '{provider_name}' not found"
            )
        return self.providers[provider_name]
    
    def get_default_provider(self) -> Optional[ProviderConfig]:
        """Get the default provider configuration."""
        if not self.default_provider:
            return None
        return self.get_provider(self.default_provider)


class ConfigManager:
    """Configuration manager for LLM backend."""
    
    DEFAULT_CONFIG_PATHS = [
        "~/.contextbox/llm_config.json",
        "./contextbox_config.json",
        "/workspace/contextbox/llm_config.json"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config: Optional[LLMBackendConfig] = None
    
    def load_config(self, config_path: Optional[str] = None) -> LLMBackendConfig:
        """Load configuration from file or create default."""
        config_path = config_path or self.config_path
        
        if config_path:
            return self._load_from_file(config_path)
        else:
            # Try to find config file in default locations
            for path in self.DEFAULT_CONFIG_PATHS:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    return self._load_from_file(expanded_path)
            
            # Create default configuration
            return self._create_default_config()
    
    def _load_from_file(self, config_path: str) -> LLMBackendConfig:
        """Load configuration from a file."""
        expanded_path = os.path.expanduser(config_path)
        
        if not os.path.exists(expanded_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(expanded_path, 'r') as f:
                config_data = json.load(f)
            
            return self._parse_config_data(config_data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _create_default_config(self) -> LLMBackendConfig:
        """Create a default configuration."""
        # Create default rate limit and cost configs
        rate_limit = RateLimitConfig()
        cost_config = CostConfig()
        
        # Create default configuration
        config = LLMBackendConfig(
            enable_logging=True,
            log_level="INFO",
            log_format="json",
            enable_monitoring=True
        )
        
        return config
    
    def _parse_config_data(self, data: Dict[str, Any]) -> LLMBackendConfig:
        """Parse configuration data into LLMBackendConfig."""
        try:
            providers = {}
            
            for provider_name, provider_data in data.get("providers", {}).items():
                models = {}
                
                for model_name, model_data in provider_data.get("models", {}).items():
                    model_config = ModelConfig(
                        name=model_name,
                        model_type=ModelType(model_data["model_type"]),
                        provider=provider_name,
                        max_tokens=model_data.get("max_tokens", 4096),
                        temperature=model_data.get("temperature", 0.7),
                        top_p=model_data.get("top_p", 1.0),
                        frequency_penalty=model_data.get("frequency_penalty", 0.0),
                        presence_penalty=model_data.get("presence_penalty", 0.0),
                        stop_sequences=model_data.get("stop_sequences"),
                        api_endpoint=model_data.get("api_endpoint"),
                        cost_per_input_token=model_data.get("cost_per_input_token"),
                        cost_per_output_token=model_data.get("cost_per_output_token")
                    )
                    model_config.validate()
                    models[model_name] = model_config
                
                rate_limit_data = provider_data.get("rate_limit", {})
                rate_limit = RateLimitConfig(**rate_limit_data)
                
                cost_data = provider_data.get("cost_config", {})
                cost_config = CostConfig(**cost_data)
                
                provider_config = ProviderConfig(
                    name=provider_name,
                    api_key=provider_data.get("api_key"),
                    base_url=provider_data.get("base_url"),
                    models=models,
                    default_model=provider_data.get("default_model"),
                    timeout=provider_data.get("timeout", 30),
                    max_retries=provider_data.get("max_retries", 3),
                    retry_delay=provider_data.get("retry_delay", 1.0),
                    rate_limit=rate_limit,
                    cost_config=cost_config
                )
                
                providers[provider_name] = provider_config
            
            config = LLMBackendConfig(
                providers=providers,
                default_provider=data.get("default_provider"),
                enable_logging=data.get("enable_logging", True),
                log_level=data.get("log_level", "INFO"),
                log_format=data.get("log_format", "json"),
                enable_monitoring=data.get("enable_monitoring", True),
                monitoring_interval=data.get("monitoring_interval", 60)
            )
            
            return config
        except Exception as e:
            raise ConfigurationError(f"Error parsing configuration data: {e}")
    
    def save_config(self, config: LLMBackendConfig, config_path: Optional[str] = None):
        """Save configuration to file."""
        config_path = config_path or self.config_path
        if not config_path:
            raise ConfigurationError("No config path specified")
        
        expanded_path = os.path.expanduser(config_path)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        
        try:
            with open(expanded_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def validate_config(self, config: LLMBackendConfig):
        """Validate a complete configuration."""
        if not config.providers:
            raise ConfigurationError("No providers configured")
        
        for provider_name, provider in config.providers.items():
            try:
                provider.validate()
            except ConfigurationError as e:
                raise ConfigurationError(
                    f"Invalid configuration for provider '{provider_name}': {e}"
                )