"""
Enhanced Configuration Management System for ContextBox.

This module provides comprehensive configuration loading, validation, management,
and hot-reloading capabilities with support for multiple profiles.
"""

import json
import yaml
import os
import sys
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataclasses import dataclass, field, asdict
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "contextbox.db"
    backup_enabled: bool = True
    backup_interval: int = 24  # hours
    backup_retention: int = 7  # days
    auto_vacuum: bool = True
    cache_size: int = 1000  # pages
    timeout: int = 30  # seconds
    max_retries: int = 3
    connection_pool_size: int = 5


@dataclass
class CaptureConfig:
    """Context capture configuration."""
    screenshot_enabled: bool = True
    screenshot_format: str = "png"
    screenshot_quality: int = 95
    auto_screenshot: bool = False
    screenshot_interval: int = 10  # seconds
    clipboard_monitor: bool = True
    active_window_capture: bool = True
    extract_text_automatically: bool = True
    extract_urls_automatically: bool = True
    artifact_retention: int = 30  # days
    compression_enabled: bool = True
    max_screenshot_size: int = 10  # MB


@dataclass
class ContentExtractionConfig:
    """Content extraction configuration."""
    enabled: bool = True
    extract_urls: bool = True
    extract_text: bool = True
    extract_images: bool = True
    ocr_enabled: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["eng"])
    ocr_min_confidence: int = 70
    url_timeout: int = 30
    max_url_content_length: int = 100000  # chars
    store_extracted_content: bool = True
    extract_metadata: bool = True
    follow_redirects: bool = True
    respect_robots_txt: bool = True
    rate_limit_delay: float = 1.0  # seconds between requests


@dataclass
class LLMConfig:
    """LLM integration configuration."""
    provider: str = "openai"  # openai, anthropic, local, none
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 60
    retry_attempts: int = 3
    local_model_path: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "contextbox.log"
    file_max_size: int = 10  # MB
    file_backup_count: int = 5
    console_enabled: bool = True
    structured_logging: bool = False
    log_to_database: bool = False


@dataclass
class UIConfig:
    """User interface configuration."""
    theme: str = "dark"
    language: str = "en"
    auto_start: bool = False
    minimize_to_tray: bool = True
    show_notifications: bool = True
    window_geometry: Dict[str, int] = field(default_factory=lambda: {
        "width": 1200,
        "height": 800,
        "x": 100,
        "y": 100
    })
    last_opened_tab: str = "overview"


@dataclass
class SecurityConfig:
    """Security configuration."""
    encrypt_database: bool = False
    encrypt_artifacts: bool = False
    api_key_encryption: bool = True
    audit_logging: bool = True
    session_timeout: int = 3600  # seconds
    max_file_size: int = 100  # MB
    allowed_file_types: List[str] = field(default_factory=lambda: [
        ".txt", ".json", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".doc", ".docx"
    ])
    scan_for_malware: bool = False


@dataclass
class ContextBoxConfig:
    """Main configuration class."""
    profile: str = "default"
    debug: bool = False
    data_directory: str = ""
    temp_directory: str = ""
    
    # Subsystem configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    content_extraction: ContentExtractionConfig = field(default_factory=ContentExtractionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadata
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default directories if not specified
        if not self.data_directory:
            from .utils import get_app_data_dir
            self.data_directory = get_app_data_dir()
        
        if not self.temp_directory:
            import tempfile
            self.temp_directory = tempfile.gettempdir()
        
        # Ensure directories exist
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.temp_directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextBoxConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ContextBoxConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert nested dicts to configuration objects
        config_dict = cls._convert_dict_to_config(data)
        return cls.from_dict(config_dict)
    
    @staticmethod
    def _convert_dict_to_config(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested dictionaries to configuration objects."""
        config_classes = {
            'database': DatabaseConfig,
            'capture': CaptureConfig,
            'content_extraction': ContentExtractionConfig,
            'llm': LLMConfig,
            'logging': LoggingConfig,
            'ui': UIConfig,
            'security': SecurityConfig
        }
        
        result = data.copy()
        for key, config_class in config_classes.items():
            if key in result and isinstance(result[key], dict):
                result[key] = config_class(**result[key])
        
        return result


class ConfigValidator:
    """Configuration validation with helpful error messages."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: ContextBoxConfig) -> bool:
        """Validate configuration and return True if valid."""
        self.errors.clear()
        self.warnings.clear()
        
        self._validate_directories(config)
        self._validate_database_config(config.database)
        self._validate_capture_config(config.capture)
        self._validate_content_extraction_config(config.content_extraction)
        self._validate_llm_config(config.llm)
        self._validate_logging_config(config.logging)
        self._validate_security_config(config.security)
        
        return len(self.errors) == 0
    
    def _validate_directories(self, config: ContextBoxConfig) -> None:
        """Validate directory configurations."""
        # Check data directory
        if not os.path.exists(config.data_directory):
            try:
                os.makedirs(config.data_directory, exist_ok=True)
            except Exception as e:
                self.errors.append(f"Cannot create data directory '{config.data_directory}': {e}")
        
        if not os.access(config.data_directory, os.W_OK):
            self.errors.append(f"Data directory is not writable: {config.data_directory}")
        
        # Check temp directory
        if not os.path.exists(config.temp_directory):
            self.warnings.append(f"Temp directory does not exist: {config.temp_directory}")
        
        if not os.access(config.temp_directory, os.W_OK):
            self.errors.append(f"Temp directory is not writable: {config.temp_directory}")
    
    def _validate_database_config(self, config: DatabaseConfig) -> None:
        """Validate database configuration."""
        if config.cache_size <= 0:
            self.errors.append("Database cache size must be positive")
        
        if config.timeout <= 0:
            self.errors.append("Database timeout must be positive")
        
        if config.max_retries < 0:
            self.errors.append("Database max_retries cannot be negative")
        
        if config.backup_interval <= 0:
            self.errors.append("Database backup interval must be positive")
        
        if config.backup_retention < 0:
            self.errors.append("Database backup retention cannot be negative")
    
    def _validate_capture_config(self, config: CaptureConfig) -> None:
        """Validate capture configuration."""
        if config.screenshot_quality < 0 or config.screenshot_quality > 100:
            self.errors.append("Screenshot quality must be between 0 and 100")
        
        if config.screenshot_interval <= 0:
            self.errors.append("Screenshot interval must be positive")
        
        if config.artifact_retention < 0:
            self.errors.append("Artifact retention cannot be negative")
        
        if config.max_screenshot_size <= 0:
            self.errors.append("Max screenshot size must be positive")
    
    def _validate_content_extraction_config(self, config: ContentExtractionConfig) -> None:
        """Validate content extraction configuration."""
        if config.ocr_min_confidence < 0 or config.ocr_min_confidence > 100:
            self.errors.append("OCR confidence must be between 0 and 100")
        
        if config.url_timeout <= 0:
            self.errors.append("URL timeout must be positive")
        
        if config.max_url_content_length <= 0:
            self.errors.append("Max URL content length must be positive")
        
        if config.rate_limit_delay < 0:
            self.errors.append("Rate limit delay cannot be negative")
        
        if not config.ocr_languages:
            self.warnings.append("No OCR languages specified - OCR may not work")
    
    def _validate_llm_config(self, config: LLMConfig) -> None:
        """Validate LLM configuration."""
        if config.max_tokens <= 0:
            self.errors.append("LLM max_tokens must be positive")
        
        if config.temperature < 0 or config.temperature > 2:
            self.errors.append("LLM temperature must be between 0 and 2")
        
        if config.timeout <= 0:
            self.errors.append("LLM timeout must be positive")
        
        if config.retry_attempts < 0:
            self.errors.append("LLM retry_attempts cannot be negative")
        
        # Validate provider-specific settings
        if config.provider == "openai" and not config.api_key:
            self.warnings.append("OpenAI API key not configured")
        
        if config.provider == "local" and not config.local_model_path:
            self.errors.append("Local model path required for local LLM provider")
    
    def _validate_logging_config(self, config: LoggingConfig) -> None:
        """Validate logging configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.level.upper() not in valid_levels:
            self.errors.append(f"Invalid log level: {config.level}")
        
        if config.file_max_size <= 0:
            self.errors.append("Log file max size must be positive")
        
        if config.file_backup_count < 0:
            self.errors.append("Log file backup count cannot be negative")
        
        if config.file_enabled:
            log_dir = os.path.dirname(config.file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    self.warnings.append(f"Cannot create log directory: {e}")
    
    def _validate_security_config(self, config: SecurityConfig) -> None:
        """Validate security configuration."""
        if config.session_timeout <= 0:
            self.errors.append("Security session timeout must be positive")
        
        if config.max_file_size <= 0:
            self.errors.append("Security max file size must be positive")
        
        if not config.allowed_file_types:
            self.errors.append("At least one file type must be allowed")


class ConfigManager:
    """Enhanced configuration manager with hot-reloading and profile support."""
    
    def __init__(self, config_dir: Optional[str] = None):
        desired_dir = Path(config_dir or self._get_default_config_dir())
        self.config_dir = self._ensure_writable_directory(desired_dir)
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.profiles_dir = os.path.join(self.config_dir, "profiles")
        self.config: Optional[ContextBoxConfig] = None
        self.validator = ConfigValidator()
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Hot-reloading
        self.file_watcher = None
        self.watcher_thread = None
        self.config_listeners: List[callable] = []
        
        # Initialize directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def _get_default_config_dir(self) -> str:
        """Get default configuration directory."""
        from .utils import get_app_data_dir
        return os.path.join(get_app_data_dir(), "config")

    def _ensure_writable_directory(self, target_dir: Path) -> str:
        """Ensure configuration directory is writable, falling back if necessary."""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            test_file = target_dir / ".contextbox_write_test"
            with test_file.open("w") as handle:
                handle.write("ok")
            test_file.unlink(missing_ok=True)
            return str(target_dir)
        except Exception:
            fallback = Path(tempfile.gettempdir()) / "contextbox_config"
            fallback.mkdir(parents=True, exist_ok=True)
            return str(fallback)
    
    def load_config(self, profile: str = "default", create_if_missing: bool = True) -> ContextBoxConfig:
        """Load configuration for specified profile."""
        profile_config = self._get_profile_config_file(profile)
        
        try:
            if os.path.exists(profile_config):
                config = ContextBoxConfig.load(profile_config)
            else:
                if create_if_missing:
                    config = self._create_default_config(profile)
                    self.save_config(config, profile)
                else:
                    raise FileNotFoundError(f"Configuration file not found: {profile_config}")
            
            # Validate configuration
            if not self.validator.validate_config(config):
                self._display_validation_errors()
                raise ValueError("Configuration validation failed")
            
            self.config = config
            self.logger.info(f"Loaded configuration for profile: {profile}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_config(self, config: ContextBoxConfig, profile: str = "default") -> None:
        """Save configuration to file."""
        profile_config = self._get_profile_config_file(profile)
        
        # Update metadata
        config.updated_at = datetime.now().isoformat()
        
        # Save configuration
        config.save(profile_config)
        
        # Update current config if it matches the saved profile
        if self.config and self.config.profile == profile:
            self.config = config
        
        self.logger.info(f"Saved configuration for profile: {profile}")
    
    def _get_profile_config_file(self, profile: str) -> str:
        """Get configuration file path for profile."""
        if profile == "default":
            return self.config_file
        
        return os.path.join(self.profiles_dir, f"{profile}.json")
    
    def _create_default_config(self, profile: str = "default") -> ContextBoxConfig:
        """Create default configuration."""
        config = ContextBoxConfig()
        config.profile = profile
        return config
    
    def _display_validation_errors(self) -> None:
        """Display validation errors and warnings."""
        if self.validator.errors:
            self.console.print("\n[red]Configuration Errors:[/red]")
            for error in self.validator.errors:
                self.console.print(f"  • {error}")
        
        if self.validator.warnings:
            self.console.print("\n[yellow]Configuration Warnings:[/yellow]")
            for warning in self.validator.warnings:
                self.console.print(f"  • {warning}")
        
        if self.validator.errors or self.validator.warnings:
            self.console.print()
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles."""
        profiles = ["default"]
        
        # Check profiles directory
        if os.path.exists(self.profiles_dir):
            for file in os.listdir(self.profiles_dir):
                if file.endswith('.json'):
                    profile_name = file[:-5]  # Remove .json extension
                    profiles.append(profile_name)
        
        return sorted(list(set(profiles)))
    
    def create_profile(self, source_profile: str = "default", new_profile: str = None) -> str:
        """Create new profile from existing profile."""
        if new_profile is None:
            new_profile = Prompt.ask("Enter new profile name")
        
        if new_profile in self.list_profiles():
            raise ValueError(f"Profile '{new_profile}' already exists")
        
        # Load source configuration
        source_config = self.load_config(source_profile)
        
        # Create new profile
        new_config = ContextBoxConfig.from_dict(source_config.to_dict())
        new_config.profile = new_profile
        new_config.created_at = datetime.now().isoformat()
        new_config.updated_at = datetime.now().isoformat()
        
        # Save new profile
        self.save_config(new_config, new_profile)
        
        return new_profile
    
    def delete_profile(self, profile: str) -> None:
        """Delete configuration profile."""
        if profile == "default":
            raise ValueError("Cannot delete default profile")
        
        profile_config = self._get_profile_config_file(profile)
        if os.path.exists(profile_config):
            os.remove(profile_config)
            self.logger.info(f"Deleted profile: {profile}")
        else:
            raise FileNotFoundError(f"Profile '{profile}' not found")
    
    def export_config(self, profile: str, export_path: str) -> None:
        """Export configuration to file."""
        config = self.load_config(profile)
        config.save(export_path)
        self.console.print(f"[green]Configuration exported to: {export_path}[/green]")
    
    def import_config(self, import_path: str, profile: str = None) -> str:
        """Import configuration from file."""
        if profile is None:
            profile = Prompt.ask("Enter profile name for imported configuration")
        
        if os.path.exists(self._get_profile_config_file(profile)):
            if not Confirm.ask(f"Profile '{profile}' already exists. Overwrite?"):
                return profile
        
        # Load and validate imported config
        imported_config = ContextBoxConfig.load(import_path)
        imported_config.profile = profile
        imported_config.updated_at = datetime.now().isoformat()
        
        if not self.validator.validate_config(imported_config):
            self._display_validation_errors()
            raise ValueError("Imported configuration validation failed")
        
        # Save imported config
        self.save_config(imported_config, profile)
        self.console.print(f"[green]Configuration imported as profile: {profile}[/green]")
        
        return profile
    
    def start_hot_reload(self, callback: callable = None) -> None:
        """Start configuration hot-reloading."""
        if self.file_watcher:
            self.console.print("[yellow]Hot-reload already active[/yellow]")
            return
        
        if callback:
            self.config_listeners.append(callback)
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
                self.last_modified = {}
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                filepath = event.src_path
                if filepath.endswith(('.json', '.yaml', '.yml')):
                    try:
                        stat = os.stat(filepath)
                        last_modified = stat.st_mtime
                        
                        if filepath in self.manager.last_modified:
                            if last_modified > self.manager.last_modified[filepath]:
                                # File changed, reload config
                                self.manager._reload_config_from_file(filepath)
                                self.manager.last_modified[filepath] = last_modified
                        else:
                            self.manager.last_modified[filepath] = last_modified
                    except Exception as e:
                        logging.warning(f"Error checking config file modification: {e}")
        
        self.file_watcher = Observer()
        self.file_watcher.schedule(ConfigFileHandler(self), self.config_dir, recursive=True)
        self.file_watcher.start()
        
        self.console.print("[green]Configuration hot-reload started[/green]")
    
    def stop_hot_reload(self) -> None:
        """Stop configuration hot-reloading."""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
            self.file_watcher = None
            self.console.print("[green]Configuration hot-reload stopped[/green]")
    
    def _reload_config_from_file(self, filepath: str) -> None:
        """Reload configuration from file."""
        try:
            profile = "default"
            if self.profiles_dir in filepath:
                profile = os.path.splitext(os.path.basename(filepath))[0]
            
            old_config = self.config
            new_config = self.load_config(profile, create_if_missing=False)
            
            # Notify listeners
            for listener in self.config_listeners:
                try:
                    listener(old_config, new_config)
                except Exception as e:
                    logging.warning(f"Config listener error: {e}")
            
            self.console.print(f"[blue]Configuration reloaded from {filepath}[/blue]")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def get_config_value(self, key_path: str) -> Any:
        """Get configuration value using dot notation (e.g., 'database.path')."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Configuration key not found: {key_path}")
        
        return value
    
    def set_config_value(self, key_path: str, value: Any, save: bool = True) -> None:
        """Set configuration value using dot notation."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        keys = key_path.split('.')
        obj = self.config
        
        # Navigate to parent object
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                raise KeyError(f"Configuration key not found: {key_path}")
        
        # Set the value
        final_key = keys[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        elif isinstance(obj, dict):
            obj[final_key] = value
        else:
            raise KeyError(f"Cannot set value on {final_key}")
        
        # Save if requested
        if save:
            self.save_config(self.config, self.config.profile)


class ConfigWizard:
    """Interactive configuration wizard for first-time setup."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.console = Console()
    
    def run(self) -> ContextBoxConfig:
        """Run configuration wizard."""
        self.console.print(Panel.fit(
            "[bold blue]ContextBox Configuration Wizard[/bold blue]\n"
            "Let's set up your ContextBox configuration step by step.",
            border_style="blue"
        ))
        
        # Initialize with default config
        config = self.config_manager._create_default_config()
        
        # Run wizard steps
        config = self._setup_basic_settings(config)
        config = self._setup_database_settings(config)
        config = self._setup_capture_settings(config)
        config = self._setup_extraction_settings(config)
        config = self._setup_llm_settings(config)
        config = self._setup_logging_settings(config)
        config = self._setup_security_settings(config)
        
        # Final review and save
        config = self._final_review_and_save(config)
        
        return config
    
    def _setup_basic_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup basic settings."""
        self.console.print("\n[bold]Basic Settings[/bold]")
        
        # Profile name
        config.profile = Prompt.ask("Profile name", default=config.profile)
        
        # Data directory
        data_dir = Prompt.ask(
            "Data directory",
            default=config.data_directory
        )
        config.data_directory = data_dir
        
        # Temp directory
        temp_dir = Prompt.ask(
            "Temp directory",
            default=config.temp_directory
        )
        config.temp_directory = temp_dir
        
        # Debug mode
        config.debug = Confirm.ask("Enable debug mode", default=config.debug)
        
        return config
    
    def _setup_database_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup database settings."""
        self.console.print("\n[bold]Database Settings[/bold]")
        
        config.database.path = Prompt.ask(
            "Database path",
            default=config.database.path
        )
        
        config.database.backup_enabled = Confirm.ask(
            "Enable database backups",
            default=config.database.backup_enabled
        )
        
        if config.database.backup_enabled:
            config.database.backup_interval = int(Prompt.ask(
                "Backup interval (hours)",
                default=str(config.database.backup_interval)
            ))
        
        config.database.cache_size = int(Prompt.ask(
            "Database cache size (pages)",
            default=str(config.database.cache_size)
        ))
        
        return config
    
    def _setup_capture_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup capture settings."""
        self.console.print("\n[bold]Capture Settings[/bold]")
        
        config.capture.screenshot_enabled = Confirm.ask(
            "Enable screenshots",
            default=config.capture.screenshot_enabled
        )
        
        config.capture.clipboard_monitor = Confirm.ask(
            "Monitor clipboard",
            default=config.capture.clipboard_monitor
        )
        
        config.capture.active_window_capture = Confirm.ask(
            "Capture active window info",
            default=config.capture.active_window_capture
        )
        
        config.capture.artifact_retention = int(Prompt.ask(
            "Artifact retention (days)",
            default=str(config.capture.artifact_retention)
        ))
        
        return config
    
    def _setup_extraction_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup content extraction settings."""
        self.console.print("\n[bold]Content Extraction Settings[/bold]")
        
        config.content_extraction.enabled = Confirm.ask(
            "Enable content extraction",
            default=config.content_extraction.enabled
        )
        
        if config.content_extraction.enabled:
            config.content_extraction.extract_urls = Confirm.ask(
                "Extract URLs",
                default=config.content_extraction.extract_urls
            )
            
            config.content_extraction.extract_text = Confirm.ask(
                "Extract text",
                default=config.content_extraction.extract_text
            )
            
            config.content_extraction.ocr_enabled = Confirm.ask(
                "Enable OCR",
                default=config.content_extraction.ocr_enabled
            )
            
            if config.content_extraction.ocr_enabled:
                ocr_langs = Prompt.ask(
                    "OCR languages (comma-separated)",
                    default=",".join(config.content_extraction.ocr_languages)
                )
                config.content_extraction.ocr_languages = [
                    lang.strip() for lang in ocr_langs.split(',')
                ]
        
        return config
    
    def _setup_llm_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup LLM settings."""
        self.console.print("\n[bold]LLM Settings[/bold]")
        
        providers = ["none", "openai", "anthropic", "local"]
        config.llm.provider = Prompt.ask(
            "LLM provider",
            choices=providers,
            default=config.llm.provider
        )
        
        if config.llm.provider in ["openai", "anthropic"]:
            config.llm.api_key = Prompt.ask(
                f"{config.llm.provider.title()} API key",
                password=True
            )
        
        config.llm.model = Prompt.ask(
            "Model name",
            default=config.llm.model
        )
        
        config.llm.temperature = float(Prompt.ask(
            "Temperature (0-2)",
            default=str(config.llm.temperature)
        ))
        
        return config
    
    def _setup_logging_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup logging settings."""
        self.console.print("\n[bold]Logging Settings[/bold]")
        
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        config.logging.level = Prompt.ask(
            "Log level",
            choices=levels,
            default=config.logging.level
        )
        
        config.logging.file_enabled = Confirm.ask(
            "Log to file",
            default=config.logging.file_enabled
        )
        
        if config.logging.file_enabled:
            config.logging.file_path = Prompt.ask(
                "Log file path",
                default=config.logging.file_path
            )
        
        config.logging.console_enabled = Confirm.ask(
            "Log to console",
            default=config.logging.console_enabled
        )
        
        return config
    
    def _setup_security_settings(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Setup security settings."""
        self.console.print("\n[bold]Security Settings[/bold]")
        
        config.security.encrypt_database = Confirm.ask(
            "Encrypt database",
            default=config.security.encrypt_database
        )
        
        config.security.audit_logging = Confirm.ask(
            "Enable audit logging",
            default=config.security.audit_logging
        )
        
        config.security.scan_for_malware = Confirm.ask(
            "Scan files for malware",
            default=config.security.scan_for_malware
        )
        
        return config
    
    def _final_review_and_save(self, config: ContextBoxConfig) -> ContextBoxConfig:
        """Final review and save configuration."""
        self.console.print("\n[bold]Configuration Summary[/bold]")
        
        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Profile", config.profile)
        table.add_row("Data Directory", config.data_directory)
        table.add_row("Debug Mode", str(config.debug))
        table.add_row("Screenshot Enabled", str(config.capture.screenshot_enabled))
        table.add_row("Content Extraction", str(config.content_extraction.enabled))
        table.add_row("LLM Provider", config.llm.provider)
        table.add_row("Log Level", config.logging.level)
        
        self.console.print(table)
        
        if Confirm.ask("\nSave this configuration"):
            self.config_manager.save_config(config, config.profile)
            self.console.print(f"[green]Configuration saved as '{config.profile}'[/green]")
        else:
            if Confirm.ask("Re-run wizard"):
                return self.run()
        
        return config


# CLI Command Functions
def config_list_command(args: argparse.Namespace) -> None:
    """Handle config list command."""
    try:
        config_manager = ConfigManager()
        profiles = config_manager.list_profiles()
        
        console = Console()
        table = Table(title="Configuration Profiles")
        table.add_column("Profile", style="cyan")
        table.add_column("Status", style="green")
        
        for profile in profiles:
            status = "Current" if profile == "default" else "Available"
            table.add_row(profile, status)
        
        console.print(table)
        
    except Exception as e:
        print(f"Error listing profiles: {e}")
        sys.exit(1)


def config_show_command(args: argparse.Namespace) -> None:
    """Handle config show command."""
    try:
        config_manager = ConfigManager()
        profile = getattr(args, 'profile', 'default')
        
        if not args.all:
            config = config_manager.load_config(profile)
            
            console = Console()
            config_dict = config.to_dict()
            
            # Pretty print configuration
            json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            
            console.print(Panel(
                syntax,
                title=f"Configuration: {profile}",
                expand=False
            ))
        else:
            # Show all profiles
            profiles = config_manager.list_profiles()
            console = Console()
            
            for profile_name in profiles:
                try:
                    config = config_manager.load_config(profile_name)
                    config_dict = config.to_dict()
                    json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
                    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
                    
                    console.print(Panel(
                        syntax,
                        title=f"Configuration: {profile_name}",
                        expand=False
                    ))
                except Exception as e:
                    console.print(f"[red]Error loading {profile_name}: {e}[/red]")
        
    except Exception as e:
        print(f"Error showing configuration: {e}")
        sys.exit(1)


def config_edit_command(args: argparse.Namespace) -> None:
    """Handle config edit command."""
    try:
        config_manager = ConfigManager()
        profile = getattr(args, 'profile', 'default')
        
        config = config_manager.load_config(profile)
        
        if args.key and args.value:
            # Set specific value
            config_manager.set_config_value(args.key, args.value)
            print(f"Set {args.key} = {args.value}")
        elif args.key:
            # Get specific value
            try:
                value = config_manager.get_config_value(args.key)
                print(f"{args.key} = {value}")
            except KeyError as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            # Interactive editing (simplified - would need full TUI for real implementation)
            print("Interactive editing not implemented in this version.")
            print("Use --key and --value to set specific values.")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error editing configuration: {e}")
        sys.exit(1)


def config_create_command(args: argparse.Namespace) -> None:
    """Handle config create command."""
    try:
        config_manager = ConfigManager()
        source_profile = getattr(args, 'source', 'default')
        new_profile = args.profile
        
        profile_name = config_manager.create_profile(source_profile, new_profile)
        print(f"Created profile: {profile_name}")
        
    except Exception as e:
        print(f"Error creating profile: {e}")
        sys.exit(1)


def config_delete_command(args: argparse.Namespace) -> None:
    """Handle config delete command."""
    try:
        config_manager = ConfigManager()
        profile = args.profile
        
        if profile == "default":
            print("Error: Cannot delete default profile")
            sys.exit(1)
        
        if Confirm.ask(f"Delete profile '{profile}'? This cannot be undone."):
            config_manager.delete_profile(profile)
            print(f"Deleted profile: {profile}")
        else:
            print("Deletion cancelled")
        
    except Exception as e:
        print(f"Error deleting profile: {e}")
        sys.exit(1)


def config_validate_command(args: argparse.Namespace) -> None:
    """Handle config validate command."""
    try:
        config_manager = ConfigManager()
        profile = getattr(args, 'profile', 'default')
        
        config = config_manager.load_config(profile)
        validator = ConfigValidator()
        
        is_valid = validator.validate_config(config)
        
        if is_valid:
            print(f"✓ Configuration '{profile}' is valid")
        else:
            print(f"✗ Configuration '{profile}' has errors:")
            for error in validator.errors:
                print(f"  Error: {error}")
            
            if validator.warnings:
                print("\nWarnings:")
                for warning in validator.warnings:
                    print(f"  Warning: {warning}")
            
            sys.exit(1)
        
    except Exception as e:
        print(f"Error validating configuration: {e}")
        sys.exit(1)


def config_wizard_command(args: argparse.Namespace) -> None:
    """Handle config wizard command."""
    try:
        config_manager = ConfigManager()
        wizard = ConfigWizard(config_manager)
        
        config = wizard.run()
        print(f"Configuration wizard completed. Profile '{config.profile}' created.")
        
    except KeyboardInterrupt:
        print("\nConfiguration wizard cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running configuration wizard: {e}")
        sys.exit(1)


def config_export_command(args: argparse.Namespace) -> None:
    """Handle config export command."""
    try:
        config_manager = ConfigManager()
        profile = getattr(args, 'profile', 'default')
        
        config_manager.export_config(profile, args.output)
        print(f"Configuration exported to: {args.output}")
        
    except Exception as e:
        print(f"Error exporting configuration: {e}")
        sys.exit(1)


def config_import_command(args: argparse.Namespace) -> None:
    """Handle config import command."""
    try:
        config_manager = ConfigManager()
        
        profile = config_manager.import_config(args.file, args.profile)
        print(f"Configuration imported as profile: {profile}")
        
    except Exception as e:
        print(f"Error importing configuration: {e}")
        sys.exit(1)


def config_hot_reload_command(args: argparse.Namespace) -> None:
    """Handle config hot-reload command."""
    try:
        config_manager = ConfigManager()
        
        if args.action == 'start':
            config_manager.start_hot_reload()
            print("Configuration hot-reload started. Press Ctrl+C to stop.")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                config_manager.stop_hot_reload()
                print("\nHot-reload stopped.")
        elif args.action == 'stop':
            config_manager.stop_hot_reload()
            print("Configuration hot-reload stopped.")
        else:
            print("Invalid action. Use 'start' or 'stop'.")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error managing hot-reload: {e}")
        sys.exit(1)


def add_config_subcommands(parser: argparse.ArgumentParser) -> None:
    """Add configuration subcommands to parser."""
    # This function is deprecated - use add_config_subparsers instead
    # Keeping for backward compatibility
    add_config_subparsers(parser)


def add_config_subparsers(parser: argparse.ArgumentParser) -> None:
    """Add configuration subcommands to parser."""
    config_parser = parser.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config commands')
    
    # config list
    list_parser = config_subparsers.add_parser('list', help='List configuration profiles')
    list_parser.set_defaults(func=config_list_command)
    
    # config show
    show_parser = config_subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument('--profile', '-p', default='default', help='Profile to show')
    show_parser.add_argument('--all', action='store_true', help='Show all profiles')
    show_parser.set_defaults(func=config_show_command)
    
    # config edit
    edit_parser = config_subparsers.add_parser('edit', help='Edit configuration')
    edit_parser.add_argument('--profile', '-p', default='default', help='Profile to edit')
    edit_parser.add_argument('--key', help='Configuration key to get/set')
    edit_parser.add_argument('--value', help='Value to set')
    edit_parser.set_defaults(func=config_edit_command)
    
    # config create
    create_parser = config_subparsers.add_parser('create', help='Create new profile')
    create_parser.add_argument('profile', help='New profile name')
    create_parser.add_argument('--source', '-s', default='default', help='Source profile')
    create_parser.set_defaults(func=config_create_command)
    
    # config delete
    delete_parser = config_subparsers.add_parser('delete', help='Delete profile')
    delete_parser.add_argument('profile', help='Profile to delete')
    delete_parser.set_defaults(func=config_delete_command)
    
    # config validate
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--profile', '-p', default='default', help='Profile to validate')
    validate_parser.set_defaults(func=config_validate_command)
    
    # config wizard
    wizard_parser = config_subparsers.add_parser('wizard', help='Run configuration wizard')
    wizard_parser.set_defaults(func=config_wizard_command)
    
    # config export
    export_parser = config_subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('--profile', '-p', default='default', help='Profile to export')
    export_parser.add_argument('output', help='Output file path')
    export_parser.set_defaults(func=config_export_command)
    
    # config import
    import_parser = config_subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('file', help='Configuration file to import')
    import_parser.add_argument('--profile', '-p', help='Profile name for imported config')
    import_parser.set_defaults(func=config_import_command)
    
    # config hot-reload
    hotreload_parser = config_subparsers.add_parser('hot-reload', help='Manage configuration hot-reload')
    hotreload_parser.add_argument('action', choices=['start', 'stop'], help='Action to perform')
    hotreload_parser.set_defaults(func=config_hot_reload_command)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(profile: str = "default") -> ContextBoxConfig:
    """Get configuration for profile."""
    return get_config_manager().load_config(profile)


def set_global_config(config: ContextBoxConfig) -> None:
    """Set global configuration."""
    get_config_manager().config = config
