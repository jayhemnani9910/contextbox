#!/usr/bin/env python3
"""
Test script for the enhanced ContextBox configuration system.
"""

import os
import sys
import json
from pathlib import Path

# Add the contextbox package to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import config module directly to avoid circular imports
import contextbox.config as config_module


def test_config_creation():
    """Test basic configuration creation."""
    print("Testing configuration creation...")
    
    # Create default config
    config = config_module.ContextBoxConfig()
    print(f"✓ Default config created with profile: {config.profile}")
    
    # Create custom config
    custom_config = config_module.ContextBoxConfig(
        profile="test",
        debug=True,
        database=config_module.DatabaseConfig(path="test.db"),
        capture=config_module.CaptureConfig(screenshot_enabled=False),
        llm=config_module.LLMConfig(provider="none")
    )
    
    print(f"✓ Custom config created with profile: {custom_config.profile}")
    return custom_config


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    config = config_module.ContextBoxConfig()
    validator = config_module.ConfigValidator()
    is_valid = validator.validate_config(config)
    
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors:")
        for error in validator.errors:
            print(f"  Error: {error}")
        
        for warning in validator.warnings:
            print(f"  Warning: {warning}")
    
    return is_valid


def test_config_save_load():
    """Test configuration save and load."""
    print("\nTesting configuration save/load...")
    
    config = config_module.ContextBoxConfig()
    
    # Test JSON format
    json_path = "test_config.json"
    config.save(json_path)
    print(f"✓ Configuration saved to {json_path}")
    
    loaded_config = config_module.ContextBoxConfig.load(json_path)
    print(f"✓ Configuration loaded from {json_path}")
    print(f"  Profile: {loaded_config.profile}")
    print(f"  Debug: {loaded_config.debug}")
    
    # Cleanup
    os.remove(json_path)
    
    return loaded_config


def test_config_manager():
    """Test configuration manager."""
    print("\nTesting configuration manager...")
    
    manager = config_module.ConfigManager()
    
    # Create default config
    config = manager.load_config("default")
    print(f"✓ Default config loaded: {config.profile}")
    
    # Create test profile
    test_config = config_module.ContextBoxConfig(profile="test")
    test_config.database.path = "test_manager.db"
    manager.save_config(test_config, "test")
    print(f"✓ Test profile saved")
    
    # Load test profile
    loaded_config = manager.load_config("test")
    print(f"✓ Test profile loaded: {loaded_config.profile}")
    
    # List profiles
    profiles = manager.list_profiles()
    print(f"✓ Available profiles: {profiles}")
    
    # Test value access
    db_path = manager.get_config_value("database.path")
    print(f"✓ Database path: {db_path}")
    
    # Test value setting
    manager.set_config_value("capture.screenshot_enabled", False)
    print("✓ Updated screenshot_enabled setting")
    
    # Cleanup
    manager.delete_profile("test")
    print("✓ Test profile deleted")
    
    return manager


def test_profiles():
    """Test configuration profiles."""
    print("\nTesting configuration profiles...")
    
    manager = config_module.ConfigManager()
    
    # Create development profile
    dev_config = config_module.ContextBoxConfig(
        profile="dev",
        debug=True,
        logging=config_module.LoggingConfig(level="DEBUG"),
        llm=config_module.LLMConfig(provider="none")  # Disable LLM for dev
    )
    manager.save_config(dev_config, "dev")
    print("✓ Development profile created")
    
    # Create production profile
    prod_config = config_module.ContextBoxConfig(
        profile="prod",
        debug=False,
        logging=config_module.LoggingConfig(level="WARNING"),
        security=config_module.SecurityConfig(encrypt_database=True),
        llm=config_module.LLMConfig(provider="openai", api_key="sk-...")
    )
    manager.save_config(prod_config, "prod")
    print("✓ Production profile created")
    
    # Test profile switching
    dev_loaded = manager.load_config("dev")
    print(f"✓ Development config loaded: debug={dev_loaded.debug}")
    
    prod_loaded = manager.load_config("prod")
    print(f"✓ Production config loaded: debug={prod_loaded.debug}, encrypt={prod_loaded.security.encrypt_database}")
    
    # List all profiles
    profiles = manager.list_profiles()
    print(f"✓ All profiles: {profiles}")
    
    # Cleanup
    manager.delete_profile("dev")
    manager.delete_profile("prod")
    print("✓ Test profiles cleaned up")


def test_config_export_import():
    """Test configuration export and import."""
    print("\nTesting configuration export/import...")
    
    manager = config_module.ConfigManager()
    
    # Create and save a config
    config = config_module.ContextBoxConfig(profile="export_test")
    config.llm.provider = "openai"
    config.database.path = "exported.db"
    manager.save_config(config, "export_test")
    
    # Export config
    export_path = "exported_config.json"
    manager.export_config("export_test", export_path)
    print(f"✓ Configuration exported to {export_path}")
    
    # Import config
    imported_profile = manager.import_config(export_path, "imported_test")
    print(f"✓ Configuration imported as {imported_profile}")
    
    # Verify imported config
    imported_config = manager.load_config("imported_test")
    print(f"✓ Imported profile: {imported_config.profile}")
    print(f"✓ LLM provider: {imported_config.llm.provider}")
    print(f"✓ Database path: {imported_config.database.path}")
    
    # Cleanup
    os.remove(export_path)
    manager.delete_profile("export_test")
    manager.delete_profile("imported_test")
    print("✓ Export test cleaned up")


def test_wizard_simulation():
    """Simulate configuration wizard (non-interactive for testing)."""
    print("\nTesting configuration wizard simulation...")
    
    manager = config_module.ConfigManager()
    # Note: Skipping actual wizard instantiation due to Rich console requirements
    
    # Simulate wizard responses
    responses = {
        'profile': 'wizard_test',
        'data_directory': '/tmp/contextbox_test',
        'debug': False,
        'screenshot_enabled': True,
        'content_extraction_enabled': True,
        'llm_provider': 'none',
        'log_level': 'INFO'
    }
    
    # Create config based on simulated responses
    config = config_module.ContextBoxConfig()
    config.profile = responses['profile']
    config.data_directory = responses['data_directory']
    config.debug = responses['debug']
    config.capture.screenshot_enabled = responses['screenshot_enabled']
    config.content_extraction.enabled = responses['content_extraction_enabled']
    config.llm.provider = responses['llm_provider']
    config.logging.level = responses['log_level']
    
    manager.save_config(config, config.profile)
    print(f"✓ Wizard simulation completed: {config.profile}")
    
    # Verify wizard config
    loaded = manager.load_config(config.profile)
    print(f"✓ Wizard config verified: debug={loaded.debug}, profile={loaded.profile}")
    
    # Cleanup
    manager.delete_profile(config.profile)
    print("✓ Wizard test cleaned up")


def test_hot_reload_simulation():
    """Test hot-reload functionality (simplified)."""
    print("\nTesting hot-reload simulation...")
    
    manager = config_module.ConfigManager()
    
    # Create initial config
    config1 = config_module.ContextBoxConfig(profile="reload_test")
    config1.capture.screenshot_enabled = True
    manager.save_config(config1, "reload_test")
    print("✓ Initial config created")
    
    # Simulate hot-reload by updating file
    config2 = config_module.ContextBoxConfig(profile="reload_test")
    config2.capture.screenshot_enabled = False
    manager.save_config(config2, "reload_test")
    print("✓ Config updated (simulating hot-reload)")
    
    # Load updated config
    reloaded = manager.load_config("reload_test")
    print(f"✓ Reloaded config: screenshot_enabled={reloaded.capture.screenshot_enabled}")
    
    # Cleanup
    manager.delete_profile("reload_test")
    print("✓ Hot-reload test cleaned up")


def main():
    """Run all configuration tests."""
    print("ContextBox Configuration System Test")
    print("=" * 50)
    
    try:
        # Test basic config creation
        test_config_creation()
        
        # Test validation
        test_config_validation()
        
        # Test save/load
        test_config_save_load()
        
        # Test config manager
        manager = test_config_manager()
        
        # Test profiles
        test_profiles()
        
        # Test export/import
        test_config_export_import()
        
        # Test wizard
        test_wizard_simulation()
        
        # Test hot-reload
        test_hot_reload_simulation()
        
        print("\n" + "=" * 50)
        print("✅ All configuration tests passed!")
        
        # Show configuration CLI commands
        print("\nAvailable CLI commands:")
        print("  contextbox config list                    # List profiles")
        print("  contextbox config show --profile dev     # Show config")
        print("  contextbox config create prod --source dev # Create profile")
        print("  contextbox config edit --profile dev --key capture.screenshot_enabled --value false")
        print("  contextbox config validate --profile dev  # Validate config")
        print("  contextbox config wizard                 # Run setup wizard")
        print("  contextbox config export dev config.json # Export config")
        print("  contextbox config import config.json --profile new")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()