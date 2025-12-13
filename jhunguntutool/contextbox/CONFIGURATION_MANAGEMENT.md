# ContextBox Configuration Management

ContextBox includes a comprehensive configuration management system that handles loading, validation, and management of settings with support for multiple profiles and hot-reloading.

## Quick Start

```bash
# Run the configuration wizard for first-time setup
contextbox config wizard

# List all configuration profiles
contextbox config list

# Show current configuration
contextbox config show

# Create a new profile from default
contextbox config create production --source default

# Set a specific configuration value
contextbox config edit --key database.path --value "/opt/contextbox/data.db"

# Validate configuration
contextbox config validate

# Export configuration to file
contextbox config export my_config.json

# Import configuration from file
contextbox config import my_config.json --profile imported
```

## Configuration Profiles

The system supports multiple configuration profiles, allowing you to maintain different settings for different environments:

- **default**: Default configuration profile
- **development**: Settings optimized for development
- **production**: Settings optimized for production use

### Profile Management

```bash
# Create new profile
contextbox config create my-profile --source default

# Delete profile (except default)
contextbox config delete my-profile

# Switch to different profile (affects subsequent operations)
contextbox config show --profile production
```

## Configuration Structure

The configuration is organized into several subsystems:

### Database Configuration
- `database.path`: Database file path
- `database.backup_enabled`: Enable automatic backups
- `database.backup_interval`: Backup frequency (hours)
- `database.cache_size`: Database cache size
- `database.timeout`: Connection timeout

### Capture Configuration
- `capture.screenshot_enabled`: Enable screenshot capture
- `capture.clipboard_monitor`: Monitor clipboard changes
- `capture.artifact_retention`: How long to keep artifacts (days)
- `capture.compression_enabled`: Compress stored artifacts

### Content Extraction Configuration
- `content_extraction.enabled`: Enable content extraction
- `content_extraction.extract_text`: Extract text from content
- `content_extraction.extract_urls`: Extract URLs
- `content_extraction.ocr_enabled`: Enable OCR for images
- `content_extraction.ocr_languages`: OCR languages
- `content_extraction.rate_limit_delay`: Delay between requests

### LLM Configuration
- `llm.provider`: LLM provider (openai, anthropic, local, none)
- `llm.api_key`: API key for external providers
- `llm.model`: Model name
- `llm.temperature`: Model temperature
- `llm.max_tokens`: Maximum tokens per request

### Logging Configuration
- `logging.level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file_enabled`: Log to file
- `logging.file_path`: Log file path
- `logging.console_enabled`: Log to console
- `logging.structured_logging`: Use structured logging

### UI Configuration
- `ui.theme`: UI theme (dark, light)
- `ui.language`: Interface language
- `ui.auto_start`: Start application automatically
- `ui.minimize_to_tray`: Minimize to system tray

### Security Configuration
- `security.encrypt_database`: Encrypt database files
- `security.audit_logging`: Enable audit logging
- `security.max_file_size`: Maximum file size (MB)
- `security.allowed_file_types`: Allowed file extensions

## Configuration Wizard

The interactive configuration wizard helps set up ContextBox for the first time:

```bash
contextbox config wizard
```

The wizard will guide you through:
1. Basic settings (profile name, directories)
2. Database configuration
3. Capture settings
4. Content extraction options
5. LLM provider setup
6. Logging preferences
7. Security settings

## Configuration Validation

The system validates configuration values and provides helpful error messages:

```bash
# Validate current configuration
contextbox config validate

# Validate specific profile
contextbox config validate --profile production
```

Validation checks include:
- Directory accessibility and permissions
- File path validity
- Numeric ranges and constraints
- Provider-specific requirements
- Security setting consistency

## Hot-Reload

Configuration hot-reloading allows changes to take effect without restarting ContextBox:

```bash
# Start hot-reload monitoring
contextbox config hot-reload start

# Stop hot-reload monitoring
contextbox config hot-reload stop
```

When hot-reload is active, configuration file changes are automatically detected and applied.

## Configuration File Locations

Configuration files are stored in:
- **Default**: `~/.contextbox/config/config.json`
- **Profiles**: `~/.contextbox/config/profiles/{profile}.json`
- **Custom**: Specify with `--config` flag

## Advanced Usage

### Setting Configuration Values

```bash
# Set database path
contextbox config edit --key database.path --value "/new/path/contextbox.db"

# Enable debug mode
contextbox config edit --key debug --value true

# Set LLM temperature
contextbox config edit --key llm.temperature --value 0.8
```

### Getting Configuration Values

```bash
# Get specific value
contextbox config edit --key database.path

# Show all configuration
contextbox config show

# Show specific profile
contextbox config show --profile production
```

### Configuration Export/Import

```bash
# Export configuration
contextbox config export --profile production production_config.json

# Import configuration
contextbox config import production_config.json --profile production-backup
```

## Environment Variables

You can override configuration values with environment variables:

- `CONTEXTBOX_CONFIG_PATH`: Custom configuration file path
- `CONTEXTBOX_PROFILE`: Configuration profile to use
- `CONTEXTBOX_DEBUG`: Enable debug mode
- `CONTEXTBOX_LOG_LEVEL`: Override log level

## Troubleshooting

### Common Issues

**Configuration validation fails:**
- Check directory permissions
- Verify file paths exist
- Ensure numeric values are in valid ranges
- Check API keys for external providers

**Hot-reload not working:**
- Ensure configuration files are in the correct directory
- Check file permissions
- Verify file system supports file watching

**Profile not found:**
- Check profile exists with `contextbox config list`
- Verify profile file exists in the profiles directory
- Use `contextbox config create` to create missing profiles

### Debug Configuration

Enable debug logging to troubleshoot configuration issues:

```bash
contextbox config edit --key debug --value true
contextbox config edit --key logging.level --value DEBUG
```

### Reset to Defaults

To reset configuration to defaults:

```bash
# Remove all profiles except default
contextbox config delete development
contextbox config delete production

# Run wizard to reconfigure
contextbox config wizard
```

## Examples

### Development Profile

```bash
# Create development profile with optimized settings
contextbox config create development --source default

# Configure for development
contextbox config edit --profile development --key debug --value true
contextbox config edit --profile development --key database.backup_enabled --value false
contextbox config edit --profile development --key logging.level --value DEBUG
contextbox config edit --profile development --key llm.provider --value local
```

### Production Profile

```bash
# Create production profile with secure settings
contextbox config create production --source default

# Configure for production
contextbox config edit --profile production --key security.encrypt_database --value true
contextbox config edit --profile production --key security.audit_logging --value true
contextbox config edit --profile production --key logging.level --value WARNING
contextbox config edit --profile production --key capture.artifact_retention --value 7
```

### Quick Setup for Different Scenarios

```bash
# Minimal configuration for testing
contextbox config edit --key capture.screenshot_enabled --value false
contextbox config edit --key content_extraction.enabled --value false
contextbox config edit --key llm.provider --value none

# Full-featured configuration
contextbox config edit --key capture.screenshot_enabled --value true
contextbox config edit --key capture.clipboard_monitor --value true
contextbox config edit --key content_extraction.enabled --value true
contextbox config edit --key content_extraction.ocr_enabled --value true
contextbox config edit --key llm.provider --value openai
```

This configuration system provides a flexible and powerful way to manage ContextBox settings across different environments and use cases.