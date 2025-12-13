"""
Utility functions for ContextBox.
"""

import logging
import json
import os
import platform
from typing import Dict, Any, Optional


def setup_logging(level: str = 'INFO') -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    handlers = [logging.StreamHandler()]
    log_path = os.environ.get('CONTEXTBOX_LOG_PATH')
    if not log_path:
        log_dir = os.path.join(get_app_data_dir(), 'logs')
        log_path = os.path.join(log_dir, 'contextbox.log')
    
    try:
        ensure_directory(os.path.dirname(log_path))
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    except (OSError, IOError) as exc:
        fallback_logger = logging.getLogger(__name__)
        fallback_logger.warning(
            "Unable to create log file at %s (%s). Continuing with console logging only.",
            log_path,
            exc
        )
    
    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_platform_info() -> Dict[str, str]:
    """
    Get platform-specific information.
    
    Returns:
        Dictionary containing platform information
    """
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'node': platform.node(),
        'platform': platform.platform()
    }


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_directory(dirpath: str) -> None:
    """
    Ensure directory exists, create if needed.
    
    Args:
        dirpath: Directory path
    """
    os.makedirs(dirpath, exist_ok=True)


def format_timestamp(dt) -> str:
    """
    Format datetime object to ISO string.
    
    Args:
        dt: Datetime object
        
    Returns:
        ISO formatted timestamp string
    """
    return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for cross-platform compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = 'unnamed'
    
    return filename


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == 'windows'


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system().lower() == 'darwin'


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system().lower() == 'linux'


def get_home_directory() -> str:
    """
    Get user's home directory.
    
    Returns:
        Path to home directory
    """
    return os.path.expanduser('~')


def get_app_data_dir() -> str:
    """
    Get application data directory.
    
    Returns:
        Path to application data directory
    """
    home = get_home_directory()
    
    if is_windows():
        return os.path.join(home, 'AppData', 'Roaming', 'ContextBox')
    elif is_macos():
        return os.path.join(home, 'Library', 'Application Support', 'ContextBox')
    else:
        return os.path.join(home, '.config', 'contextbox')
