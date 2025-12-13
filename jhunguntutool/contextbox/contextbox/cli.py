"""
Command-line interface for ContextBox.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from .main import ContextBox
from .utils import load_config, get_platform_info, ensure_directory
from .config import (
    ContextBoxConfig, ConfigManager, ConfigWizard, add_config_subparsers,
    get_config, get_config_manager
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="ContextBox - Capture and organize digital context"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='ContextBox 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Configuration subcommands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config commands')
    
    # config list
    list_parser = config_subparsers.add_parser('list', help='List configuration profiles')
    
    # config show
    show_parser = config_subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument('--profile', '-p', default='default', help='Profile to show')
    show_parser.add_argument('--all', action='store_true', help='Show all profiles')
    
    # config edit
    edit_parser = config_subparsers.add_parser('edit', help='Edit configuration')
    edit_parser.add_argument('--profile', '-p', default='default', help='Profile to edit')
    edit_parser.add_argument('--key', help='Configuration key to get/set (dot notation)')
    edit_parser.add_argument('--value', help='Value to set')
    
    # config create
    create_parser = config_subparsers.add_parser('create', help='Create new profile')
    create_parser.add_argument('profile', help='New profile name')
    create_parser.add_argument('--source', '-s', default='default', help='Source profile')
    
    # config delete
    delete_parser = config_subparsers.add_parser('delete', help='Delete profile')
    delete_parser.add_argument('profile', help='Profile to delete')
    
    # config validate
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--profile', '-p', default='default', help='Profile to validate')
    
    # config wizard
    wizard_parser = config_subparsers.add_parser('wizard', help='Run configuration wizard')
    
    # config export
    export_parser = config_subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('--profile', '-p', default='default', help='Profile to export')
    export_parser.add_argument('output', help='Output file path')
    
    # config import
    import_parser = config_subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('file', help='Configuration file to import')
    import_parser.add_argument('--profile', '-p', help='Profile name for imported config')
    
    # config hot-reload
    hotreload_parser = config_subparsers.add_parser('hot-reload', help='Manage configuration hot-reload')
    hotreload_parser.add_argument('action', choices=['start', 'stop'], help='Action to perform')
    
    # Main capture command (the primary command as requested)
    capture_parser = subparsers.add_parser('capture', help='Capture screenshot and extract context')
    capture_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for capture results (JSON format)'
    )
    capture_parser.add_argument(
        '--artifact-dir', '-a',
        type=str,
        default='artifacts',
        help='Directory to save artifacts (screenshots, etc.)'
    )
    capture_parser.add_argument(
        '--no-screenshot',
        action='store_true',
        help='Skip taking screenshot'
    )
    capture_parser.add_argument(
        '--extract-text',
        action='store_true',
        help='Extract text content'
    )
    capture_parser.add_argument(
        '--extract-urls',
        action='store_true',
        help='Extract URLs from content'
    )
    
    # Start capture command
    start_parser = subparsers.add_parser('start', help='Start context capture')
    
    # Stop capture command  
    stop_parser = subparsers.add_parser('stop', help='Stop context capture')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract context from data')
    extract_parser.add_argument(
        'data_file',
        type=str,
        help='Path to data file for extraction'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query stored context')
    query_parser.add_argument(
        'context_id',
        type=str,
        help='Context identifier to retrieve'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List stored contexts')
    
    # Extract content command - manual content extraction
    extract_content_parser = subparsers.add_parser('extract-content', help='Manually extract content from data')
    extract_content_parser.add_argument(
        'input_file',
        type=str,
        help='Path to input file (text, JSON, or image)'
    )
    extract_content_parser.add_argument(
        '--type', '-t',
        choices=['auto', 'text', 'image', 'json'],
        default='auto',
        help='Input data type (auto-detect by default)'
    )
    extract_content_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for extraction results'
    )
    extract_content_parser.add_argument(
        '--format', '-f',
        choices=['json', 'pretty', 'summary', 'detailed'],
        default='json',
        help='Output format'
    )
    extract_content_parser.add_argument(
        '--extract-urls',
        action='store_true',
        default=True,
        help='Extract URLs from content'
    )
    extract_content_parser.add_argument(
        '--no-extract-urls',
        dest='extract_urls',
        action='store_false',
        help='Do not extract URLs'
    )
    extract_content_parser.add_argument(
        '--extract-text',
        action='store_true',
        default=True,
        help='Extract and process text content'
    )
    extract_content_parser.add_argument(
        '--no-extract-text',
        dest='extract_text',
        action='store_false',
        help='Do not extract text content'
    )
    extract_content_parser.add_argument(
        '--extract-images',
        action='store_true',
        default=True,
        help='Extract and process images (OCR)'
    )
    extract_content_parser.add_argument(
        '--no-extract-images',
        dest='extract_images',
        action='store_false',
        help='Do not process images'
    )
    extract_content_parser.add_argument(
        '--config-extraction',
        type=str,
        help='Path to content extraction configuration file'
    )
    
    return parser


def start_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle start command."""
    print("Starting ContextBox capture...")
    app.start_capture()
    print("Context capture started. Press Ctrl+C to stop.")
    
    # Keep the process alive - the capture thread will run
    try:
        import time
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping capture...")
        app.stop_capture()
        print("Capture stopped.")


def stop_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle stop command."""
    print("Stopping ContextBox capture...")
    app.stop_capture()
    print("Context capture stopped.")


def extract_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle extract command."""
    print(f"Extracting context from: {args.data_file}")
    
    # Load data file (simplified - in real implementation would handle various formats)
    try:
        with open(args.data_file, 'r') as f:
            import json
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
    
    # Extract context
    context = app.extract_context(data)
    
    # Store context
    context_id = app.store_context(context)
    
    print(f"Context extracted and stored with ID: {context_id}")


def query_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle query command."""
    context = app.get_context(args.context_id)
    
    if context:
        import json
        print("Context found:")
        print(json.dumps(context, indent=2))
    else:
        print(f"Context with ID '{args.context_id}' not found.")


def list_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle list command."""
    print("Listing stored contexts...")
    try:
        # Query database for all contexts
        contexts = app.database.list_captures()
        
        if not contexts:
            print("No contexts found.")
            return
            
        # Display contexts in a nice format
        console = Console()
        table = Table(title="Stored Contexts")
        
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Created", style="magenta")
        table.add_column("Source Window", style="green")
        table.add_column("Screenshot", style="blue")
        table.add_column("Has Content", style="yellow")
        
        for context in contexts:
            screenshot = "Yes" if context.get('screenshot_path') else "No"
            has_content = "Yes" if context.get('clipboard_text') or context.get('notes') else "No"
            source_window = context.get('source_window') or "Unknown"
            
            table.add_row(
                str(context['id']),
                context['created_at'],
                source_window,
                screenshot,
                has_content
            )
        
        console.print(table)
        print(f"\nTotal: {len(contexts)} contexts")
        
    except Exception as e:
        print(f"Error listing contexts: {e}")


def extract_content_command(args: argparse.Namespace, app: ContextBox) -> None:
    """Handle extract-content command - manual content extraction."""
    print(f"ContextBox Content Extraction - Processing: {args.input_file}")
    
    try:
        # Load input data
        input_data = load_input_data(args.input_file, args.type)
        if not input_data:
            print(f"Error: Could not load or parse input file: {args.input_file}")
            sys.exit(1)
        
        print(f"Loaded input data: {input_data.get('data_type', 'unknown')} ({len(str(input_data))} chars)")
        
        # Update extraction configuration if provided
        if args.config_extraction:
            try:
                with open(args.config_extraction, 'r') as f:
                    config_updates = json.load(f)
                app.update_content_extraction_config(config_updates)
                print(f"Updated extraction configuration from: {args.config_extraction}")
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
        
        # Perform manual content extraction
        print("Starting content extraction...")
        result = app.extract_content_manually(
            input_data=input_data['data'],
            extract_urls=args.extract_urls,
            extract_text=args.extract_text,
            extract_images=args.extract_images,
            output_format=args.format
        )
        
        # Display results
        print("\n" + "="*60)
        print("CONTENT EXTRACTION RESULTS")
        print("="*60)
        print(result)
        print("="*60)
        
        # Save to file if requested
        if args.output:
            output_path = args.output
            if not output_path.endswith(('.json', '.txt')):
                if args.format == 'json':
                    output_path += '.json'
                else:
                    output_path += '.txt'
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    if args.format == 'json':
                        # Parse JSON result for file writing
                        import json
                        result_dict = app.extract_content_manually(
                            input_data=input_data['data'],
                            extract_urls=args.extract_urls,
                            extract_text=args.extract_text,
                            extract_images=args.extract_images,
                            output_format='json'
                        )
                        # result is a string, need to get the dict version
                        result_for_file = json.loads(result)
                        json.dump(result_for_file, f, indent=2, ensure_ascii=False)
                    else:
                        f.write(result)
                
                print(f"\nResults saved to: {output_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save results: {e}")
        
        print(f"\n✅ Content extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Content extraction failed: {e}")
        logging.error(f"Content extraction error: {e}", exc_info=True)
        sys.exit(1)


def load_input_data(file_path: str, data_type: str) -> Optional[Dict[str, Any]]:
    """
    Load input data from file with type detection.
    
    Args:
        file_path: Path to input file
        data_type: Expected data type ('auto', 'text', 'image', 'json')
        
    Returns:
        Dictionary with data and type information
    """
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            print(f"Error: File not found: {file_path}")
            return None
        
        # Auto-detect type if requested
        if data_type == 'auto':
            suffix = file_path_obj.suffix.lower()
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                data_type = 'image'
            elif suffix == '.json':
                data_type = 'json'
            else:
                data_type = 'text'
        
        if data_type == 'text':
            # Load as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'data_type': 'text',
                'data': {'text': content},
                'source_file': file_path
            }
        
        elif data_type == 'json':
            # Load as JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            return {
                'data_type': 'json',
                'data': content,
                'source_file': file_path
            }
        
        elif data_type == 'image':
            # Load as image file
            return {
                'data_type': 'image',
                'data': {'image_path': file_path},
                'source_file': file_path
            }
        
        else:
            print(f"Error: Unsupported data type: {data_type}")
            return None
            
    except Exception as e:
        print(f"Error loading input file: {e}")
        return None


def config_list_command(args: argparse.Namespace) -> None:
    """Handle config list command."""
    from .config import get_config_manager
    config_manager = get_config_manager()
    profiles = config_manager.list_profiles()
    
    console = Console()
    table = Table(title="Configuration Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Status", style="green")
    
    for profile in profiles:
        status = "Current" if profile == "default" else "Available"
        table.add_row(profile, status)
    
    console.print(table)


def config_show_command(args: argparse.Namespace) -> None:
    """Handle config show command."""
    from .config import get_config_manager
    from rich.syntax import Syntax
    from rich.panel import Panel
    
    config_manager = get_config_manager()
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


def config_edit_command(args: argparse.Namespace) -> None:
    """Handle config edit command."""
    from .config import get_config_manager
    from rich.prompt import Prompt, Confirm
    
    config_manager = get_config_manager()
    profile = getattr(args, 'profile', 'default')
    
    config = config_manager.load_config(profile)
    
    if args.key and args.value:
        # Set specific value
        try:
            config_manager.set_config_value(args.key, args.value)
            print(f"Set {args.key} = {args.value}")
        except Exception as e:
            print(f"Error setting value: {e}")
            sys.exit(1)
    elif args.key:
        # Get specific value
        try:
            value = config_manager.get_config_value(args.key)
            print(f"{args.key} = {value}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive editing
        console = Console()
        console.print("[yellow]Interactive editing not fully implemented[/yellow]")
        console.print("Use --key and --value to set specific values.")
        console.print("\nAvailable sections:")
        sections = ['database', 'capture', 'content_extraction', 'llm', 'logging', 'ui', 'security']
        for section in sections:
            console.print(f"  • {section}")


def config_create_command(args: argparse.Namespace) -> None:
    """Handle config create command."""
    from .config import get_config_manager
    
    config_manager = get_config_manager()
    source_profile = getattr(args, 'source', 'default')
    new_profile = args.profile
    
    try:
        profile_name = config_manager.create_profile(source_profile, new_profile)
        print(f"Created profile: {profile_name}")
    except Exception as e:
        print(f"Error creating profile: {e}")
        sys.exit(1)


def config_delete_command(args: argparse.Namespace) -> None:
    """Handle config delete command."""
    from .config import get_config_manager
    from rich.prompt import Confirm
    
    config_manager = get_config_manager()
    profile = args.profile
    
    if profile == "default":
        print("Error: Cannot delete default profile")
        sys.exit(1)
    
    if Confirm.ask(f"Delete profile '{profile}'? This cannot be undone."):
        try:
            config_manager.delete_profile(profile)
            print(f"Deleted profile: {profile}")
        except Exception as e:
            print(f"Error deleting profile: {e}")
            sys.exit(1)
    else:
        print("Deletion cancelled")


def config_validate_command(args: argparse.Namespace) -> None:
    """Handle config validate command."""
    from .config import get_config_manager, ConfigValidator
    
    config_manager = get_config_manager()
    profile = getattr(args, 'profile', 'default')
    
    try:
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
    from .config import get_config_manager, ConfigWizard
    
    config_manager = get_config_manager()
    wizard = ConfigWizard(config_manager)
    
    try:
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
    from .config import get_config_manager
    
    config_manager = get_config_manager()
    profile = getattr(args, 'profile', 'default')
    
    try:
        config_manager.export_config(profile, args.output)
        print(f"Configuration exported to: {args.output}")
    except Exception as e:
        print(f"Error exporting configuration: {e}")
        sys.exit(1)


def config_import_command(args: argparse.Namespace) -> None:
    """Handle config import command."""
    from .config import get_config_manager, ConfigWizard
    
    config_manager = get_config_manager()
    
    try:
        profile = config_manager.import_config(args.file, args.profile)
        print(f"Configuration imported as profile: {profile}")
    except Exception as e:
        print(f"Error importing configuration: {e}")
        sys.exit(1)


def config_hot_reload_command(args: argparse.Namespace) -> None:
    """Handle config hot-reload command."""
    from .config import get_config_manager
    
    config_manager = get_config_manager()
    
    try:
        if args.action == 'start':
            config_manager.start_hot_reload()
            print("Configuration hot-reload started. Press Ctrl+C to stop.")
            try:
                import time
                while True:
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
    """Handle capture command - take screenshot and extract context."""
    print("ContextBox Capture - Taking screenshot and extracting context...")
    
    try:
        # Create artifact directory
        ensure_directory(args.artifact_dir)
        
        # Initialize capture data structure
        capture_data = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'platform': get_platform_info(),
            'artifacts': {},
            'extracted': {},
            'status': 'in_progress'
        }
        
        # Take screenshot if not disabled
        screenshot_path = None
        if not args.no_screenshot:
            print("Taking screenshot...")
            screenshot_path = take_screenshot(args.artifact_dir)
            if screenshot_path:
                capture_data['artifacts']['screenshot'] = screenshot_path
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Warning: Could not take screenshot")
        
        # Extract text if requested
        if args.extract_text or True:  # Always extract text for screenshots
            print("Extracting text content...")
            extracted_text = extract_text_from_screenshot(screenshot_path) if screenshot_path else extract_current_context()
            if extracted_text:
                capture_data['extracted']['text'] = extracted_text
                print(f"Extracted text ({len(extracted_text)} characters)")
        
        # Extract URLs if requested or if text contains potential URLs
        if args.extract_urls or 'text' in capture_data['extracted']:
            print("Extracting URLs...")
            urls = extract_urls_from_text(capture_data['extracted'].get('text', ''))
            capture_data['extracted']['urls'] = urls
            if urls:
                print(f"Found {len(urls)} URLs")
        
        # Store in database
        print("Storing context in database...")
        context_id = app.store_context(capture_data)
        capture_data['context_id'] = context_id
        capture_data['status'] = 'completed'
        
        # Generate output
        if args.output:
            output_path = args.output
            if not output_path.endswith('.json'):
                output_path += '.json'
        else:
            output_path = os.path.join(args.artifact_dir, f"capture_{context_id[:8]}.json")
        
        with open(output_path, 'w') as f:
            json.dump(capture_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Capture completed successfully!")
        print(f"Context ID: {context_id}")
        print(f"Results saved to: {output_path}")
        print(f"Artifacts directory: {args.artifact_dir}")
        
        # Print summary
        print("\n--- Capture Summary ---")
        print(f"Timestamp: {capture_data['timestamp']}")
        print(f"Platform: {capture_data['platform'].get('system', 'Unknown')}")
        
        if 'screenshot' in capture_data['artifacts']:
            print(f"Screenshot: ✓")
        
        if 'text' in capture_data['extracted']:
            text_len = len(capture_data['extracted']['text'])
            print(f"Extracted text: ✓ ({text_len} chars)")
        
        if 'urls' in capture_data['extracted']:
            url_count = len(capture_data['extracted']['urls'])
            print(f"Extracted URLs: ✓ ({url_count} found)")
        
        print(f"Database storage: ✓")
        
    except Exception as e:
        print(f"❌ Capture failed: {e}")
        logging.error(f"Capture error: {e}", exc_info=True)
        sys.exit(1)


def take_screenshot(artifact_dir: str) -> Optional[str]:
    """
    Take a screenshot of the current screen.
    
    Args:
        artifact_dir: Directory to save screenshot
        
    Returns:
        Path to screenshot file or None if failed
    """
    try:
        # Try different screenshot methods based on platform
        platform = sys.platform
        
        if platform == 'darwin':  # macOS
            import subprocess
            import tempfile
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(artifact_dir, f"screenshot_{timestamp}.png")
            
            # Use screencapture command on macOS
            subprocess.run([
                'screencapture', '-x', '-t', 'png', screenshot_path
            ], check=True, capture_output=True)
            
            return screenshot_path if os.path.exists(screenshot_path) else None
            
        elif platform == 'win32':  # Windows
            try:
                import pyautogui
                import datetime
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(artifact_dir, f"screenshot_{timestamp}.png")
                
                pyautogui.screenshot(screenshot_path)
                return screenshot_path if os.path.exists(screenshot_path) else None
            except ImportError:
                print("Warning: pyautogui not installed for Windows screenshots")
                return None
                
        elif platform.startswith('linux'):  # Linux
            try:
                import subprocess
                import datetime
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(artifact_dir, f"screenshot_{timestamp}.png")
                
                # Try scrot first, then gnome-screenshot, then flameshot
                commands = [
                    ['scrot', screenshot_path],
                    ['gnome-screenshot', '-f', screenshot_path],
                    ['flameshot', 'full', '-p', screenshot_path]
                ]
                
                for cmd in commands:
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        if os.path.exists(screenshot_path):
                            return screenshot_path
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                print("Warning: No screenshot tool found (tried: scrot, gnome-screenshot, flameshot)")
                return None
                
            except Exception as e:
                print(f"Warning: Could not take Linux screenshot: {e}")
                return None
        else:
            print(f"Warning: Screenshot not supported on platform: {platform}")
            return None
            
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None


def extract_text_from_screenshot(screenshot_path: Optional[str]) -> Optional[str]:
    """
    Extract text from screenshot using OCR.
    
    Args:
        screenshot_path: Path to screenshot
        
    Returns:
        Extracted text or None if failed
    """
    if not screenshot_path or not os.path.exists(screenshot_path):
        return None
    
    try:
        # Try pytesseract for OCR
        try:
            from PIL import Image
            import pytesseract
            
            image = Image.open(screenshot_path)
            text = pytesseract.image_to_string(image, lang='eng')
            
            if text.strip():
                return text.strip()
            else:
                print("Warning: No text found in screenshot")
                return None
                
        except ImportError:
            print("Warning: pytesseract or PIL not available for OCR")
            return None
        except Exception as e:
            print(f"Warning: OCR extraction failed: {e}")
            return None
            
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return None


def extract_current_context() -> str:
    """
    Extract current context information as text.
    
    Returns:
        Context information as text string
    """
    try:
        # Get basic context information
        context_parts = []
        
        # Add timestamp
        from datetime import datetime
        context_parts.append(f"Timestamp: {datetime.now().isoformat()}")
        
        # Add platform info
        platform_info = get_platform_info()
        context_parts.append(f"Platform: {platform_info.get('system', 'Unknown')} {platform_info.get('release', '')}")
        
        # Add current working directory
        context_parts.append(f"Working Directory: {os.getcwd()}")
        
        # Add environment variables (selected ones)
        important_env = ['USER', 'HOME', 'SHELL', 'PATH']
        for env_var in important_env:
            if env_var in os.environ:
                context_parts.append(f"{env_var}: {os.environ[env_var]}")
        
        return '\n'.join(context_parts)
        
    except Exception as e:
        return f"Error extracting context: {e}"


def extract_urls_from_text(text: str) -> list:
    """
    Extract URLs from text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of found URLs
    """
    if not text:
        return []
    
    import re
    
    # URL pattern
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    urls = url_pattern.findall(text)
    return list(set(urls))  # Remove duplicates


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config_dict = {}
    if args.config:
        # Legacy config file loading
        config_dict = load_config(args.config)
    else:
        # Use enhanced config manager
        try:
            config_manager = get_config_manager()
            # Load profile from args or use default
            profile = getattr(args, 'profile', 'default')
            config_obj = config_manager.load_config(profile)
            config_dict = config_obj.to_dict()
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            print("Using default configuration...")
            config_dict = {}
    
    # Override config with command line arguments
    config_dict['log_level'] = args.log_level
    
    # Initialize ContextBox
    app = ContextBox(config_dict)
    
    # Execute command
    try:
        if args.command == 'capture':
            capture_command(args, app)
        elif args.command == 'start':
            start_command(args, app)
        elif args.command == 'stop':
            stop_command(args, app)
        elif args.command == 'extract':
            extract_command(args, app)
        elif args.command == 'query':
            query_command(args, app)
        elif args.command == 'list':
            list_command(args, app)
        elif args.command == 'extract-content':
            extract_content_command(args, app)
        elif args.command == 'config':
            # Handle config commands
            config_command = getattr(args, 'config_command', None)
            if config_command == 'list':
                config_manager = get_config_manager()
                profiles = config_manager.list_profiles()
                print("Available profiles:")
                for profile in profiles:
                    print(f"  - {profile}")
            elif config_command == 'show':
                profile = getattr(args, 'profile', 'default')
                config_manager = get_config_manager()
                config = config_manager.load_config(profile)
                import json
                print(json.dumps(config.to_dict(), indent=2))
            elif config_command == 'validate':
                profile = getattr(args, 'profile', 'default')
                config_manager = get_config_manager()
                config = config_manager.load_config(profile)
                validator = config_module.ConfigValidator()
                if validator.validate_config(config):
                    print(f"✓ Configuration '{profile}' is valid")
                else:
                    print(f"✗ Configuration '{profile}' has errors:")
                    for error in validator.errors:
                        print(f"  Error: {error}")
                    sys.exit(1)
            elif config_command == 'create':
                config_manager = get_config_manager()
                new_profile = args.profile
                source_profile = getattr(args, 'source', 'default')
                profile_name = config_manager.create_profile(source_profile, new_profile)
                print(f"Created profile: {profile_name}")
            elif config_command == 'create':
                config_manager = get_config_manager()
                new_profile = args.profile
                source_profile = getattr(args, 'source', 'default')
                profile_name = config_manager.create_profile(source_profile, new_profile)
                print(f"Created profile: {profile_name}")
            elif config_command == 'delete':
                config_manager = get_config_manager()
                profile = args.profile
                if profile == "default":
                    print("Error: Cannot delete default profile")
                    sys.exit(1)
                if Confirm.ask(f"Delete profile '{profile}'? This cannot be undone."):
                    config_manager.delete_profile(profile)
                    print(f"Deleted profile: {profile}")
                else:
                    print("Deletion cancelled")
            elif config_command == 'edit':
                config_manager = get_config_manager()
                profile = getattr(args, 'profile', 'default')
                config = config_manager.load_config(profile)
                
                if args.key and args.value:
                    # Set specific value
                    try:
                        config_manager.set_config_value(args.key, args.value)
                        print(f"Set {args.key} = {args.value}")
                    except Exception as e:
                        print(f"Error setting value: {e}")
                        sys.exit(1)
                elif args.key:
                    # Get specific value
                    try:
                        value = config_manager.get_config_value(args.key)
                        print(f"{args.key} = {value}")
                    except KeyError as e:
                        print(f"Error: {e}")
                        sys.exit(1)
                else:
                    # Interactive editing not fully implemented
                    print("Use --key and --value to set specific values.")
                    print("Example: --key database.path --value new_path.db")
            elif config_command == 'export':
                config_manager = get_config_manager()
                profile = getattr(args, 'profile', 'default')
                config_manager.export_config(profile, args.output)
                print(f"Configuration exported to: {args.output}")
            elif config_command == 'import':
                config_manager = get_config_manager()
                profile = config_manager.import_config(args.file, args.profile)
                print(f"Configuration imported as profile: {profile}")
            elif config_command == 'hot-reload':
                config_manager = get_config_manager()
                if args.action == 'start':
                    config_manager.start_hot_reload()
                    print("Configuration hot-reload started. Press Ctrl+C to stop.")
                    try:
                        import time
                        while True:
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
            elif config_command == 'wizard':
                print("Configuration wizard - Run interactively for first-time setup")
                config_manager = get_config_manager()
                wizard = ConfigWizard(config_manager)
                config = wizard.run()
                print(f"Configuration wizard completed. Profile '{config.profile}' created.")
            else:
                parser.print_help()
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()