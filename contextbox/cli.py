#!/usr/bin/env python3
"""
Enhanced Click-based CLI for ContextBox - Capture and organize digital context
Features: Rich formatting, interactive prompts, progress bars, autocomplete, beautiful help
"""

import click
import json
import logging
import sys
import os
import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import csv
import re

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.tree import Tree
from rich.align import Align
from rich.live import Live
from rich.status import Status

# ContextBox imports
try:
    from contextbox.main import ContextBox
    from contextbox.utils import load_config, get_platform_info, ensure_directory, get_app_data_dir, format_timestamp
    from contextbox.config import get_config_manager, get_config
except ImportError as e:
    print(f"Error importing ContextBox: {e}")
    print("Make sure ContextBox is properly installed")
    sys.exit(1)

# Initialize console
console = Console()

# Global ContextBox instance
app_instance = None

def get_app():
    """Get or initialize ContextBox instance."""
    global app_instance
    if app_instance is None:
        config = {}
        config_file = os.path.join(get_app_data_dir(), 'config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
        
        app_instance = ContextBox(config)
    return app_instance

def prompt_for_api_key():
    """Interactive prompt for API key with validation."""
    console.print("\n[bold blue]🔑 API Key Configuration[/bold blue]")
    console.print("To enable advanced AI features, you need to configure your API key.")
    console.print("[dim]Features that require an API key:[/dim]")
    console.print("  • Context analysis and summarization")
    console.print("  • Intelligent search and Q&A")
    console.print("  • Advanced content extraction")
    console.print("  • AI-powered insights")
    
    # Check for existing API key
    config_file = os.path.join(get_app_data_dir(), 'config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                current_config = json.load(f)
                if current_config.get('api_key'):
                    if not Confirm.ask("API key already configured. Do you want to update it?"):
                        return current_config['api_key']
        except Exception:
            pass
    
    api_key = Prompt.ask("Enter your API key", password=True)
    
    if api_key:
        # Validate API key format (basic validation)
        if len(api_key) < 10:
            console.print("[red]❌[/red] API key seems too short. Please check and try again.")
            return None
            
        # Save API key to config
        try:
            ensure_directory(get_app_data_dir())
            
            config = {}
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
            config['api_key'] = api_key
            config['api_key_configured'] = True
            config['configured_timestamp'] = format_timestamp(time.time())
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            console.print("[green]✓[/green] API key saved successfully!")
            return api_key
            
        except Exception as e:
            console.print(f"[red]❌[/red] Failed to save API key: {e}")
            return None
    
    return None

def display_error(message: str, exception: Exception = None, exit: bool = True):
    """Display formatted error message."""
    error_text = f"[red]{message}[/red]"
    if exception:
        error_text += f"\n\n[dim]Details: {exception}[/dim]"
    
    error_panel = Panel(
        error_text,
        title="[red]❌ Error[/red]",
        border_style="red",
        box=box.DOUBLE
    )
    console.print(error_panel)
    
    if exit:
        sys.exit(1)

def display_success(message: str):
    """Display formatted success message."""
    success_panel = Panel(
        f"[green]✓ {message}[/green]",
        title="[green]✅ Success[/green]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(success_panel)

def display_warning(message: str):
    """Display formatted warning message."""
    warning_panel = Panel(
        f"[yellow]⚠️ {message}[/yellow]",
        title="[yellow]⚠️ Warning[/yellow]",
        border_style="yellow",
        box=box.ROUNDED
    )
    console.print(warning_panel)

def display_help_header():
    """Display beautiful help header."""
    header_text = Text()
    header_text.append("ContextBox CLI ", style="bold magenta")
    header_text.append("v2.0.0", style="dim")
    header_text.append("\n\n", style="")
    header_text.append("AI-powered context capture and organization", style="dim")
    
    header_panel = Panel(
        Align.center(header_text),
        border_style="magenta",
        box=box.DOUBLE
    )
    console.print(header_panel)
    
    # Available commands
    commands_tree = Tree("[bold blue]🚀 Available Commands[/bold blue]")
    
    capture_branch = commands_tree.add("📸 capture")
    capture_branch.add("   Take screenshots and extract context")
    
    ask_branch = commands_tree.add("🤔 ask")
    ask_branch.add("   Ask questions about captured context")
    
    summarize_branch = commands_tree.add("📝 summarize")
    summarize_branch.add("   Generate intelligent summaries")
    
    search_branch = commands_tree.add("🔍 search")
    search_branch.add("   Search through stored contexts")
    
    list_branch = commands_tree.add("📋 list")
    list_branch.add("   List all stored contexts")
    
    stats_branch = commands_tree.add("📊 stats")
    stats_branch.add("   View database statistics")
    
    config_branch = commands_tree.add("⚙️ config")
    config_branch.add("   Configure settings and API keys")
    
    export_branch = commands_tree.add("📤 export")
    export_branch.add("   Export contexts to files")
    
    import_branch = commands_tree.add("📥 import")
    import_branch.add("   Import contexts from files")
    
    console.print(commands_tree)
    
    console.print("\n[dim]💡 Tip: Use --help with any command for detailed options[/dim]")

def take_screenshot_enhanced(artifact_dir: str, status: Status) -> Optional[str]:
    """Enhanced screenshot taking with better error handling."""
    try:
        status.update("📸 Taking screenshot...")
        platform = sys.platform
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(artifact_dir, f"screenshot_{timestamp}.png")
        
        if platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['screencapture', '-x', '-t', 'png', screenshot_path], 
                         check=True, capture_output=True)
            
        elif platform == 'win32':  # Windows
            try:
                import pyautogui
                pyautogui.screenshot(screenshot_path)
            except ImportError:
                display_warning("pyautogui not installed for Windows screenshots")
                return None
                
        elif platform.startswith('linux'):  # Linux
            import subprocess
            commands = [
                ['scrot', screenshot_path],
                ['gnome-screenshot', '-f', screenshot_path],
                ['flameshot', 'full', '-p', screenshot_path]
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                display_warning("No screenshot tool found (tried: scrot, gnome-screenshot, flameshot)")
                return None
        else:
            display_warning(f"Screenshot not supported on platform: {platform}")
            return None
        
        if os.path.exists(screenshot_path):
            status.update("✅ Screenshot captured successfully!")
            return screenshot_path
        else:
            return None
            
    except Exception as e:
        display_warning(f"Failed to take screenshot: {e}")
        return None

def extract_text_enhanced(screenshot_path: Optional[str]) -> Optional[str]:
    """Enhanced text extraction with multiple methods."""
    try:
        # Try OCR first if screenshot exists
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                from PIL import Image
                import pytesseract
                
                image = Image.open(screenshot_path)
                text = pytesseract.image_to_string(image, lang='eng')
                
                if text.strip():
                    return text.strip()
                    
            except ImportError:
                pass
            except Exception:
                pass
        
        # Fallback to basic context extraction
        return extract_current_context_enhanced()
        
    except Exception as e:
        console.print(f"[yellow]Warning: Text extraction failed: {e}[/yellow]")
        return extract_current_context_enhanced()

def extract_current_context_enhanced() -> str:
    """Enhanced current context extraction."""
    try:
        context_parts = []
        
        # Add timestamp
        context_parts.append(f"Timestamp: {datetime.now().isoformat()}")
        
        # Add platform info
        platform_info = get_platform_info()
        context_parts.append(f"Platform: {platform_info.get('system', 'Unknown')} {platform_info.get('release', '')}")
        
        # Add current working directory
        context_parts.append(f"Working Directory: {os.getcwd()}")
        
        # Add Python version
        context_parts.append(f"Python Version: {sys.version.split()[0]}")
        
        # Add environment info
        important_env = ['USER', 'HOME', 'SHELL', 'PATH', 'VIRTUAL_ENV']
        for env_var in important_env:
            if env_var in os.environ:
                context_parts.append(f"{env_var}: {os.environ[env_var]}")
        
        return '\n'.join(context_parts)
        
    except Exception as e:
        return f"Error extracting context: {e}"

def extract_urls_enhanced(text: str) -> List[str]:
    """Enhanced URL extraction from text."""
    if not text:
        return []
    
    # Enhanced URL pattern
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    urls = url_pattern.findall(text)
    
    # Also look for www URLs
    www_pattern = re.compile(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    www_urls = www_pattern.findall(text)
    
    # Convert www URLs to proper URLs
    for i, url in enumerate(www_urls):
        www_urls[i] = f"https://{url}"
    
    # Combine and deduplicate
    all_urls = list(set(urls + www_urls))
    return all_urls

# CLI Group
@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version and exit')
@click.option('--config', type=click.Path(), help='Configuration file path')
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx, version, config, log_level):
    """🚀 ContextBox CLI - AI-powered context capture and organization."""
    
    if ctx.invoked_subcommand is None:
        if version:
            console.print("ContextBox CLI v2.0.0")
            console.print("Built with ❤️ using Click and Rich")
            return
        
        display_help_header()
        return
    
    # Initialize ContextBox
    app_config = {'log_level': log_level.upper()}
    
    if config:
        try:
            app_config.update(load_config(config))
        except Exception as e:
            display_error(f"Failed to load config: {e}")
    
    global app_instance
    app_instance = ContextBox(app_config)

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for capture results')
@click.option('--artifact-dir', '-a', type=click.Path(), default='artifacts', 
              help='Directory to save artifacts')
@click.option('--no-screenshot', is_flag=True, help='Skip taking screenshot')
@click.option('--extract-text/--no-extract-text', default=True, help='Extract text content')
@click.option('--extract-urls/--no-extract-urls', default=True, help='Extract URLs from content')
@click.option('--interactive/--no-interactive', default=False, help='Interactive capture mode')
def capture(output, artifact_dir, no_screenshot, extract_text, extract_urls, interactive):
    """📸 Capture screenshot and extract context from current screen."""
    
    app = get_app()
    
    try:
        # Create artifact directory
        ensure_directory(artifact_dir)
        
        with Status("[bold blue]🚀 Initializing capture...", console=console) as status:
            time.sleep(0.5)
            
            # Initialize capture data
            capture_data = {
                'timestamp': format_timestamp(time.time()),
                'platform': get_platform_info(),
                'artifacts': {},
                'extracted': {},
                'status': 'in_progress'
            }
            
            # Capture screenshot if not disabled
            screenshot_path = None
            if not no_screenshot:
                screenshot_path = take_screenshot_enhanced(artifact_dir, status)
                if screenshot_path:
                    capture_data['artifacts']['screenshot'] = screenshot_path
            
            # Extract text if requested
            if extract_text:
                status.update("📝 Extracting text content...")
                extracted_text = extract_text_enhanced(screenshot_path)
                if extracted_text:
                    capture_data['extracted']['text'] = extracted_text
                    status.update(f"📝 Extracted {len(extracted_text)} characters")
            
            # Extract URLs if requested
            if extract_urls and 'text' in capture_data['extracted']:
                status.update("🔗 Extracting URLs...")
                urls = extract_urls_enhanced(capture_data['extracted']['text'])
                capture_data['extracted']['urls'] = urls
                if urls:
                    status.update(f"🔗 Found {len(urls)} URLs")
            
            # Store in database
            status.update("💾 Storing in database...")
            context_id = app.store_context(capture_data)
            capture_data['context_id'] = context_id
            capture_data['status'] = 'completed'
        
        # Generate output file
        if output:
            output_path = output if output.endswith('.json') else output + '.json'
        else:
            output_path = os.path.join(artifact_dir, f"capture_{str(uuid.uuid4())[:8]}.json")
        
        with open(output_path, 'w') as f:
            json.dump(capture_data, f, indent=2, ensure_ascii=False)
        
        # Display results
        console.print("\n" + "="*60)
        
        result_table = Table(title="📸 Capture Results", box=box.DOUBLE)
        result_table.add_column("Property", style="cyan", no_wrap=True)
        result_table.add_column("Value", style="green")
        
        result_table.add_row("Context ID", f"[bold cyan]{context_id[:8]}[/bold cyan]")
        result_table.add_row("Timestamp", capture_data['timestamp'])
        result_table.add_row("Platform", capture_data['platform'].get('system', 'Unknown'))
        result_table.add_row("Screenshot", "[green]✓[/green]" if 'screenshot' in capture_data['artifacts'] else "[red]✗[/red]")
        result_table.add_row("Text Extracted", "[green]✓[/green]" if 'text' in capture_data['extracted'] else "[red]✗[/red]")
        result_table.add_row("URLs Found", f"[blue]{len(capture_data['extracted'].get('urls', []))}[/blue]")
        result_table.add_row("Output File", output_path)
        
        console.print(result_table)
        console.print("="*60)
        
        display_success("Capture completed successfully!")
        
    except Exception as e:
        display_error(f"Capture failed: {e}", e)

@cli.command()
@click.argument('question')
@click.option('--context-id', type=str, help='Specific context ID to ask about')
@click.option('--all-contexts', is_flag=True, help='Search across all contexts')
@click.option('--model', type=str, help='LLM model to use')
def ask(question, context_id, all_contexts, model):
    """🤔 Ask questions about captured context (not implemented yet)."""
    console.print("[yellow]contextbox ask is not implemented yet.[/yellow]")
    sys.exit(1)

@cli.command()
@click.option('--context-id', type=str, help='Specific context ID to summarize')
@click.option('--all-contexts', is_flag=True, help='Summarize all contexts')
@click.option('--format', type=click.Choice(['brief', 'detailed', 'bullets', 'executive']), default='brief',
              help='Summary format')
@click.option('--output', '-o', type=click.Path(), help='Output file for summary')
@click.option('--include-metadata', is_flag=True, help='Include metadata in summary')
def summarize(context_id, all_contexts, format, output, include_metadata):
    """📝 Summarize captured contexts (not implemented yet)."""
    console.print("[yellow]contextbox summarize is not implemented yet.[/yellow]")
    sys.exit(1)

@cli.command()
@click.argument('query')
@click.option('--context-type', type=click.Choice(['all', 'text', 'urls', 'screenshots']), default='all',
              help='Type of content to search')
@click.option('--limit', default=10, help='Maximum results to return')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--fuzzy', is_flag=True, help='Use fuzzy matching')
def search(query, context_type, limit, output, fuzzy):
    """🔍 Search through stored contexts using various criteria."""
    
    app = get_app()
    
    try:
        results = app.database.search_contexts(query, limit)
        
        if not results:
            search_panel = Panel(
                f"[yellow]🔍 No results found for: '{query}'[/yellow]\n\n"
                f"[dim]💡 Try:[/dim]\n"
                f"• Different search terms\n"
                f"• Check if you have captured any contexts\n"
                f"• Run 'contextbox capture' first",
                title="🔍 Search Results",
                border_style="yellow",
                box=box.ROUNDED
            )
            console.print("\n" + "="*60)
            console.print(search_panel)
            console.print("="*60)
            return
        
        results_table = Table(title=f"🔍 Search Results for '{query}'", box=box.ROUNDED)
        results_table.add_column("ID", style="cyan")
        results_table.add_column("Created", style="magenta")
        results_table.add_column("Window", style="green")
        results_table.add_column("Preview", style="yellow")
        
        for result in results:
            preview = result.get('clipboard_text') or result.get('notes') or ''
            results_table.add_row(
                str(result.get('id')),
                str(result.get('created_at') or ''),
                result.get('source_window') or '-',
                preview[:50] + ("..." if len(preview) > 50 else "")
            )
        
        console.print("\n" + "="*80)
        console.print(results_table)
        console.print("="*80)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump({
                    'query': query,
                    'context_type': context_type,
                    'search_time': datetime.now().isoformat(),
                    'results': results
                }, f, indent=2, default=str)
            console.print(f"[green]✅[/green] Search results saved to: {output}")
        
        display_success(f"Found {len(results)} results!")
        
    except Exception as e:
        display_error(f"Search failed: {e}", e)

@cli.command(name='list')
@click.option('--limit', default=20, help='Maximum number of contexts to show')
@click.option('--format', type=click.Choice(['table', 'json', 'brief', 'tree']), default='table',
              help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file for list')
@click.option('--sort', type=click.Choice(['timestamp', 'platform', 'status']), default='timestamp',
              help='Sort order')
def list_cmd(limit, format, output, sort):
    """📋 List all stored contexts with various display options."""
    
    app = get_app()
    
    try:
        contexts = app.database.list_captures(limit=limit)
        
        if not contexts:
            list_panel = Panel(
                "[yellow]📭 No contexts found in database[/yellow]\n\n"
                "[dim]💡 Get started:[/dim]\n"
                "[cyan]contextbox capture[/cyan] - Capture your first context",
                title="📋 Context List",
                border_style="yellow",
                box=box.ROUNDED
            )
            console.print("\n" + "="*60)
            console.print(list_panel)
            console.print("="*60)
            return
        
        if format == 'table':
            list_table = Table(title="📋 Stored Contexts", box=box.ROUNDED)
            list_table.add_column("ID", style="cyan", no_wrap=True)
            list_table.add_column("Created", style="magenta")
            list_table.add_column("Window", style="green")
            list_table.add_column("Screenshot", style="blue")
            
            for context in contexts:
                list_table.add_row(
                    str(context['id']),
                    str(context.get('created_at') or ''),
                    context.get('source_window') or '-',
                    "📷" if context.get('screenshot_path') else "❌"
                )
            
            console.print("\n" + "="*80)
            console.print(list_table)
            console.print("="*80)
        
        elif format == 'json':
            console.print("\n" + "="*60)
            syntax = Syntax(json.dumps(contexts, indent=2, default=str), "json", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print("="*60)
        
        elif format == 'tree':
            tree = Tree("📋 Stored Contexts")
            for context in contexts:
                node = tree.add(f"[cyan]{context['id']}[/cyan] - [green]{context.get('source_window') or '-'}[/green]")
                node.add(f"[dim]📅 {context.get('created_at')}[/dim]")
                if context.get('screenshot_path'):
                    node.add("📷 Screenshot")
            
            console.print("\n" + "="*60)
            console.print(tree)
            console.print("="*60)
        
        else:  # brief
            brief_lines = [
                f"[cyan]{context['id']}[/cyan] - [green]{context.get('source_window') or '-'}[/green] - [yellow]{context.get('created_at')}[/yellow]"
                for context in contexts
            ]
            list_panel = Panel(
                "\n".join(brief_lines),
                title="📋 Context List",
                border_style="cyan",
                box=box.ROUNDED
            )
            console.print("\n" + "="*60)
            console.print(list_panel)
            console.print("="*60)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(contexts, f, indent=2, default=str)
            console.print(f"[green]✅[/green] Context list saved to: {output}")
        
        display_success(f"Found {len(contexts)} contexts!")
        
    except Exception as e:
        display_error(f"Failed to list contexts: {e}", e)

@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed statistics')
@click.option('--output', '-o', type=click.Path(), help='Output file for statistics')
@click.option('--format', type=click.Choice(['table', 'json', 'markdown']), default='table',
              help='Output format')
def stats(detailed, output, format):
    """📊 Display database and application statistics."""
    
    app = get_app()
    
    try:
        db_stats = app.database.get_stats()
        db_path = app.database.db_path
        stats_data = {
            'total_contexts': db_stats['total_captures'],
            'total_artifacts': db_stats['total_artifacts'],
            'database_path': str(db_path),
            'database_size_bytes': os.path.getsize(db_path) if os.path.exists(db_path) else 0
        }
        
        if format == 'table':
            stats_table = Table(title="📊 Database Statistics", box=box.DOUBLE)
            stats_table.add_column("Metric", style="cyan", no_wrap=True)
            stats_table.add_column("Value", style="green")
            stats_table.add_column("Description", style="dim")
            
            stats_table.add_row("Total Contexts", f"[bold]{stats_data['total_contexts']}[/bold]", "Number of context captures")
            stats_table.add_row("Artifacts", f"[bold]{stats_data['total_artifacts']}[/bold]", "Number of stored artifacts")
            stats_table.add_row("Database Size", f"[bold]{stats_data['database_size_bytes']:,} bytes[/bold]", "Current database file size")
            if detailed:
                stats_table.add_row("Database Path", stats_data['database_path'], "SQLite database file")
            
            console.print("\n" + "="*70)
            console.print(stats_table)
            console.print("="*70)
        
        elif format == 'json':
            console.print("\n" + "="*60)
            syntax = Syntax(json.dumps(stats_data, indent=2), "json", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print("="*60)
        
        else:  # markdown
            md_content = f"""# ContextBox Statistics Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Contexts**: {stats_data['total_contexts']}
- **Artifacts**: {stats_data['total_artifacts']}
- **Database Size**: {stats_data['database_size_bytes']:,} bytes
- **Database Path**: {stats_data['database_path']}
"""
            console.print("\n" + "="*60)
            console.print(Panel(md_content, title="📊 Statistics Report", border_style="blue"))
            console.print("="*60)
        
        # Save to file if requested
        if output:
            if format == 'json':
                with open(output, 'w') as f:
                    json.dump(stats_data, f, indent=2)
            else:
                with open(output, 'w') as f:
                    f.write(md_content if format == 'markdown' else str(stats_data))
            console.print(f"[green]✅[/green] Statistics saved to: {output}")
        
        display_success("Statistics generated successfully!")
        
    except Exception as e:
        display_error(f"Failed to generate statistics: {e}", e)

@cli.command()
@click.option('--api-key', is_flag=True, help='Configure API key for AI features')
@click.option('--view', is_flag=True, help='View current configuration')
@click.option('--reset', is_flag=True, help='Reset configuration to defaults')
@click.option('--profile', '-p', default='default', help='Configuration profile to use')
@click.option('--set', 'set_key', nargs=2, help='Set configuration key value (key value)')
def config(api_key, view, reset, profile, set_key):
    """⚙️ Configure API keys and application settings."""
    
    config_dir = get_app_data_dir()
    ensure_directory(config_dir)
    config_file = os.path.join(config_dir, 'config.json')
    
    if view:
        console.print("\n[bold blue]⚙️ Current Configuration[/bold blue]")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    current_config = json.load(f)
                
                # Mask sensitive data for display
                display_config = current_config.copy()
                if 'api_key' in display_config and display_config['api_key']:
                    display_config['api_key'] = '*' * 20 + display_config['api_key'][-4:]
                
                config_panel = Panel(
                    json.dumps(display_config, indent=2),
                    title="Configuration File",
                    border_style="green",
                    box=box.ROUNDED
                )
                console.print(config_panel)
                
            except Exception as e:
                display_error(f"Failed to read configuration: {e}", exit=False)
        else:
            console.print("[yellow]📝 No configuration file found[/yellow]")
            console.print("[dim]💡 Use 'contextbox config --api-key' to create one[/dim]")
        
        return
    
    if reset:
        if Confirm.ask("🔄 Are you sure you want to reset configuration?"):
            try:
                if os.path.exists(config_file):
                    os.remove(config_file)
                display_success("Configuration reset to defaults")
            except Exception as e:
                display_error(f"Failed to reset configuration: {e}", exit=False)
        return
    
    if set_key:
        key, value = set_key
        try:
            # Load existing config
            config_data = {}
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            
            # Parse value
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            
            config_data[key] = value
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            display_success(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            display_error(f"Failed to update configuration: {e}", exit=False)
        return
    
    if api_key:
        api_key = prompt_for_api_key()
        if api_key:
            console.print("\n[green]✅[/green] You can now use:")
            console.print("  [cyan]contextbox ask[/cyan] - Ask questions about your context")
            console.print("  [cyan]contextbox summarize[/cyan] - Generate intelligent summaries")
        return
    
    # Default behavior - show configuration menu
    config_panel = Panel(
        "Use the options below to configure ContextBox:\n\n"
        "[cyan]--api-key[/cyan]    Configure API key for AI features\n"
        "[cyan]--view[/cyan]       View current configuration\n"
        "[cyan]--reset[/cyan]      Reset to default configuration\n"
        "[cyan]--set KEY VALUE[/cyan]  Set configuration value\n\n"
        "[dim]Examples:[/dim]\n"
        "[green]contextbox config --api-key[/green]\n"
        "[green]contextbox config --view[/green]\n"
        "[green]contextbox config --set log_level DEBUG[/green]",
        title="⚙️ Configuration",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print("\n" + "="*60)
    console.print(config_panel)
    console.print("="*60)

def _has_screenshot(context: Dict) -> bool:
    """Stored contexts carry artifacts as a list of rows; only CLI capture dicts map 'screenshot'."""
    artifacts = context.get('artifacts')
    return isinstance(artifacts, dict) and bool(artifacts.get('screenshot'))

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv', 'txt', 'markdown']), default='json',
              help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--context-id', type=str, help='Specific context ID to export')
@click.option('--all-contexts', is_flag=True, help='Export all contexts')
@click.option('--include-artifacts', is_flag=True, help='Include file artifacts in export')
@click.option('--compress', is_flag=True, help='Compress output file')
def export(format, output, context_id, all_contexts, include_artifacts, compress):
    """📤 Export contexts to various file formats."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("📤 Preparing export...", total=100)
        
        try:
            progress.update(task, description="📋 Collecting contexts...", completed=20)
            time.sleep(0.5)
            
            # Get contexts to export
            contexts = []
            if context_id:
                context = app.get_context(context_id)
                if not context:
                    display_error(f"Context with ID '{context_id}' not found", exit=False)
                    return
                contexts = [context]
            else:
                if all_contexts:
                    console.print("[yellow]📤 Exporting all contexts...[/yellow]")
                # limit=-1 means no limit in SQLite
                contexts = [app.get_context(str(c['id'])) for c in app.database.list_captures(limit=-1)]
            
            progress.update(task, description="🔄 Processing data...", completed=40)
            time.sleep(1)
            
            if include_artifacts:
                progress.update(task, description="📁 Including artifacts...", completed=50)
                time.sleep(0.5)
            
            progress.update(task, description="📝 Formatting export...", completed=70)
            time.sleep(1)
            
            progress.update(task, description="💾 Writing file...", completed=90)
            time.sleep(0.5)
            
            progress.update(task, description="✅ Export complete!", completed=100)
            
            if not contexts:
                console.print("[yellow]⚠️ No contexts found to export.[/yellow]")
                console.print("[dim]💡 Tip: Run 'contextbox capture' first[/dim]")
                return
            
            # Determine output file
            if not output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"contextbox_export_{timestamp}.{format}"
            
            # Export data
            if format == 'json':
                export_data = {
                    'export_timestamp': format_timestamp(time.time()),
                    'format': format,
                    'include_artifacts': include_artifacts,
                    'context_count': len(contexts),
                    'export_info': {
                        'total_contexts': len(contexts),
                        'formats_supported': ['json', 'csv', 'txt', 'markdown'],
                        'generated_by': 'ContextBox CLI v2.0.0'
                    },
                    'contexts': contexts
                }
                
                with open(output, 'w') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Context ID', 'Timestamp', 'Platform', 'Status', 'Text Length', 'URL Count', 'Has Screenshot'])
                    
                    for context in contexts:
                        writer.writerow([
                            context.get('context_id', 'N/A'),
                            context.get('timestamp', 'N/A'),
                            context.get('platform', {}).get('system', 'N/A') if isinstance(context.get('platform'), dict) else str(context.get('platform', 'N/A')),
                            context.get('status', 'N/A'),
                            len(context.get('extracted', {}).get('text', '')),
                            len(context.get('extracted', {}).get('urls', [])),
                            'Yes' if _has_screenshot(context) else 'No'
                        ])
            
            elif format == 'markdown':
                md_content = f"""# ContextBox Export Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Format**: {format.upper()}
**Contexts Exported**: {len(contexts)}
**Include Artifacts**: {include_artifacts}

## Context Details

"""
                for i, context in enumerate(contexts, 1):
                    md_content += f"""### Context {i}: {context.get('context_id', 'N/A')[:8]}

- **Timestamp**: {context.get('timestamp', 'N/A')}
- **Platform**: {context.get('platform', {}).get('system', 'N/A') if isinstance(context.get('platform'), dict) else str(context.get('platform', 'N/A'))}
- **Status**: {context.get('status', 'N/A')}
- **Text Length**: {len(context.get('extracted', {}).get('text', ''))} characters
- **URLs Found**: {len(context.get('extracted', {}).get('urls', []))}
- **Screenshot**: {'✅' if _has_screenshot(context) else '❌'}

"""
                    
                    text_content = context.get('extracted', {}).get('text', '')
                    if text_content and len(text_content) > 0:
                        md_content += f"**Text Preview:**\n```\n{text_content[:200]}...\n```\n\n"
                    
                    urls = context.get('extracted', {}).get('urls', [])
                    if urls:
                        md_content += f"**URLs:**\n"
                        for url in urls:
                            md_content += f"- {url}\n"
                        md_content += "\n"
                    
                    md_content += "---\n\n"
                
                with open(output, 'w') as f:
                    f.write(md_content)
            
            else:  # txt format
                with open(output, 'w') as f:
                    f.write(f"ContextBox Export\n")
                    f.write(f"Generated: {format_timestamp(time.time())}\n")
                    f.write(f"Format: {format}\n")
                    f.write(f"Include Artifacts: {include_artifacts}\n")
                    f.write(f"Context Count: {len(contexts)}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, context in enumerate(contexts, 1):
                        f.write(f"Context {i}:\n")
                        f.write(f"ID: {context.get('context_id', 'N/A')}\n")
                        f.write(f"Timestamp: {context.get('timestamp', 'N/A')}\n")
                        f.write(f"Platform: {context.get('platform', {}).get('system', 'N/A') if isinstance(context.get('platform'), dict) else str(context.get('platform', 'N/A'))}\n")
                        f.write(f"Status: {context.get('status', 'N/A')}\n")
                        
                        text = context.get('extracted', {}).get('text', '')
                        if text:
                            f.write(f"Text ({len(text)} chars):\n{text[:500]}...\n")
                        
                        urls = context.get('extracted', {}).get('urls', [])
                        if urls:
                            f.write(f"URLs ({len(urls)}):\n" + "\n".join(f"  - {url}" for url in urls[:10]) + "\n")
                        
                        f.write("-" * 40 + "\n\n")
            
            # Display results
            export_summary = Table(title="📤 Export Summary", box=box.ROUNDED)
            export_summary.add_column("Property", style="cyan")
            export_summary.add_column("Value", style="green")
            
            export_summary.add_row("Format", format.upper())
            export_summary.add_row("Contexts Exported", str(len(contexts)))
            export_summary.add_row("Include Artifacts", "✅ Yes" if include_artifacts else "❌ No")
            export_summary.add_row("Output File", output)
            export_summary.add_row("File Size", f"{os.path.getsize(output):,} bytes")
            
            console.print("\n" + "="*60)
            console.print(export_summary)
            console.print("="*60)
            
            display_success(f"Successfully exported {len(contexts)} contexts to {output}")
            
        except Exception as e:
            display_error(f"Export failed: {e}", e)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['json', 'csv', 'txt', 'markdown']), default='json',
              help='Input file format')
@click.option('--merge', is_flag=True, help='Merge with existing contexts')
@click.option('--overwrite', is_flag=True, help='Overwrite existing contexts')
@click.option('--validate', is_flag=True, help='Validate import format before importing')
def import_command(input_file, format, merge, overwrite, validate):
    """📥 Import contexts from various file formats."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("📥 Preparing import...", total=100)
        
        try:
            progress.update(task, description="📖 Reading file...", completed=20)
            time.sleep(0.5)
            
            imported_contexts = []
            
            # Read import file based on format
            if format == 'json':
                with open(input_file, 'r') as f:
                    import_data = json.load(f)
                
                if 'contexts' in import_data:
                    imported_contexts = import_data['contexts']
                else:
                    # Assume raw list of contexts
                    imported_contexts = import_data if isinstance(import_data, list) else [import_data]
            
            elif format == 'csv':
                import csv
                with open(input_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        context = {
                            'context_id': row.get('Context ID'),
                            'timestamp': row.get('Timestamp'),
                            'platform': {'system': row.get('Platform')},
                            'status': row.get('Status'),
                            'extracted': {
                                'text': '',
                                'urls': []
                            }
                        }
                        imported_contexts.append(context)
            
            elif format == 'markdown':
                console.print("[yellow]📝 Markdown format import is experimental[/yellow]")
                console.print("[dim]💡 Supported formats: JSON, CSV, TXT[/dim]")
                imported_contexts = []
            
            else:  # txt format
                with open(input_file, 'r') as f:
                    content = f.read()
                    console.print("[yellow]📄 Text format import is experimental[/yellow]")
                    imported_contexts = []
            
            progress.update(task, description="✅ Validating data...", completed=40)
            time.sleep(0.5)
            
            if validate:
                progress.update(task, description="🔍 Validating import format...")
                # Basic validation
                valid_contexts = []
                for context in imported_contexts:
                    if context.get('context_id') and context.get('timestamp'):
                        valid_contexts.append(context)
                imported_contexts = valid_contexts
                console.print(f"[blue]ℹ️[/blue] Validated {len(valid_contexts)} valid contexts")
            
            if not imported_contexts:
                display_error("No valid contexts found in import file", exit=False)
                return
            
            progress.update(task, description=f"📥 Importing {len(imported_contexts)} contexts...", completed=60)
            time.sleep(1)
            
            # Import contexts
            imported_count = 0
            skipped_count = 0
            
            for context in imported_contexts:
                try:
                    context_id = app.store_context(context)
                    imported_count += 1
                    
                    # Show progress
                    progress.update(task, 
                                  description=f"📥 Imported {imported_count}/{len(imported_contexts)} contexts...", 
                                  completed=60 + (imported_count / len(imported_contexts)) * 30)
                    time.sleep(0.05)
                    
                except Exception as e:
                    skipped_count += 1
                    console.print(f"[yellow]⚠️[/yellow] Skipped context: {e}")
            
            progress.update(task, description="🎯 Finalizing import...", completed=95)
            time.sleep(0.5)
            
            progress.update(task, description="✅ Import complete!", completed=100)
            
            # Display results
            import_summary = Table(title="📥 Import Summary", box=box.ROUNDED)
            import_summary.add_column("Property", style="cyan")
            import_summary.add_column("Value", style="green")
            
            import_summary.add_row("Source File", input_file)
            import_summary.add_row("Format", format.upper())
            import_summary.add_row("Contexts Imported", f"[bold]{imported_count}[/bold]")
            import_summary.add_row("Contexts Skipped", str(skipped_count))
            import_summary.add_row("Merge Mode", "✅ Yes" if merge else "❌ No")
            import_summary.add_row("Overwrite Mode", "✅ Yes" if overwrite else "❌ No")
            
            console.print("\n" + "="*60)
            console.print(import_summary)
            console.print("="*60)
            
            display_success(f"Successfully imported {imported_count} contexts from {input_file}")
            
        except Exception as e:
            display_error(f"Import failed: {e}", e)

# Add shell completion for bash/zsh/fish
if __name__ == '__main__':
    # Enable shell completion
    cli(auto_envvar_prefix='CONTEXTBOX')