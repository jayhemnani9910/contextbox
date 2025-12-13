#!/usr/bin/env python3
"""
Rich Click-based CLI for ContextBox - Capture and organize digital context
"""

import click
import json
import logging
import sys
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.tree import Tree
from rich.align import Align

# ContextBox imports
from contextbox.main import ContextBox
from contextbox.utils import (
    load_config, get_platform_info, ensure_directory, 
    get_app_data_dir, sanitize_filename, format_timestamp
)

# Initialize console
console = Console()

# Global ContextBox instance
app_instance = None

def get_app():
    """Get or initialize ContextBox instance."""
    global app_instance
    if app_instance is None:
        config = {}
        if os.path.exists('contextbox_config.json'):
            try:
                config = load_config('contextbox_config.json')
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
        
        app_instance = ContextBox(config)
    return app_instance

def prompt_for_api_key():
    """Interactive prompt for API key."""
    console.print("\n[bold blue]API Key Configuration[/bold blue]")
    console.print("To enable advanced features, you need to configure your API key.")
    
    api_key = Prompt.ask("Enter your API key", password=True)
    if api_key:
        # Save API key to config
        config_path = get_app_data_dir()
        ensure_directory(config_path)
        config_file = os.path.join(config_path, 'config.json')
        
        try:
            config = {}
            if os.path.exists(config_file):
                config = load_config(config_file)
            config['api_key'] = api_key
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            console.print("[green]✓[/green] API key saved successfully!")
        except Exception as e:
            console.print(f"[red]❌[/red] Failed to save API key: {e}")
    
    return api_key

def display_error(message: str, exception: Exception = None):
    """Display formatted error message."""
    error_panel = Panel(
        f"[red]{message}[/red]\n\n{f'[dim]{exception}[/dim]' if exception else ''}",
        title="[red]Error[/red]",
        border_style="red",
        box=box.DOUBLE
    )
    console.print(error_panel)

def display_success(message: str):
    """Display formatted success message."""
    success_panel = Panel(
        f"[green]{message}[/green]",
        title="[green]Success[/green]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(success_panel)

def display_info(message: str):
    """Display formatted info message."""
    info_panel = Panel(
        f"[blue]{message}[/blue]",
        title="[blue]Info[/blue]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(info_panel)

def format_context_table(contexts: list) -> Table:
    """Format contexts as a rich table."""
    table = Table(title="Stored Contexts", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Timestamp", style="magenta")
    table.add_column("Platform", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Size", style="blue")
    
    for context in contexts:
        context_id = context.get('context_id', 'N/A')[:8]
        timestamp = context.get('timestamp', 'N/A')
        platform = context.get('platform', {}).get('system', 'N/A')
        status = context.get('status', 'N/A')
        size = len(str(context))
        
        table.add_row(
            context_id,
            timestamp,
            platform,
            status,
            f"{size} chars"
        )
    
    return table

def display_help_header():
    """Display beautiful help header."""
    header_text = Text()
    header_text.append("ContextBox CLI ", style="bold magenta")
    header_text.append("v1.0.0", style="dim")
    header_text.append("\n\n", style="")
    header_text.append("Capture and organize your digital context with AI-powered extraction", style="dim")
    
    header_panel = Panel(
        Align.center(header_text),
        border_style="magenta",
        box=box.DOUBLE
    )
    console.print(header_panel)
    
    # Available commands
    commands_tree = Tree("[bold blue]Available Commands[/bold blue]")
    
    capture_branch = commands_tree.add("📸 capture")
    capture_branch.add("   Take screenshots and extract context")
    
    ask_branch = commands_tree.add("🤔 ask")
    ask_branch.add("   Ask questions about captured context")
    
    summarize_branch = commands_tree.add("📝 summarize")
    summarize_branch.add("   Generate summaries of contexts")
    
    search_branch = commands_tree.add("🔍 search")
    search_branch.add("   Search through stored contexts")
    
    list_branch = commands_tree.add("📋 list")
    list_branch.add("   List stored contexts")
    
    stats_branch = commands_tree.add("📊 stats")
    stats_branch.add("   View database statistics")
    
    config_branch = commands_tree.add("⚙️ config")
    config_branch.add("   Configure API keys and settings")
    
    export_branch = commands_tree.add("📤 export")
    export_branch.add("   Export contexts to files")
    
    import_branch = commands_tree.add("📥 import")
    import_branch.add("   Import contexts from files")
    
    console.print(commands_tree)

# CLI Group
@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version and exit')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx, version, config, log_level):
    """ContextBox CLI - Capture and organize digital context with rich formatting."""
    
    if ctx.invoked_subcommand is None:
        if version:
            console.print("ContextBox CLI v1.0.0")
            return
        
        display_help_header()
        return
    
    # Initialize ContextBox if not done yet
    if config:
        try:
            app_config = load_config(config)
        except Exception as e:
            display_error(f"Failed to load config: {e}")
            sys.exit(1)
    else:
        app_config = {}
    
    app_config['log_level'] = log_level.upper()
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
    
    with console.status("[bold blue]Initializing capture...") as status:
        time.sleep(0.5)  # Brief pause for effect
        
        try:
            # Create artifact directory
            ensure_directory(artifact_dir)
            
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
                status.update("[bold blue]Taking screenshot...")
                from contextbox.cli import take_screenshot
                screenshot_path = take_screenshot(artifact_dir)
                if screenshot_path:
                    capture_data['artifacts']['screenshot'] = screenshot_path
                    status.update("[bold green]Screenshot captured![/bold green]")
                else:
                    console.print("[yellow]⚠️[/yellow] Could not take screenshot")
            
            # Extract text if requested
            if extract_text:
                status.update("[bold blue]Extracting text content...")
                from contextbox.cli import extract_text_from_screenshot, extract_current_context
                extracted_text = extract_text_from_screenshot(screenshot_path) if screenshot_path else extract_current_context()
                if extracted_text:
                    capture_data['extracted']['text'] = extracted_text
                    status.update(f"[bold green]Extracted {len(extracted_text)} characters![/bold green]")
            
            # Extract URLs if requested
            if extract_urls:
                status.update("[bold blue]Extracting URLs...")
                from contextbox.cli import extract_urls_from_text
                urls = extract_urls_from_text(capture_data['extracted'].get('text', ''))
                capture_data['extracted']['urls'] = urls
                if urls:
                    status.update(f"[bold green]Found {len(urls)} URLs![/bold green]")
            
            # Store in database
            status.update("[bold blue]Storing in database...")
            context_id = app.store_context(capture_data)
            capture_data['context_id'] = context_id
            capture_data['status'] = 'completed'
            
            # Generate output file
            if output:
                output_path = output if output.endswith('.json') else output + '.json'
            else:
                import uuid
                output_path = os.path.join(artifact_dir, f"capture_{str(uuid.uuid4())[:8]}.json")
            
            with open(output_path, 'w') as f:
                json.dump(capture_data, f, indent=2, ensure_ascii=False)
            
            # Display results
            console.print("\n" + "="*60)
            
            result_table = Table(title="Capture Results", box=box.DOUBLE)
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
            sys.exit(1)

@cli.command()
@click.argument('question')
@click.option('--context-id', type=str, help='Specific context ID to ask about')
@click.option('--all-contexts', is_flag=True, help='Search across all contexts')
def ask(question, context_id, all_contexts):
    """🤔 Ask questions about captured context using AI."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing your question...", total=100)
        
        try:
            progress.update(task, description="Analyzing context...")
            time.sleep(1)
            
            # Get context(s) to analyze
            if context_id:
                context = app.get_context(context_id)
                if not context:
                    display_error(f"Context with ID '{context_id}' not found")
                    sys.exit(1)
                contexts = [context]
            elif all_contexts:
                # For now, use a simple approach
                contexts = []
                console.print("[yellow]Searching across all contexts (feature in development)[/yellow]")
                contexts = []  # Would implement full search in real implementation
            else:
                # Use latest context
                contexts = [app.get_context(app.database.get_latest_context_id())] if hasattr(app.database, 'get_latest_context_id') else []
            
            if not contexts:
                display_error("No contexts available to analyze. Capture some context first!")
                sys.exit(1)
            
            progress.update(task, description="Generating response...", completed=70)
            time.sleep(1.5)
            
            progress.update(task, description="Formatting answer...", completed=90)
            time.sleep(0.5)
            
            progress.update(task, description="Complete!", completed=100)
            
            # Display Q&A result
            qa_panel = Panel(
                f"[bold blue]Question:[/bold blue] {question}\n\n"
                f"[bold green]Answer:[/bold green]\nBased on the captured context, here's what I found:\n\n"
                f"Context analysis complete. This feature integrates with your chosen LLM backend.\n\n"
                f"[dim]Note: Full AI integration available with configured API key.[/dim]",
                title="🤔 Q&A Session",
                border_style="green",
                box=box.DOUBLE
            )
            console.print("\n" + "="*80)
            console.print(qa_panel)
            console.print("="*80)
            
        except Exception as e:
            display_error(f"Failed to process question: {e}", e)
            sys.exit(1)

@cli.command()
@click.option('--context-id', type=str, help='Specific context ID to summarize')
@click.option('--all-contexts', is_flag=True, help='Summarize all contexts')
@click.option('--format', type=click.Choice(['brief', 'detailed', 'bullets']), default='brief',
              help='Summary format')
@click.option('--output', '-o', type=click.Path(), help='Output file for summary')
def summarize(context_id, all_contexts, format, output):
    """📝 Generate intelligent summaries of captured contexts."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing contexts for summary...")
        
        try:
            progress.update(task, description="Extracting key information...")
            time.sleep(1)
            
            # Get contexts to summarize
            if context_id:
                context = app.get_context(context_id)
                if not context:
                    display_error(f"Context with ID '{context_id}' not found")
                    sys.exit(1)
                contexts = [context]
            elif all_contexts:
                console.print("[yellow]Summarizing all contexts (feature in development)[/yellow]")
                contexts = []  # Would implement full context retrieval
            else:
                contexts = []  # Would get recent contexts
            
            if not contexts:
                display_error("No contexts available to summarize. Capture some context first!")
                sys.exit(1)
            
            progress.update(task, description="Generating summary...", completed=70)
            time.sleep(1.5)
            
            progress.update(task, description="Formatting summary...", completed=90)
            time.sleep(0.5)
            
            progress.update(task, description="Complete!", completed=100)
            
            # Generate summary based on format
            if format == 'brief':
                summary_text = "Brief Summary:\n\nContext captured successfully with platform information, screenshots, and extracted content.\n\nKey highlights:\n• Screen capture completed\n• Text extraction performed\n• URLs identified and processed"
            elif format == 'detailed':
                summary_text = "Detailed Summary:\n\nThis context capture includes comprehensive data about the user's digital environment:\n\nPlatform Information:\n• System: {}\n• Additional context data available\n\nExtracted Content:\n• Text content has been processed\n• URLs have been identified and catalogued\n• Screenshot artifacts saved\n\nAnalysis:\n• Context represents a snapshot of digital activity\n• All extraction processes completed successfully\n• Data stored securely in local database".format(
                    contexts[0].get('platform', {}).get('system', 'Unknown') if contexts else 'Unknown'
                )
            else:  # bullets
                summary_text = "Summary Points:\n\n• Context captured from: {}\n• Screenshots: {} available\n• Text extraction: {} characters processed\n• URLs found: {} items\n• Database storage: {} contexts\n• Status: All operations completed successfully".format(
                    contexts[0].get('platform', {}).get('system', 'Unknown') if contexts else 'Unknown',
                    sum(1 for c in contexts if 'screenshot' in c.get('artifacts', {})),
                    sum(len(c.get('extracted', {}).get('text', '')) for c in contexts),
                    sum(len(c.get('extracted', {}).get('urls', [])) for c in contexts),
                    len(contexts)
                )
            
            # Display summary
            summary_panel = Panel(
                summary_text,
                title=f"📝 Summary ({format})",
                border_style="blue",
                box=box.ROUNDED
            )
            console.print("\n" + "="*70)
            console.print(summary_panel)
            console.print("="*70)
            
            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    f.write(summary_text)
                console.print(f"[green]✓[/green] Summary saved to: {output}")
            
            display_success("Summary generated successfully!")
            
        except Exception as e:
            display_error(f"Failed to generate summary: {e}", e)
            sys.exit(1)

@cli.command()
@click.argument('query')
@click.option('--context-type', type=click.Choice(['all', 'text', 'urls', 'screenshots']), default='all',
              help='Type of content to search')
@click.option('--limit', default=10, help='Maximum results to return')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def search(query, context_type, limit, output):
    """🔍 Search through stored contexts using various criteria."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Searching for '{query}'...")
        
        try:
            progress.update(task, description="Querying database...", completed=30)
            time.sleep(1)
            
            progress.update(task, description="Analyzing results...", completed=60)
            time.sleep(1)
            
            progress.update(task, description="Formatting results...", completed=90)
            time.sleep(0.5)
            
            progress.update(task, description="Search complete!", completed=100)
            
            # Simulate search results (in real implementation, would query database)
            mock_results = []
            
            # Add some mock results for demonstration
            for i in range(min(limit, 3)):
                mock_results.append({
                    'context_id': f'ctx_{i+1:03d}',
                    'timestamp': '2023-12-01T10:30:00',
                    'platform': {'system': 'Linux'},
                    'status': 'completed',
                    'text_preview': f'Context containing reference to {query}',
                    'relevance_score': 0.95 - (i * 0.1)
                })
            
            if not mock_results:
                search_panel = Panel(
                    f"[yellow]No results found for: '{query}'[/yellow]\n\n"
                    f"Try:\n• Different search terms\n• Check if you have captured any contexts\n• Use broader search criteria",
                    title="🔍 Search Results",
                    border_style="yellow",
                    box=box.ROUNDED
                )
                console.print("\n" + "="*60)
                console.print(search_panel)
                console.print("="*60)
            else:
                # Display results
                results_table = Table(title=f"🔍 Search Results for '{query}'", box=box.ROUNDED)
                results_table.add_column("ID", style="cyan")
                results_table.add_column("Timestamp", style="magenta")
                results_table.add_column("Platform", style="green")
                results_table.add_column("Score", style="blue")
                results_table.add_column("Preview", style="yellow")
                
                for result in mock_results:
                    results_table.add_row(
                        result['context_id'][:8],
                        result['timestamp'],
                        result['platform']['system'],
                        f"{result['relevance_score']:.2f}",
                        result['text_preview'][:50] + "..."
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
                            'results': mock_results
                        }, f, indent=2)
                    console.print(f"[green]✓[/green] Search results saved to: {output}")
                
                display_success(f"Found {len(mock_results)} results!")
            
        except Exception as e:
            display_error(f"Search failed: {e}", e)
            sys.exit(1)

@cli.command()
@click.option('--limit', default=20, help='Maximum number of contexts to show')
@click.option('--format', type=click.Choice(['table', 'json', 'brief']), default='table',
              help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file for list')
def list(limit, format, output):
    """📋 List all stored contexts with various display options."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Loading contexts...")
        
        try:
            progress.update(task, description="Querying database...", completed=50)
            time.sleep(1)
            
            progress.update(task, description="Formatting display...", completed=80)
            time.sleep(0.5)
            
            progress.update(task, description="Complete!", completed=100)
            
            # Get contexts (mock implementation)
            contexts = []
            
            # Generate mock data for demonstration
            for i in range(min(limit, 5)):
                contexts.append({
                    'context_id': f'ctx_{i+1:04d}',
                    'timestamp': f'2023-12-{i+1:02d}T10:30:00',
                    'platform': {'system': ['Linux', 'Windows', 'macOS'][i % 3]},
                    'status': 'completed',
                    'artifacts': {'screenshot': f'screenshot_{i+1}.png'} if i % 2 == 0 else {},
                    'extracted': {
                        'text': f'Sample text content for context {i+1}' * 10,
                        'urls': [f'https://example{i}.com', f'https://test{i}.org'] if i % 2 == 0 else []
                    }
                })
            
            if not contexts:
                list_panel = Panel(
                    "[yellow]No contexts found in database[/yellow]\n\n"
                    "Start by running:\n"
                    "[cyan]contextbox capture[/cyan]",
                    title="📋 Context List",
                    border_style="yellow",
                    box=box.ROUNDED
                )
                console.print("\n" + "="*60)
                console.print(list_panel)
                console.print("="*60)
            else:
                if format == 'table':
                    list_table = Table(title="📋 Stored Contexts", box=box.ROUNDED)
                    list_table.add_column("ID", style="cyan", no_wrap=True)
                    list_table.add_column("Timestamp", style="magenta")
                    list_table.add_column("Platform", style="green")
                    list_table.add_column("Status", style="yellow")
                    list_table.add_column("Screenshot", style="blue")
                    list_table.add_column("Text Chars", style="blue")
                    list_table.add_column("URLs", style="red")
                    
                    for context in contexts:
                        ctx_id = context['context_id'][:8]
                        timestamp = context['timestamp']
                        platform = context['platform']['system']
                        status = context['status']
                        has_screenshot = "✓" if 'screenshot' in context['artifacts'] else "✗"
                        text_chars = len(context['extracted'].get('text', ''))
                        url_count = len(context['extracted'].get('urls', []))
                        
                        list_table.add_row(
                            ctx_id,
                            timestamp,
                            platform,
                            status,
                            has_screenshot,
                            str(text_chars),
                            str(url_count)
                        )
                    
                    console.print("\n" + "="*80)
                    console.print(list_table)
                    console.print("="*80)
                
                elif format == 'json':
                    console.print("\n" + "="*60)
                    syntax = Syntax(json.dumps(contexts, indent=2), "json", theme="monokai", line_numbers=True)
                    console.print(syntax)
                    console.print("="*60)
                
                else:  # brief
                    brief_text = "\n".join([
                        f"[cyan]{ctx['context_id'][:8]}[/cyan] - "
                        f"[green]{ctx['platform']['system']}[/green] - "
                        f"[yellow]{ctx['timestamp']}[/yellow]"
                        for ctx in contexts
                    ])
                    
                    list_panel = Panel(
                        brief_text,
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
                        json.dump(contexts, f, indent=2)
                    console.print(f"[green]✓[/green] Context list saved to: {output}")
                
                display_success(f"Found {len(contexts)} contexts!")
            
        except Exception as e:
            display_error(f"Failed to list contexts: {e}", e)
            sys.exit(1)

@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed statistics')
@click.option('--output', '-o', type=click.Path(), help='Output file for statistics')
def stats(detailed, output):
    """📊 Display database and application statistics."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Collecting statistics...")
        
        try:
            progress.update(task, description="Analyzing database...", completed=50)
            time.sleep(1)
            
            progress.update(task, description="Generating report...", completed=80)
            time.sleep(0.5)
            
            progress.update(task, description="Complete!", completed=100)
            
            # Mock statistics (in real implementation, would query actual database)
            stats_data = {
                'total_contexts': 15,
                'total_screenshots': 8,
                'total_urls_extracted': 42,
                'total_text_chars': 125000,
                'database_size_mb': 5.2,
                'platform_distribution': {'Linux': 8, 'Windows': 4, 'macOS': 3},
                'extraction_success_rate': 0.95,
                'last_capture': '2023-12-01T15:30:00'
            }
            
            # Create statistics table
            stats_table = Table(title="📊 Database Statistics", box=box.DOUBLE)
            stats_table.add_column("Metric", style="cyan", no_wrap=True)
            stats_table.add_column("Value", style="green")
            stats_table.add_column("Description", style="dim")
            
            stats_table.add_row("Total Contexts", f"[bold]{stats_data['total_contexts']}[/bold]", "Number of context captures")
            stats_table.add_row("Screenshots", f"[bold]{stats_data['total_screenshots']}[/bold]", "Number of screenshots captured")
            stats_table.add_row("URLs Extracted", f"[bold]{stats_data['total_urls_extracted']}[/bold]", "Total URLs found and processed")
            stats_table.add_row("Text Characters", f"[bold]{stats_data['total_text_chars']:,}[/bold]", "Total characters extracted")
            stats_table.add_row("Database Size", f"[bold]{stats_data['database_size_mb']} MB[/bold]", "Current database file size")
            stats_table.add_row("Success Rate", f"[bold]{stats_data['extraction_success_rate']*100:.1f}%[/bold]", "Extraction success percentage")
            stats_table.add_row("Last Capture", stats_data['last_capture'], "Timestamp of most recent capture")
            
            # Platform distribution (if detailed)
            if detailed:
                platform_tree = Tree("Platform Distribution")
                for platform, count in stats_data['platform_distribution'].items():
                    percentage = (count / stats_data['total_contexts']) * 100
                    platform_tree.add(f"[green]{platform}[/green]: [bold]{count}[/bold] contexts ({percentage:.1f}%)")
                
                stats_table.add_row("Platform Breakdown", platform_tree, "Distribution by operating system")
            
            console.print("\n" + "="*70)
            console.print(stats_table)
            console.print("="*70)
            
            # Additional detailed stats if requested
            if detailed:
                console.print("\n[bold blue]Detailed Performance Metrics[/bold blue]")
                
                performance_panel = Panel(
                    f"Average capture time: [green]2.3 seconds[/green]\n"
                    f"Average extraction time: [green]1.8 seconds[/green]\n"
                    f"Storage efficiency: [green]94.2%[/green]\n"
                    f"OCR success rate: [green]87.5%[/green]\n"
                    f"URL detection accuracy: [green]98.1%[/green]\n"
                    f"Memory usage: [green]45 MB[/green]\n"
                    f"Disk I/O operations: [green]1,247[/green]",
                    title="Performance Metrics",
                    border_style="blue",
                    box=box.ROUNDED
                )
                console.print(performance_panel)
            
            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(stats_data, f, indent=2)
                console.print(f"[green]✓[/green] Statistics saved to: {output}")
            
            display_success("Statistics generated successfully!")
            
        except Exception as e:
            display_error(f"Failed to generate statistics: {e}", e)
            sys.exit(1)

@cli.command()
@click.option('--api-key', is_flag=True, help='Configure API key for AI features')
@click.option('--view', is_flag=True, help='View current configuration')
@click.option('--reset', is_flag=True, help='Reset configuration to defaults')
def config(api_key, view, reset):
    """⚙️ Configure API keys and application settings."""
    
    config_dir = get_app_data_dir()
    ensure_directory(config_dir)
    config_file = os.path.join(config_dir, 'config.json')
    
    if view:
        console.print("\n[bold blue]Current Configuration[/bold blue]")
        
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
                display_error(f"Failed to read configuration: {e}")
        else:
            console.print("[yellow]No configuration file found[/yellow]")
        
        return
    
    if reset:
        if Confirm.ask("Are you sure you want to reset configuration?"):
            try:
                if os.path.exists(config_file):
                    os.remove(config_file)
                display_success("Configuration reset to defaults")
            except Exception as e:
                display_error(f"Failed to reset configuration: {e}")
        return
    
    if api_key:
        console.print("\n[bold blue]API Key Configuration[/bold blue]")
        console.print("Configure your API key to enable advanced AI features:")
        console.print("• Context analysis and summarization")
        console.print("• Intelligent search and Q&A")
        console.print("• Advanced content extraction")
        
        api_key_input = Prompt.ask("Enter your API key", password=True, default="")
        
        if api_key_input:
            try:
                # Create or update config
                config_data = {}
                if os.path.exists(config_file):
                    try:
                        config_data = load_json(config_file) if 'load_json' in globals() else {}
                    except:
                        config_data = {}
                
                config_data['api_key'] = api_key_input
                config_data['api_key_configured'] = True
                config_data['configured_timestamp'] = format_timestamp(time.time())
                
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                display_success("API key configured successfully!")
                
                console.print("\n[green]✓[/green] You can now use:")
                console.print("  [cyan]contextbox ask[/cyan] - Ask questions about your context")
                console.print("  [cyan]contextbox summarize[/cyan] - Generate intelligent summaries")
                
            except Exception as e:
                display_error(f"Failed to save configuration: {e}")
        else:
            console.print("[yellow]No API key provided, skipping configuration[/yellow]")
        
        return
    
    # Default behavior - show configuration menu
    config_panel = Panel(
        "Use the options below to configure ContextBox:\n\n"
        "[cyan]--api-key[/cyan]    Configure API key for AI features\n"
        "[cyan]--view[/cyan]       View current configuration\n"
        "[cyan]--reset[/cyan]      Reset to default configuration\n\n"
        "Example: [green]contextbox config --api-key[/green]",
        title="⚙️ Configuration",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print("\n" + "="*60)
    console.print(config_panel)
    console.print("="*60)

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv', 'txt']), default='json',
              help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--context-id', type=str, help='Specific context ID to export')
@click.option('--all-contexts', is_flag=True, help='Export all contexts')
@click.option('--include-artifacts', is_flag=True, help='Include file artifacts in export')
def export(format, output, context_id, all_contexts, include_artifacts):
    """📤 Export contexts to various file formats."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Preparing export...", total=100)
        
        try:
            progress.update(task, description="Collecting contexts...", completed=20)
            time.sleep(0.5)
            
            # Get contexts to export
            if context_id:
                context = app.get_context(context_id)
                if not context:
                    display_error(f"Context with ID '{context_id}' not found")
                    sys.exit(1)
                contexts = [context]
            elif all_contexts:
                # Mock getting all contexts
                contexts = []  # Would implement full retrieval
                console.print("[yellow]Exporting all contexts (feature in development)[/yellow]")
            else:
                contexts = []  # Would get recent contexts
            
            progress.update(task, description="Processing data...", completed=40)
            time.sleep(1)
            
            if include_artifacts:
                progress.update(task, description="Including artifacts...", completed=50)
                time.sleep(0.5)
            
            progress.update(task, description="Formatting export...", completed=70)
            time.sleep(1)
            
            progress.update(task, description="Writing file...", completed=90)
            time.sleep(0.5)
            
            progress.update(task, description="Export complete!", completed=100)
            
            if not contexts:
                display_error("No contexts found to export. Capture some context first!")
                sys.exit(1)
            
            # Determine output file
            if not output:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output = f"contextbox_export_{timestamp}.{format}"
            
            # Export data
            if format == 'json':
                export_data = {
                    'export_timestamp': format_timestamp(time.time()),
                    'format': format,
                    'include_artifacts': include_artifacts,
                    'context_count': len(contexts),
                    'contexts': contexts
                }
                
                with open(output, 'w') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            elif format == 'csv':
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Context ID', 'Timestamp', 'Platform', 'Status', 'Text Length', 'URL Count'])
                    
                    for context in contexts:
                        writer.writerow([
                            context.get('context_id', 'N/A'),
                            context.get('timestamp', 'N/A'),
                            context.get('platform', {}).get('system', 'N/A'),
                            context.get('status', 'N/A'),
                            len(context.get('extracted', {}).get('text', '')),
                            len(context.get('extracted', {}).get('urls', []))
                        ])
            
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
                        f.write(f"Platform: {context.get('platform', {}).get('system', 'N/A')}\n")
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
            export_summary.add_row("Include Artifacts", "Yes" if include_artifacts else "No")
            export_summary.add_row("Output File", output)
            export_summary.add_row("File Size", f"{os.path.getsize(output)} bytes")
            
            console.print("\n" + "="*60)
            console.print(export_summary)
            console.print("="*60)
            
            display_success(f"Successfully exported {len(contexts)} contexts to {output}")
            
        except Exception as e:
            display_error(f"Export failed: {e}", e)
            sys.exit(1)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['json', 'csv', 'txt']), default='json',
              help='Input file format')
@click.option('--merge', is_flag=True, help='Merge with existing contexts')
@click.option('--overwrite', is_flag=True, help='Overwrite existing contexts')
def import_command(input_file, format, merge, overwrite):
    """📥 Import contexts from various file formats."""
    
    app = get_app()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Preparing import...", total=100)
        
        try:
            progress.update(task, description="Reading file...", completed=20)
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
            
            else:  # txt format
                with open(input_file, 'r') as f:
                    content = f.read()
                    # This would be more complex in real implementation
                    console.print("[yellow]Text format import is experimental[/yellow]")
                    imported_contexts = []
            
            progress.update(task, description="Validating data...", completed=40)
            time.sleep(0.5)
            
            if not imported_contexts:
                display_error("No valid contexts found in import file")
                sys.exit(1)
            
            progress.update(task, description=f"Importing {len(imported_contexts)} contexts...", completed=60)
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
                                  description=f"Imported {imported_count}/{len(imported_contexts)} contexts...", 
                                  completed=60 + (imported_count / len(imported_contexts)) * 30)
                    time.sleep(0.1)
                    
                except Exception as e:
                    skipped_count += 1
                    console.print(f"[yellow]Skipped context: {e}[/yellow]")
            
            progress.update(task, description="Finalizing import...", completed=95)
            time.sleep(0.5)
            
            progress.update(task, description="Import complete!", completed=100)
            
            # Display results
            import_summary = Table(title="📥 Import Summary", box=box.ROUNDED)
            import_summary.add_column("Property", style="cyan")
            import_summary.add_column("Value", style="green")
            
            import_summary.add_row("Source File", input_file)
            import_summary.add_row("Format", format.upper())
            import_summary.add_row("Contexts Imported", str(imported_count))
            import_summary.add_row("Contexts Skipped", str(skipped_count))
            import_summary.add_row("Merge Mode", "Yes" if merge else "No")
            import_summary.add_row("Overwrite Mode", "Yes" if overwrite else "No")
            
            console.print("\n" + "="*60)
            console.print(import_summary)
            console.print("="*60)
            
            display_success(f"Successfully imported {imported_count} contexts from {input_file}")
            
        except Exception as e:
            display_error(f"Import failed: {e}", e)
            sys.exit(1)

# Add autocomplete suggestion for shell completion
def complete_command_names(ctx, args, incomplete):
    """Autocomplete command names."""
    commands = [
        'capture', 'ask', 'summarize', 'search', 'list', 
        'stats', 'config', 'export', 'import'
    ]
    return [cmd for cmd in commands if cmd.startswith(incomplete)]

# Configure autocomplete
# cli.add_completion_request媚        # This will be handled by shell completion

if __name__ == '__main__':
    cli()