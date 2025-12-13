
#!/usr/bin/env python3
"""ContextBox Advanced Search CLI"""

import click
import json
from datetime import datetime
from search import SearchEngine, SearchCriteria, SearchHistory, SavedSearch

@click.group()
def cli():
    """ContextBox Advanced Search Command Line Interface"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--type', '-t', multiple=True, help='Content types (url, text, ocr_text, etc.)')
@click.option('--from-date', help='Start date (YYYY-MM-DD)')
@click.option('--to-date', help='End date (YYYY-MM-DD)')
@click.option('--url-pattern', help='URL pattern to match')
@click.option('--regex', is_flag=True, help='Use regex matching')
@click.option('--fuzzy', type=int, help='Fuzzy matching threshold (0-100)')
@click.option('--highlight/--no-highlight', default=True, help='Include highlighting')
@click.option('--limit', default=100, help='Result limit')
@click.option('--sort', type=click.Choice(['relevance', 'date', 'title']), 
              default='relevance', help='Sort order')
def search(query, type, from_date, to_date, url_pattern, regex, fuzzy, 
          highlight, limit, sort):
    """Perform advanced search with filters"""
    
    search_engine = SearchEngine()
    
    criteria = SearchCriteria(
        query=query,
        content_types=list(type) if type else None,
        date_from=datetime.fromisoformat(from_date) if from_date else None,
        date_to=datetime.fromisoformat(to_date) if to_date else None,
        url_pattern=url_pattern,
        use_regex=regex,
        fuzzy_threshold=fuzzy,
        highlight=highlight,
        limit=limit,
        sort_by=sort
    )
    
    try:
        results, metadata = search_engine.search(criteria)
        
        click.echo(f"Found {len(results)} results in {metadata['execution_time_ms']:.2f}ms")
        click.echo("-" * 80)
        
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.title}")
            if result.url:
                click.echo(f"   URL: {result.url}")
            click.echo(f"   Type: {result.kind} | Score: {result.relevance_score:.3f}")
            if result.context_snippets:
                click.echo(f"   Context: {result.context_snippets[0][:100]}...")
            click.echo("")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option('--format', type=click.Choice(['csv', 'json', 'txt']), default='csv')
@click.option('--output', '-o', help='Output file path')
@click.argument('search_results_json')
def export(format, output, search_results_json):
    """Export search results"""
    
    if not output:
        output = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    
    try:
        with open(search_results_json, 'r') as f:
            data = json.load(f)
        
        search_results = data.get('results', [])
        
        from search import SearchResult
        results_objects = []
        for r in search_results:
            r['created_at'] = datetime.fromisoformat(r['created_at'])
            results_objects.append(SearchResult(**r))
        
        search_engine = SearchEngine()
        success = search_engine.export_results(results_objects, output, format)
        
        if success:
            click.echo(f"Exported {len(results_objects)} results to {output}")
        else:
            click.echo("Export failed", err=True)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
def history():
    """Show search history"""
    search_engine = SearchEngine()
    history = search_engine.get_search_history(limit=20)
    
    click.echo("Search History:")
    for entry in history:
        click.echo(f"  {entry.timestamp.strftime('%Y-%m-%d %H:%M')} | '{entry.query}' | "
                  f"{entry.result_count} results | {entry.execution_time_ms:.2f}ms")

@cli.command()
@click.option('--name', required=True, help='Search name')
@click.option('--description', help='Search description')
def save(name, description):
    """Save current search configuration"""
    # This would need to store the last search criteria
    click.echo("Feature requires implementing search persistence across CLI calls")

if __name__ == '__main__':
    cli()
