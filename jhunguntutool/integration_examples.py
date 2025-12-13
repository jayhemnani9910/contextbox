#!/usr/bin/env python3
"""
ContextBox Advanced Search Integration Example

This script demonstrates how to integrate the advanced search functionality
with an existing ContextBox database.

Author: ContextBox Advanced Search System
Version: 1.0.0
"""

import sys
import os
from datetime import datetime, timedelta
import json
import sqlite3

# Add current directory to path for imports
sys.path.insert(0, '/workspace')

from search import SearchEngine, SearchCriteria

def check_contextbox_database():
    """Check and demonstrate integration with ContextBox database."""
    print("🔗 ContextBox Advanced Search Integration")
    print("=" * 50)
    
    db_path = "contextbox.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("   Creating demo to show integration capabilities...")
        
        # Create a demo database
        demo_db_path = create_demo_database()
        search_engine = SearchEngine(demo_db_path)
    else:
        print(f"✅ Found ContextBox database: {db_path}")
        
        # Check database structure
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"📊 Database Tables: {', '.join(tables)}")
            
            if 'captures' in tables and 'artifacts' in tables:
                # Get basic statistics
                cursor.execute("SELECT COUNT(*) FROM captures")
                capture_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM artifacts")
                artifact_count = cursor.fetchone()[0]
                
                print(f"📊 ContextBox Statistics:")
                print(f"  Total captures: {capture_count}")
                print(f"  Total artifacts: {artifact_count}")
                
                # Get artifact types
                cursor.execute("SELECT DISTINCT kind FROM artifacts")
                kinds = [row[0] for row in cursor.fetchall()]
                print(f"  Artifact types: {', '.join(kinds)}")
                
                conn.close()
                
                # Initialize advanced search engine
                search_engine = SearchEngine(db_path)
            else:
                print("⚠️  Database structure incomplete, creating demo...")
                demo_db_path = create_demo_database()
                search_engine = SearchEngine(demo_db_path)
                
        except Exception as e:
            print(f"❌ Error accessing database: {e}")
            print("   Creating demo instead...")
            demo_db_path = create_demo_database()
            search_engine = SearchEngine(demo_db_path)
    
    print(f"\n🔍 Advanced Search Capabilities:")
    print(f"  Fuzzy matching available: {search_engine.get_search_statistics()['fuzzy_matching_available']}")
    
    # Demo searches on existing data
    demo_searches = [
        {
            "name": "URL Discovery",
            "criteria": SearchCriteria(
                query="http",
                content_types=["url"],
                limit=10
            )
        },
        {
            "name": "Recent Content",
            "criteria": SearchCriteria(
                query="",
                date_from=datetime.now() - timedelta(days=7),
                limit=20
            )
        },
        {
            "name": "OCR Text Search",
            "criteria": SearchCriteria(
                query="text",
                content_types=["ocr_text"],
                limit=10
            )
        }
    ]
    
    for demo in demo_searches:
        print(f"\n📋 {demo['name']}:")
        try:
            results, metadata = search_engine.search(demo['criteria'])
            print(f"  Found {len(results)} results in {metadata['execution_time_ms']:.2f}ms")
            
            if results:
                print(f"  Sample results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"    {i}. {result.title[:50]}{'...' if len(result.title) > 50 else ''}")
                    print(f"       Type: {result.kind}, Relevance: {result.relevance_score:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    search_engine.cleanup()

def create_demo_database():
    """Create a demo database for integration testing."""
    print("📦 Creating demo database...")
    
    demo_db_path = "demo_contextbox_integration.db"
    
    # Remove existing demo database
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
    
    conn = sqlite3.connect(demo_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Create tables matching ContextBox structure
    conn.execute("""
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_window TEXT,
            screenshot_path TEXT,
            clipboard_text TEXT,
            notes TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            url TEXT,
            title TEXT,
            text TEXT,
            metadata_json TEXT,
            FOREIGN KEY (capture_id) REFERENCES captures(id) ON DELETE CASCADE
        )
    """)
    
    # Insert demo data
    demo_data = [
        (1, "https://example.com/page1", "Example Webpage", "This is an example webpage with sample content.", "url"),
        (2, None, "Clipboard Text", "Important information copied from clipboard.", "text"),
        (3, "https://docs.python.org/", "Python Documentation", "Official Python documentation and tutorials.", "url"),
        (4, None, "OCR Result", "Text extracted from screenshot image.", "ocr_text"),
        (5, "https://github.com/", "GitHub Repository", "Code repository with project files.", "url")
    ]
    
    for i, (capture_id, url, title, text, kind) in enumerate(demo_data, 1):
        created_at = (datetime.now() - timedelta(days=i)).isoformat()
        
        conn.execute("""
            INSERT INTO captures (id, created_at, source_window, notes)
            VALUES (?, ?, ?, ?)
        """, (capture_id, created_at, f"Demo Window {i}", f"Demo notes for capture {i}"))
        
        conn.execute("""
            INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (capture_id, kind, url, title, text, json.dumps({'demo': True})))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Demo database created: {demo_db_path}")
    return demo_db_path

def create_search_api_example():
    """Create example REST API endpoint structure."""
    api_code = '''
# Advanced Search API Example (FastAPI)
from fastapi import FastAPI, Query, HTTPException
from datetime import datetime
from typing import Optional, List
from search import SearchEngine, SearchCriteria

app = FastAPI(title="ContextBox Advanced Search API")

@app.post("/search")
async def advanced_search(
    query: str = Query(..., description="Search query"),
    content_types: Optional[List[str]] = Query(None, description="Content types to filter"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    fuzzy_threshold: Optional[int] = Query(None, description="Fuzzy matching threshold"),
    use_regex: bool = Query(False, description="Use regex matching"),
    highlight: bool = Query(True, description="Include highlighting"),
    limit: int = Query(100, ge=1, le=1000, description="Result limit"),
    offset: int = Query(0, ge=0, description="Results offset")
):
    """Advanced search endpoint with full feature support."""
    
    try:
        search_engine = SearchEngine()
        
        criteria = SearchCriteria(
            query=query,
            content_types=content_types,
            date_from=date_from,
            date_to=date_to,
            fuzzy_threshold=fuzzy_threshold,
            use_regex=use_regex,
            highlight=highlight,
            limit=limit,
            offset=offset
        )
        
        results, metadata = search_engine.search(criteria)
        
        return {
            "results": [
                {
                    "id": r.id,
                    "title": r.title,
                    "content": r.content,
                    "url": r.url,
                    "kind": r.kind,
                    "relevance_score": r.relevance_score,
                    "highlights": r.highlights,
                    "context_snippets": r.context_snippets,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ],
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_search_results(
    results: List[dict],
    format: str = Query("csv", regex="^(csv|json|txt)$"),
    include_highlights: bool = Query(True)
):
    """Export search results to file."""
    
    try:
        search_engine = SearchEngine()
        
        # Convert dict results back to SearchResult objects
        search_results = []
        for r in results:
            from search import SearchResult
            search_results.append(SearchResult(**r))
        
        output_file = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        success = search_engine.export_results(search_results, output_file, format, include_highlights)
        
        if success:
            return {"success": True, "file": output_file}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/searches/saved")
async def get_saved_searches():
    """Get all saved searches."""
    search_engine = SearchEngine()
    return search_engine.get_saved_searches()

@app.post("/searches/save")
async def save_search(
    name: str,
    description: str = "",
    criteria: dict = {}
):
    """Save a search configuration."""
    search_engine = SearchEngine()
    
    from search import SearchCriteria
    search_criteria = SearchCriteria(**criteria)
    
    search_id = search_engine.save_search(name, search_criteria, description)
    return {"success": True, "search_id": search_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("search_api_example.py", "w") as f:
        f.write(api_code)
    
    print("📝 Created search API example: search_api_example.py")

def create_cli_interface():
    """Create command-line interface for advanced search."""
    cli_code = '''
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
'''
    
    with open("contextbox_search_cli.py", "w") as f:
        f.write(cli_code)
    
    print("📝 Created CLI interface: contextbox_search_cli.py")

def main():
    """Main integration demonstration."""
    print("🚀 ContextBox Advanced Search Integration Examples")
    print("=" * 60)
    
    # Check if contextbox database exists
    if os.path.exists("contextbox.db"):
        try:
            check_contextbox_database()
        except Exception as e:
            print(f"⚠️  Could not integrate with existing database: {e}")
    else:
        print("⚠️  No contextbox.db found. Creating integration examples anyway.")
        check_contextbox_database()
    
    # Create API example
    create_search_api_example()
    
    # Create CLI interface
    create_cli_interface()
    
    print("\n" + "=" * 60)
    print("✅ Integration examples created!")
    print("\nFiles created:")
    print("  📝 search_api_example.py - FastAPI REST API")
    print("  📝 contextbox_search_cli.py - Command-line interface")
    print("\nNext steps:")
    print("  1. Install FastAPI: pip install fastapi uvicorn")
    print("  2. Run API server: uvicorn search_api_example:app --reload")
    print("  3. Use CLI: python contextbox_search_cli.py search --help")

if __name__ == "__main__":
    main()