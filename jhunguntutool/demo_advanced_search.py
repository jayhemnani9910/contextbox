#!/usr/bin/env python3
"""
Advanced Search and Filtering Demonstration

This script demonstrates all the advanced search capabilities including:
- Basic and advanced search operations
- Content type filtering
- Date range filtering
- Regular expression search
- Fuzzy matching for typo tolerance
- Search highlighting and context snippets
- Search history and saved searches
- Export functionality

Author: ContextBox Advanced Search System
Version: 1.0.0
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# Add current directory to path for imports
sys.path.insert(0, '/workspace')

from search import SearchEngine, SearchCriteria, SearchResult, SearchHistory, SavedSearch

def setup_demo_database():
    """Set up a demo database with sample data."""
    import sqlite3
    import json
    
    demo_db_path = "demo_contextbox.db"
    
    # Remove existing demo database
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
    
    conn = sqlite3.connect(demo_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Create tables
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
        # Basic content for testing
        (1, "https://python.org/", "Python Programming Language", 
         "Python is a high-level programming language with clear syntax and powerful libraries.", 
         "url", datetime.now() - timedelta(days=1)),
        
        (2, None, "Machine Learning Tutorial", 
         "Machine learning is a subset of artificial intelligence that focuses on algorithms.", 
         "text", datetime.now() - timedelta(days=2)),
        
        (3, "https://openai.com/", "OpenAI Research", 
         "Artificial intelligence and machine learning research and deployment company.", 
         "url", datetime.now() - timedelta(days=3)),
        
        (4, None, "Deep Learning Concepts", 
         "Deep learning uses neural networks with multiple layers to model data.", 
         "text", datetime.now() - timedelta(days=4)),
        
        (5, "https://scikit-learn.org/", "Scikit-learn Documentation", 
         "Machine learning library for Python with tools for data mining and analysis.", 
         "url", datetime.now() - timedelta(days=5)),
        
        # Content for regex testing
        (6, "https://regex101.com/", "Regular Expression Tester", 
         r"Testing regex patterns like \d+ for digits and [a-zA-Z]+ for letters.", 
         "url", datetime.now() - timedelta(days=6)),
        
        # Content for fuzzy matching testing
        (7, None, "Data Science Projects", 
         "Projects involving data analysis, visualization, and machine learning models.", 
         "text", datetime.now() - timedelta(days=7)),
        
        # More diverse content
        (8, "https://numpy.org/", "NumPy Documentation", 
         "NumPy is the fundamental package for scientific computing with Python.", 
         "url", datetime.now() - timedelta(days=8)),
        
        (9, None, "Pandas Data Analysis", 
         "Pandas is a powerful data analysis library for Python programming.", 
         "text", datetime.now() - timedelta(days=9)),
        
        (10, "https://tensorflow.org/", "TensorFlow Platform", 
         "TensorFlow is an open-source machine learning framework developed by Google.", 
         "url", datetime.now() - timedelta(days=10)),
    ]
    
    for i, (capture_id, url, title, text, kind, created_at) in enumerate(demo_data, 1):
        # Insert capture
        conn.execute("""
            INSERT INTO captures (id, created_at, source_window, notes)
            VALUES (?, ?, ?, ?)
        """, (capture_id, created_at.isoformat(), f"Demo Window {i}", f"Demo notes for capture {i}"))
        
        # Insert artifact
        conn.execute("""
            INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (capture_id, kind, url, title, text, json.dumps({
            'demo': True, 
            'category': 'programming' if 'python' in text.lower() or 'programming' in text.lower() else 'ai'
        })))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Demo database created: {demo_db_path}")
    return demo_db_path

def demonstrate_basic_search(search_engine):
    """Demonstrate basic search functionality."""
    print("\n🔍 Basic Search Demonstration")
    print("-" * 50)
    
    criteria = SearchCriteria(
        query="machine learning",
        limit=10
    )
    
    results, metadata = search_engine.search(criteria)
    
    print(f"Query: 'machine learning'")
    print(f"Found {len(results)} results in {metadata['execution_time_ms']:.2f}ms")
    
    for i, result in enumerate(results[:3], 1):
        print(f"\nResult {i}:")
        print(f"  Title: {result.title}")
        print(f"  Type: {result.kind}")
        print(f"  Relevance: {result.relevance_score:.3f}")
        if result.url:
            print(f"  URL: {result.url}")

def demonstrate_content_filtering(search_engine):
    """Demonstrate content type filtering."""
    print("\n🎯 Content Type Filtering Demonstration")
    print("-" * 50)
    
    # Search only URLs
    criteria = SearchCriteria(
        query="python",
        content_types=["url"],
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"URLs containing 'python': {len(results)} results")
    
    # Search only text
    criteria = SearchCriteria(
        query="learning",
        content_types=["text"],
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"Text containing 'learning': {len(results)} results")

def demonstrate_date_filtering(search_engine):
    """Demonstrate date range filtering."""
    print("\n📅 Date Range Filtering Demonstration")
    print("-" * 50)
    
    # Search within last 3 days
    date_from = datetime.now() - timedelta(days=3)
    
    criteria = SearchCriteria(
        query="python",
        date_from=date_from,
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"Python content from last 3 days: {len(results)} results")
    
    # Show date ranges
    for result in results[:3]:
        print(f"  - {result.title} ({result.created_at.strftime('%Y-%m-%d')})")

def demonstrate_regex_search(search_engine):
    """Demonstrate regular expression search."""
    print("\n🔧 Regular Expression Search Demonstration")
    print("-" * 50)
    
    # Search for URLs using regex
    criteria = SearchCriteria(
        query=r"https?://[^\s]+",  # Match URLs
        use_regex=True,
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"URLs found using regex: {len(results)} results")
    
    # Search for specific patterns
    criteria = SearchCriteria(
        query=r"[a-zA-Z]+-\w+",  # Match hyphenated words
        use_regex=True,
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"Hyphenated words found: {len(results)} results")

def demonstrate_fuzzy_search(search_engine):
    """Demonstrate fuzzy matching."""
    print("\n🎲 Fuzzy Matching Demonstration")
    print("-" * 50)
    
    # Search with typos
    criteria = SearchCriteria(
        query="machne lerning",  # Intentional typos
        fuzzy_threshold=75,
        limit=10
    )
    
    results, _ = search_engine.search(criteria)
    print(f"Results for 'machne lerning' (with typos): {len(results)} matches")
    
    if results:
        print("Best fuzzy match:")
        best_result = max(results, key=lambda x: x.relevance_score)
        print(f"  Title: {best_result.title}")
        print(f"  Score: {best_result.relevance_score:.3f}")
        print(f"  Original content: {best_result.content[:100]}...")

def demonstrate_highlighting(search_engine):
    """Demonstrate search highlighting and context snippets."""
    print("\n✨ Search Highlighting Demonstration")
    print("-" * 50)
    
    criteria = SearchCriteria(
        query="python",
        highlight=True,
        context_window=40,
        limit=5
    )
    
    results, _ = search_engine.search(criteria)
    
    highlighted_count = sum(1 for r in results if r.highlights)
    print(f"Results with highlights: {highlighted_count}/{len(results)}")
    
    if results and results[0].highlights:
        result = results[0]
        print(f"\nExample highlights for '{criteria.query}':")
        for i, highlight in enumerate(result.highlights[:3], 1):
            matched_text = highlight.get('matched_text', 'N/A')
            field = highlight.get('field', 'unknown')
            print(f"  {i}. Field: {field}, Match: '{matched_text}'")
        
        if result.context_snippets:
            print(f"\nContext snippet:")
            print(f"  {result.context_snippets[0]}")

def demonstrate_saved_searches(search_engine):
    """Demonstrate saved searches functionality."""
    print("\n💾 Saved Searches Demonstration")
    print("-" * 50)
    
    # Create a saved search
    criteria = SearchCriteria(
        query="machine learning",
        content_types=["url", "text"],
        fuzzy_threshold=80
    )
    
    search_id = search_engine.save_search(
        name="ML Content Search",
        criteria=criteria,
        description="Search for machine learning content across all types"
    )
    
    print(f"✅ Saved search created with ID: {search_id}")
    
    # Load the saved search
    loaded_search = search_engine.load_saved_search(search_id)
    if loaded_search:
        print(f"✅ Loaded saved search: {loaded_search.name}")
        print(f"   Description: {loaded_search.description}")
        print(f"   Use count: {loaded_search.use_count}")
        
        # Execute the saved search
        saved_results, _ = search_engine.search(SearchCriteria(**loaded_search.criteria))
        print(f"   Results: {len(saved_results)} items found")
    
    # List all saved searches
    all_saved = search_engine.get_saved_searches()
    print(f"Total saved searches: {len(all_saved)}")
    
    # Clean up
    search_engine.delete_saved_search(search_id)
    print("🗑️  Saved search deleted")

def demonstrate_search_history(search_engine):
    """Demonstrate search history functionality."""
    print("\n📜 Search History Demonstration")
    print("-" * 50)
    
    # Perform several searches
    test_queries = ["python", "machine learning", "artificial intelligence"]
    
    for query in test_queries:
        criteria = SearchCriteria(query=query, limit=5)
        search_engine.search(criteria)
    
    # Get search history
    history = search_engine.get_search_history(limit=10)
    
    print(f"Search history entries: {len(history)}")
    
    for entry in history[:5]:
        print(f"  - '{entry.query}' → {entry.result_count} results ({entry.execution_time_ms:.2f}ms)")
    
    # Get search statistics
    stats = search_engine.get_search_statistics()
    print(f"\n📊 Search Statistics:")
    print(f"  Total searches: {stats['search_history']['total_searches']}")
    print(f"  Average results per search: {stats['search_history']['avg_results_per_search']:.1f}")
    print(f"  Average execution time: {stats['search_history']['avg_execution_time_ms']:.2f}ms")
    
    # Clear history
    search_engine.clear_search_history()
    print("🗑️  Search history cleared")

def demonstrate_export_functionality(search_engine):
    """Demonstrate search result export functionality."""
    print("\n📤 Export Functionality Demonstration")
    print("-" * 50)
    
    # Perform a search for export
    criteria = SearchCriteria(query="python", limit=5)
    results, _ = search_engine.search(criteria)
    
    if results:
        # Export to different formats
        formats = ["csv", "json", "txt"]
        
        for fmt in formats:
            output_file = f"demo_export.{fmt}"
            success = search_engine.export_results(results, output_file, fmt)
            
            if success:
                file_size = os.path.getsize(output_file)
                print(f"✅ Exported {len(results)} results to {output_file} ({file_size} bytes)")
            else:
                print(f"❌ Failed to export to {output_file}")
        
        print("\n📁 Exported files contain:")
        print("  - Full result data with metadata")
        print("  - Search highlights and context snippets")
        print("  - Relevance scores and timestamps")
    
    else:
        print("No results to export")

def demonstrate_advanced_combined_search(search_engine):
    """Demonstrate advanced search with multiple filters combined."""
    print("\n🎛️  Advanced Combined Search Demonstration")
    print("-" * 50)
    
    # Complex search with multiple filters
    criteria = SearchCriteria(
        query="learning",
        content_types=["url", "text"],
        date_from=datetime.now() - timedelta(days=7),
        fuzzy_threshold=70,
        highlight=True,
        sort_by="relevance",
        limit=10
    )
    
    results, metadata = search_engine.search(criteria)
    
    print(f"Advanced search results: {len(results)}")
    print(f"Execution time: {metadata['execution_time_ms']:.2f}ms")
    
    if results:
        print(f"\nTop 3 results:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result.title} (Score: {result.relevance_score:.3f})")
            if result.context_snippets:
                snippet = result.context_snippets[0][:80]
                print(f"     Context: {snippet}...")

def main():
    """Main demonstration function."""
    print("🚀 ContextBox Advanced Search and Filtering Demo")
    print("=" * 60)
    print("This demonstration shows all advanced search capabilities")
    print("including fuzzy matching, regex search, highlighting, and more.")
    print()
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Set up demo database
    demo_db_path = setup_demo_database()
    
    try:
        # Initialize search engine
        search_engine = SearchEngine(demo_db_path)
        
        # Run all demonstrations
        demonstrate_basic_search(search_engine)
        demonstrate_content_filtering(search_engine)
        demonstrate_date_filtering(search_engine)
        demonstrate_regex_search(search_engine)
        demonstrate_fuzzy_search(search_engine)
        demonstrate_highlighting(search_engine)
        demonstrate_saved_searches(search_engine)
        demonstrate_search_history(search_engine)
        demonstrate_export_functionality(search_engine)
        demonstrate_advanced_combined_search(search_engine)
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("🎉 Advanced search functionality is working perfectly!")
        
        # Show final statistics
        stats = search_engine.get_search_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  Fuzzy matching available: {stats['fuzzy_matching_available']}")
        print(f"  Total searches performed: {stats['search_history']['total_searches']}")
        print(f"  Saved searches: {stats['saved_searches']['total_saved_searches']}")
        
        search_engine.cleanup()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo database
        if os.path.exists(demo_db_path):
            os.remove(demo_db_path)
            print(f"\n🧹 Cleaned up demo database: {demo_db_path}")

if __name__ == "__main__":
    main()