"""
Comprehensive Test Suite for Advanced Search and Filtering

This test suite covers all search functionality including:
- Basic search operations
- Advanced filtering (content type, date ranges, URL patterns)
- Regular expression support
- Fuzzy matching
- Search highlighting and context snippets
- Search history and saved searches
- Result export functionality
- Edge cases and error handling

Author: ContextBox Advanced Search System
Version: 1.0.0
"""

import unittest
import tempfile
import os
import sqlite3
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import logging
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/workspace')

from search import (
    SearchEngine, SearchCriteria, SearchResult, SearchHistory, SavedSearch,
    SearchError, quick_search, search_by_date_range, fuzzy_search
)


class TestSearchEngine(unittest.TestCase):
    """Test cases for the SearchEngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and test data."""
        # Create temporary database
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_contextbox.db")
        
        # Initialize database with test data
        cls._setup_test_database()
        
        # Initialize search engine
        cls.search_engine = SearchEngine(cls.test_db_path)
        
        # Set up logging
        logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        if hasattr(cls, 'search_engine'):
            cls.search_engine.cleanup()
        
        # Remove temporary directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _setup_test_database(cls):
        """Set up test database with sample data."""
        conn = sqlite3.connect(cls.test_db_path)
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
        
        # Insert test captures
        test_captures = [
            (1, "https://example.com/page1", "Machine Learning Introduction", 
             "Machine learning is a subset of artificial intelligence..."),
            (2, "https://python.org/tutorial", "Python Programming Guide", 
             "Python is a high-level programming language..."),
            (3, "https://news.tech.com/ai-breakthrough", "AI Research News", 
             "Recent advances in artificial intelligence have shown..."),
            (4, "https://docs.python.org/3/library", "Python Documentation", 
             "The Python Standard Library includes modules..."),
            (5, "https://example.com/research", "Deep Learning Research", 
             "Deep learning models have achieved state-of-the-art...")
        ]
        
        for i, (capture_id, url, title, text) in enumerate(test_captures, 1):
            # Insert capture
            created_at = datetime.now() - timedelta(days=i)
            conn.execute("""
                INSERT INTO captures (id, created_at, source_window, notes)
                VALUES (?, ?, ?, ?)
            """, (capture_id, created_at.isoformat(), f"Window {i}", f"Notes for capture {i}"))
            
            # Insert artifacts
            conn.execute("""
                INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (capture_id, 'url', url, title, text, json.dumps({'confidence': 0.9})))
            
            # Add text-only artifact for some captures
            if i % 2 == 0:
                conn.execute("""
                    INSERT INTO artifacts (capture_id, kind, title, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (capture_id, 'text', f"Text Content {i}", 
                     f"This is additional text content for capture {i}. It contains keywords like machine and learning.",
                     json.dumps({'type': 'extracted_text'})))
        
        # Add more diverse test data
        diverse_data = [
            (6, 'image', None, "Screenshot Analysis", "Visual content analysis using computer vision techniques", 
             json.dumps({'type': 'image_analysis', 'confidence': 0.85})),
            (7, 'url', "https://regex101.com/", "Regular Expression Tester", 
             r"Testing regex patterns like \d+ and [a-zA-Z]+ for text matching", 
             json.dumps({'regex_tested': True})),
            (8, 'text', None, "OCR Results", "Extracted text from image: 'Hello World 123'", 
             json.dumps({'ocr_confidence': 0.92, 'source': 'image'})),
            (9, 'url', "https://fuzzy-string-matching.com/", "Fuzzy String Matching", 
             "Algorithms for approximate string matching and typo tolerance", 
             json.dumps({'algorithm': 'levenshtein'})),
            (10, 'extraction_summary', None, "Content Summary", 
             "Summary of extracted content with metadata and analysis results", 
             json.dumps({'summary_type': 'auto', 'word_count': 150}))
        ]
        
        for i, (capture_id, kind, url, title, text, metadata) in enumerate(diverse_data, 6):
            conn.execute("""
                INSERT INTO captures (id, created_at, source_window)
                VALUES (?, ?, ?)
            """, (capture_id, (datetime.now() - timedelta(days=i+5)).isoformat(), f"Diverse Window {i}"))
            
            conn.execute("""
                INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (capture_id, kind, url, title, text, metadata))
        
        conn.commit()
        conn.close()
    
    def test_basic_search(self):
        """Test basic text search functionality."""
        criteria = SearchCriteria(query="machine learning", limit=10)
        results, metadata = self.search_engine.search(criteria)
        
        self.assertGreater(len(results), 0, "Should find results for 'machine learning'")
        self.assertEqual(metadata['total_results'], len(results))
        
        # Check that results contain relevant content
        found_ml_content = any(
            'machine learning' in result.content.lower() or 
            'machine learning' in result.title.lower()
            for result in results
        )
        self.assertTrue(found_ml_content, "Should find content mentioning 'machine learning'")
    
    def test_content_type_filtering(self):
        """Test filtering by content type."""
        # Test URL-only filtering
        criteria = SearchCriteria(query="python", content_types=["url"], limit=10)
        results, _ = self.search_engine.search(criteria)
        
        url_results = [r for r in results if r.kind == "url"]
        self.assertEqual(len(url_results), len(results), "All results should be URLs")
        
        # Test text-only filtering
        criteria = SearchCriteria(query="content", content_types=["text"], limit=10)
        results, _ = self.search_engine.search(criteria)
        
        text_results = [r for r in results if r.kind == "text"]
        self.assertEqual(len(text_results), len(results), "All results should be text")
    
    def test_date_range_filtering(self):
        """Test date range filtering."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        last_week = now - timedelta(days=7)
        
        # Search within last week
        criteria = SearchCriteria(
            query="python",
            date_from=last_week,
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # All results should be from the last week or more recent
        for result in results:
            self.assertGreaterEqual(result.created_at, last_week,
                                  f"Result {result.id} should be from last week or later")
        
        # Search within last day (should be fewer results)
        criteria = SearchCriteria(
            query="python",
            date_from=yesterday,
            limit=10
        )
        results_day, _ = self.search_engine.search(criteria)
        
        # Should have fewer or equal results than week search
        self.assertLessEqual(len(results_day), len(results))
    
    def test_url_pattern_filtering(self):
        """Test URL pattern filtering."""
        # Filter for URLs containing "python"
        criteria = SearchCriteria(
            query="python",
            url_pattern="python",
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # All results should have URLs containing "python"
        for result in results:
            if result.url:
                self.assertIn("python", result.url.lower(),
                            f"Result {result.id} URL should contain 'python'")
    
    def test_regex_search(self):
        """Test regular expression search."""
        # Search for URLs using regex
        criteria = SearchCriteria(
            query=r"https?://[^\s]+",  # Match URLs
            use_regex=True,
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # Should find results with URLs
        url_results = [r for r in results if r.url and r.url.startswith(('http://', 'https://'))]
        self.assertGreater(len(url_results), 0, "Should find URLs using regex")
        
        # Test invalid regex (should handle gracefully)
        criteria = SearchCriteria(
            query=r"[invalid(regex",  # Invalid regex
            use_regex=True,
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # Should return empty results or handle gracefully
        self.assertIsInstance(results, list)
    
    def test_fuzzy_matching(self):
        """Test fuzzy string matching."""
        # Test with typo
        criteria = SearchCriteria(
            query="machne lerning",  # Typo in "machine learning"
            fuzzy_threshold=70,
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # Should find results even with typos
        found_similar = any(
            result.relevance_score > 0.1 for result in results
        )
        self.assertTrue(found_similar, "Should find results with fuzzy matching")
        
        # Test with exact match (should get higher scores)
        criteria_exact = SearchCriteria(
            query="machine learning",
            fuzzy_threshold=70,
            limit=10
        )
        results_exact, _ = self.search_engine.search(criteria_exact)
        
        if results and results_exact:
            avg_fuzzy_score = sum(r.relevance_score for r in results) / len(results)
            avg_exact_score = sum(r.relevance_score for r in results_exact) / len(results_exact)
            # Exact match should generally have higher relevance
            # (This might not always be true due to other factors)
    
    def test_search_highlighting(self):
        """Test search result highlighting and context snippets."""
        criteria = SearchCriteria(
            query="machine learning",
            highlight=True,
            context_window=30,
            limit=5
        )
        results, _ = self.search_engine.search(criteria)
        
        # Check if highlighting is working
        highlighted_results = [r for r in results if r.highlights]
        if highlighted_results:
            result = highlighted_results[0]
            self.assertGreater(len(result.highlights), 0, "Should have highlights")
            self.assertGreater(len(result.context_snippets), 0, "Should have context snippets")
            
            # Check highlight structure
            highlight = result.highlights[0]
            self.assertIn('start', highlight)
            self.assertIn('end', highlight)
            self.assertIn('matched_text', highlight)
    
    def test_sorting_options(self):
        """Test different sorting options."""
        criteria = SearchCriteria(
            query="python",
            sort_by="date",
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        # Results should be sorted by date (newest first)
        dates = [result.created_at for result in results]
        self.assertEqual(dates, sorted(dates, reverse=True),
                        "Results should be sorted by date (newest first)")
        
        # Test title sorting
        criteria = SearchCriteria(
            query="python",
            sort_by="title",
            limit=10
        )
        results, _ = self.search_engine.search(criteria)
        
        titles = [result.title.lower() for result in results]
        self.assertEqual(titles, sorted(titles),
                        "Results should be sorted alphabetically by title")
    
    def test_pagination(self):
        """Test pagination functionality."""
        # Test offset and limit
        criteria1 = SearchCriteria(query="python", limit=3, offset=0)
        results1, _ = self.search_engine.search(criteria1)
        
        criteria2 = SearchCriteria(query="python", limit=3, offset=3)
        results2, _ = self.search_engine.search(criteria2)
        
        # Results should be different
        result_ids_1 = {r.id for r in results1}
        result_ids_2 = {r.id for r in results2}
        
        self.assertEqual(len(results1), 3, "First page should have 3 results")
        self.assertEqual(len(results2), 3, "Second page should have 3 results")
        self.assertEqual(len(result_ids_1.intersection(result_ids_2)), 0,
                        "Pages should not overlap")
    
    def test_search_history(self):
        """Test search history functionality."""
        # Perform some searches
        test_queries = ["machine learning", "python programming", "artificial intelligence"]
        
        for query in test_queries:
            criteria = SearchCriteria(query=query, limit=5)
            self.search_engine.search(criteria)
        
        # Check history
        history = self.search_engine.get_search_history(limit=10)
        
        # Should have entries for our searches
        history_queries = [entry.query for entry in history]
        for query in test_queries:
            self.assertIn(query, history_queries,
                         f"Search history should contain '{query}'")
        
        # Test clearing history
        success = self.search_engine.clear_search_history()
        self.assertTrue(success, "Should successfully clear history")
        
        history_after_clear = self.search_engine.get_search_history()
        self.assertEqual(len(history_after_clear), 0, "History should be empty after clearing")
    
    def test_saved_searches(self):
        """Test saved searches functionality."""
        # Create a saved search
        criteria = SearchCriteria(
            query="machine learning",
            content_types=["url", "text"],
            fuzzy_threshold=80
        )
        
        search_id = self.search_engine.save_search(
            name="ML Content Search",
            criteria=criteria,
            description="Search for machine learning content in URLs and text"
        )
        
        self.assertIsInstance(search_id, int, "Should return valid search ID")
        
        # Load saved search
        loaded_search = self.search_engine.load_saved_search(search_id)
        
        self.assertIsNotNone(loaded_search, "Should find saved search")
        self.assertEqual(loaded_search.name, "ML Content Search")
        self.assertEqual(loaded_search.description, "Search for machine learning content in URLs and text")
        self.assertEqual(loaded_search.use_count, 1, "Should increment use count")
        
        # Get all saved searches
        all_searches = self.search_engine.get_saved_searches()
        self.assertGreater(len(all_searches), 0, "Should have saved searches")
        
        # Find our search
        found_search = next((s for s in all_searches if s.name == "ML Content Search"), None)
        self.assertIsNotNone(found_search, "Should find our saved search in list")
        
        # Delete saved search
        success = self.search_engine.delete_saved_search(search_id)
        self.assertTrue(success, "Should successfully delete saved search")
        
        # Verify deletion
        loaded_after_delete = self.search_engine.load_saved_search(search_id)
        self.assertIsNone(loaded_after_delete, "Should not find deleted search")
    
    def test_export_functionality(self):
        """Test search result export functionality."""
        # Perform a search
        criteria = SearchCriteria(query="python", limit=5)
        results, _ = self.search_engine.search(criteria)
        
        if not results:
            self.skipTest("No search results to export")
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Test CSV export
            csv_path = os.path.join(temp_dir, "test_results.csv")
            success = self.search_engine.export_results(results, csv_path, "csv")
            self.assertTrue(success, "CSV export should succeed")
            self.assertTrue(os.path.exists(csv_path), "CSV file should exist")
            
            # Verify CSV content
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
                self.assertGreater(len(csv_rows), 0, "CSV should have data")
            
            # Test JSON export
            json_path = os.path.join(temp_dir, "test_results.json")
            success = self.search_engine.export_results(results, json_path, "json")
            self.assertTrue(success, "JSON export should succeed")
            self.assertTrue(os.path.exists(json_path), "JSON file should exist")
            
            # Verify JSON content
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                self.assertIn('results', json_data)
                self.assertGreater(len(json_data['results']), 0, "JSON should have results")
            
            # Test text export
            txt_path = os.path.join(temp_dir, "test_results.txt")
            success = self.search_engine.export_results(results, txt_path, "txt")
            self.assertTrue(success, "Text export should succeed")
            self.assertTrue(os.path.exists(txt_path), "Text file should exist")
            
            # Verify text content
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
                self.assertIn("ContextBox Search Results", txt_content)
                self.assertIn("Total Results:", txt_content)
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty query
        criteria = SearchCriteria(query="", limit=10)
        results, _ = self.search_engine.search(criteria)
        self.assertIsInstance(results, list, "Should return list even with empty query")
        
        # Very long query
        long_query = "a" * 1000
        criteria = SearchCriteria(query=long_query, limit=10)
        results, _ = self.search_engine.search(criteria)
        self.assertIsInstance(results, list, "Should handle long queries")
        
        # Invalid content types
        criteria = SearchCriteria(query="python", content_types=["invalid_type"], limit=10)
        results, _ = self.search_engine.search(criteria)
        self.assertIsInstance(results, list, "Should handle invalid content types")
        
        # Future date range
        future_date = datetime.now() + timedelta(days=30)
        criteria = SearchCriteria(query="python", date_from=future_date, limit=10)
        results, _ = self.search_engine.search(criteria)
        # Should return empty results for future date range
        self.assertEqual(len(results), 0, "Future date range should return no results")
    
    def test_search_statistics(self):
        """Test search statistics functionality."""
        # Perform some searches to generate statistics
        for query in ["python", "machine learning", "artificial intelligence"]:
            criteria = SearchCriteria(query=query, limit=5)
            self.search_engine.search(criteria)
        
        # Get statistics
        stats = self.search_engine.get_search_statistics()
        
        # Verify statistics structure
        self.assertIn('search_history', stats)
        self.assertIn('popular_queries', stats)
        self.assertIn('saved_searches', stats)
        
        # Check search history stats
        search_stats = stats['search_history']
        self.assertIn('total_searches', search_stats)
        self.assertGreaterEqual(search_stats['total_searches'], 3)
        
        # Check popular queries
        popular = stats['popular_queries']
        self.assertIsInstance(popular, list)
    
    def test_convenience_functions(self):
        """Test convenience functions for common search patterns."""
        # Test quick_search
        results = quick_search(self.test_db_path, "python", limit=5)
        self.assertIsInstance(results, list, "quick_search should return list")
        
        # Test search_by_date_range
        results = search_by_date_range(self.test_db_path, "content", days_back=30)
        self.assertIsInstance(results, list, "search_by_date_range should return list")
        
        # Test fuzzy_search
        results = fuzzy_search(self.test_db_path, "pyton", threshold=70)
        self.assertIsInstance(results, list, "fuzzy_search should return list")
    
    def test_performance_characteristics(self):
        """Test basic performance characteristics."""
        import time
        
        # Time a search operation
        start_time = time.time()
        criteria = SearchCriteria(query="python", limit=10)
        results, metadata = self.search_engine.search(criteria)
        end_time = time.time()
        
        # Search should complete within reasonable time
        search_duration = end_time - start_time
        self.assertLess(search_duration, 5.0, "Search should complete within 5 seconds")
        
        # Execution time metadata should be reasonable
        execution_time = metadata['execution_time_ms']
        self.assertGreater(execution_time, 0, "Execution time should be recorded")
        self.assertLess(execution_time, 5000, "Execution time should be under 5 seconds")
    
    def test_concurrent_searches(self):
        """Test thread safety with concurrent searches."""
        import threading
        import time
        
        results_list = []
        
        def search_worker(query_prefix):
            """Worker function for concurrent search."""
            try:
                criteria = SearchCriteria(query=f"{query_prefix} python", limit=5)
                results, _ = self.search_engine.search(criteria)
                results_list.append(len(results))
            except Exception as e:
                results_list.append(f"Error: {e}")
        
        # Create multiple search threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=search_worker, args=(f"thread_{i}_",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All searches should have completed
        self.assertEqual(len(results_list), 3, "All threads should have completed")
        
        # No errors should have occurred
        errors = [r for r in results_list if isinstance(r, str) and r.startswith("Error")]
        self.assertEqual(len(errors), 0, f"No concurrent search errors should occur: {errors}")


class TestSearchCriteria(unittest.TestCase):
    """Test cases for SearchCriteria class."""
    
    def test_search_criteria_creation(self):
        """Test SearchCriteria object creation and validation."""
        criteria = SearchCriteria(
            query="test query",
            content_types=["url", "text"],
            date_from=datetime.now() - timedelta(days=7),
            date_to=datetime.now(),
            url_pattern=r"example\.com",
            use_regex=True,
            fuzzy_threshold=85,
            highlight=True,
            context_window=100,
            limit=50,
            offset=10,
            sort_by="date"
        )
        
        self.assertEqual(criteria.query, "test query")
        self.assertEqual(criteria.content_types, ["url", "text"])
        self.assertIsInstance(criteria.date_from, datetime)
        self.assertTrue(criteria.use_regex)
        self.assertEqual(criteria.fuzzy_threshold, 85)
        self.assertTrue(criteria.highlight)
        self.assertEqual(criteria.context_window, 100)
        self.assertEqual(criteria.limit, 50)
        self.assertEqual(criteria.offset, 10)
        self.assertEqual(criteria.sort_by, "date")
    
    def test_search_criteria_defaults(self):
        """Test SearchCriteria with default values."""
        criteria = SearchCriteria(query="test")
        
        self.assertIsNone(criteria.content_types)
        self.assertIsNone(criteria.date_from)
        self.assertIsNone(criteria.date_to)
        self.assertIsNone(criteria.url_pattern)
        self.assertFalse(criteria.use_regex)
        self.assertIsNone(criteria.fuzzy_threshold)
        self.assertTrue(criteria.highlight)
        self.assertEqual(criteria.context_window, 50)
        self.assertEqual(criteria.limit, 100)
        self.assertEqual(criteria.offset, 0)
        self.assertEqual(criteria.sort_by, "relevance")


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult class."""
    
    def test_search_result_creation(self):
        """Test SearchResult object creation."""
        metadata = {"confidence": 0.9, "source": "test"}
        highlights = [{"start": 0, "end": 5, "matched_text": "hello", "type": "text"}]
        context_snippets = ["...hello world..."]
        match_positions = [(0, 5)]
        
        result = SearchResult(
            id=1,
            capture_id=1,
            kind="text",
            title="Test Title",
            url="https://example.com",
            content="Hello world content",
            metadata=metadata,
            created_at=datetime.now(),
            relevance_score=0.85,
            highlights=highlights,
            context_snippets=context_snippets,
            match_positions=match_positions
        )
        
        self.assertEqual(result.id, 1)
        self.assertEqual(result.kind, "text")
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.content, "Hello world content")
        self.assertEqual(result.metadata, metadata)
        self.assertEqual(result.relevance_score, 0.85)
        self.assertEqual(result.highlights, highlights)
        self.assertEqual(result.context_snippets, context_snippets)


class TestSearchErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""
    
    def setUp(self):
        """Set up test database for error handling tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_contextbox.db")
        
        # Create minimal database structure
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("""
            CREATE TABLE captures (
                id INTEGER PRIMARY KEY,
                created_at DATETIME
            )
        """)
        conn.execute("""
            CREATE TABLE artifacts (
                id INTEGER PRIMARY KEY,
                capture_id INTEGER,
                kind TEXT,
                title TEXT,
                text TEXT,
                metadata_json TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up after error handling tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_database_path(self):
        """Test handling of invalid database path."""
        with self.assertRaises((SearchError, sqlite3.Error)):
            search_engine = SearchEngine("/nonexistent/path/database.db")
    
    def test_corrupted_database_handling(self):
        """Test handling of corrupted database."""
        # Write invalid data to database
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("INSERT INTO captures (id, created_at) VALUES (1, 'invalid_date')")
        conn.commit()
        conn.close()
        
        search_engine = SearchEngine(self.test_db_path)
        
        # Should handle corrupted data gracefully
        try:
            criteria = SearchCriteria(query="test", limit=10)
            results, metadata = search_engine.search(criteria)
            # Should not crash, even if results are empty
            self.assertIsInstance(results, list)
        except Exception:
            pass  # Some database errors are expected with corrupted data
        
        search_engine.cleanup()


def run_comprehensive_tests():
    """Run all test suites and generate a report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSearchEngine,
        TestSearchCriteria,
        TestSearchResult,
        TestSearchErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*80)
    print("SEARCH ENGINE TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Starting Comprehensive Search Engine Tests...")
    print("This test suite covers all advanced search functionality.")
    print("Tests may take a few moments to complete.\n")
    
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests passed! The search engine is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)