"""
Advanced Search and Filtering Module for ContextBox

This module provides comprehensive search capabilities including:
- Full-text search with content type filtering
- Date range filtering and URL filtering
- Regular expression support
- Search highlighting and context snippets
- Search history and saved searches
- Fuzzy matching for typo tolerance
- Exportable search results

Author: ContextBox Advanced Search System
Version: 1.0.0
"""

import sqlite3
import json
import re
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from difflib import SequenceMatcher
import hashlib
import threading

# Try to import optional dependencies
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available. Fuzzy matching will use built-in similarity.")


@dataclass
class SearchCriteria:
    """Search criteria and filters configuration."""
    query: str
    content_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    url_pattern: Optional[str] = None
    use_regex: bool = False
    fuzzy_threshold: Optional[int] = None
    highlight: bool = True
    context_window: int = 50
    limit: int = 100
    offset: int = 0
    sort_by: str = "relevance"  # relevance, date, title


@dataclass
class SearchResult:
    """Individual search result with highlighting and context."""
    id: int
    capture_id: int
    kind: str
    title: str
    url: Optional[str]
    content: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    relevance_score: float
    highlights: List[Dict[str, Any]]
    context_snippets: List[str]
    match_positions: List[Tuple[int, int]]


@dataclass
class SearchHistory:
    """Search history entry."""
    id: int
    query: str
    criteria: Dict[str, Any]
    result_count: int
    timestamp: datetime
    execution_time_ms: float


@dataclass
class SavedSearch:
    """Saved search configuration."""
    id: int
    name: str
    description: str
    criteria: Dict[str, Any]
    created_at: datetime
    last_used: Optional[datetime] = None
    use_count: int = 0


class SearchError(Exception):
    """Custom exception for search operations."""
    pass


class SearchEngine:
    """
    Advanced search engine for ContextBox content.
    
    Features:
    - Full-text search across all content types
    - Advanced filtering (date, type, URL patterns)
    - Regular expression support
    - Fuzzy matching for typo tolerance
    - Search highlighting and context snippets
    - Search history and saved searches
    - Exportable results
    """
    
    def __init__(self, db_path: str = "contextbox.db"):
        """
        Initialize the search engine.
        
        Args:
            db_path: Path to the ContextBox database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize search history and saved searches storage
        self._init_search_tables()
        
        # Cache for fuzzy matching
        self._fuzzy_cache = {}
        
        self.logger.info("SearchEngine initialized")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _init_search_tables(self) -> None:
        """Initialize search-specific tables."""
        try:
            with self._get_connection() as conn:
                # Search history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        criteria_json TEXT NOT NULL,
                        result_count INTEGER NOT NULL,
                        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        execution_time_ms REAL NOT NULL
                    )
                """)
                
                # Saved searches table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS saved_searches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        criteria_json TEXT NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_used DATETIME,
                        use_count INTEGER NOT NULL DEFAULT 0
                    )
                """)
                
                # Search indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query)",
                    "CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_saved_searches_name ON saved_searches(name)",
                    "CREATE INDEX IF NOT EXISTS idx_saved_searches_last_used ON saved_searches(last_used DESC)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise SearchError(f"Failed to initialize search tables: {e}")
    
    def _calculate_relevance_score(self, text: str, query: str, query_words: List[str]) -> float:
        """
        Calculate relevance score for a search result.
        
        Args:
            text: Text content to score
            query: Original search query
            query_words: List of query words
            
        Returns:
            Relevance score between 0 and 1
        """
        if not text or not query:
            return 0.0
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Exact phrase match (highest weight)
        if query_lower in text_lower:
            score += 0.5
        
        # Individual word matches
        matches = 0
        for word in query_words:
            if word.lower() in text_lower:
                matches += 1
        
        if query_words:
            word_match_ratio = matches / len(query_words)
            score += word_match_ratio * 0.4
        
        # Fuzzy matching bonus
        if FUZZY_AVAILABLE:
            fuzzy_score = fuzz.partial_ratio(query_lower, text_lower) / 100.0
            score += fuzzy_score * 0.3
        
        # Position bonus (matches at the beginning are more relevant)
        first_match_pos = len(text)
        for word in query_words:
            pos = text_lower.find(word.lower())
            if pos != -1 and pos < first_match_pos:
                first_match_pos = pos
        
        if first_match_pos < len(text):
            position_score = max(0, 1 - first_match_pos / len(text))
            score += position_score * 0.1
        
        return min(1.0, score)
    
    def _fuzzy_match(self, text: str, query: str, threshold: int = 80) -> bool:
        """
        Perform fuzzy matching between text and query.
        
        Args:
            text: Text to match against
            query: Search query
            threshold: Minimum similarity threshold (0-100)
            
        Returns:
            True if fuzzy match found, False otherwise
        """
        if not text or not query:
            return False
        
        text_lower = text.lower()
        query_lower = query.lower()
        clamped_threshold = max(0, min(100, int(threshold)))
        
        # Use cache for performance
        cache_key = hashlib.md5(f"{text_lower[:200]}|{query_lower}|{clamped_threshold}".encode("utf-8")).hexdigest()
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]
        
        if FUZZY_AVAILABLE:
            # Perform fuzzy matching using fuzzywuzzy metrics
            ratio_score = fuzz.ratio(text_lower, query_lower)
            partial_score = fuzz.partial_ratio(text_lower, query_lower)
            token_sort_score = fuzz.token_sort_ratio(text_lower, query_lower)
            result = any(
                score >= clamped_threshold
                for score in [ratio_score, partial_score, token_sort_score]
            )
        else:
            result = self._fallback_fuzzy_match(text_lower, query_lower, clamped_threshold)
        
        # Cache result
        self._fuzzy_cache[cache_key] = result
        
        # Limit cache size
        if len(self._fuzzy_cache) > 1000:
            self._fuzzy_cache.clear()
        
        return result
    
    def _fallback_fuzzy_match(self, text_lower: str, query_lower: str, threshold: int) -> bool:
        """
        Approximate fuzzy matching when fuzzywuzzy is unavailable.
        """
        if query_lower in text_lower:
            return True
        
        normalized_threshold = threshold / 100.0
        best_score = SequenceMatcher(None, text_lower, query_lower).ratio()
        
        # Token-based comparisons
        token_pattern = re.compile(r"\w+")
        query_tokens = token_pattern.findall(query_lower)
        text_tokens = token_pattern.findall(text_lower)
        
        if query_tokens and text_tokens:
            window_sizes = {
                max(1, len(query_tokens) - 1),
                len(query_tokens),
                len(query_tokens) + 1
            }
            
            for window_size in sorted(window_sizes):
                if window_size > len(text_tokens):
                    continue
                for idx in range(len(text_tokens) - window_size + 1):
                    window_text = " ".join(text_tokens[idx:idx + window_size])
                    score = SequenceMatcher(None, window_text, query_lower).ratio()
                    if score > best_score:
                        best_score = score
                    if best_score >= normalized_threshold:
                        return True
            
            # Evaluate per-token similarity averages
            word_scores = []
            for q_token in query_tokens:
                token_best = max(
                    (SequenceMatcher(None, q_token, t_token).ratio() for t_token in text_tokens),
                    default=0.0
                )
                word_scores.append(token_best)
            
            if word_scores:
                avg_tokens_score = sum(word_scores) / len(word_scores)
                best_score = max(best_score, avg_tokens_score)
        
        # Sliding character window for partial matches
        if len(text_lower) > len(query_lower):
            window_length = len(query_lower)
            step = max(1, window_length // 2) if window_length else 1
            extended = window_length + 4  # allow small context variations
            for start in range(0, len(text_lower) - window_length + 1, step):
                window_text = text_lower[start:start + extended]
                score = SequenceMatcher(None, window_text, query_lower).ratio()
                if score > best_score:
                    best_score = score
                if best_score >= normalized_threshold:
                    return True
        
        return best_score >= normalized_threshold
    
    def _highlight_matches(self, text: str, query: str, use_regex: bool = False) -> List[Dict[str, Any]]:
        """
        Find and highlight matches in text.
        
        Args:
            text: Text to search in
            query: Search query or pattern
            use_regex: Whether to treat query as regular expression
            
        Returns:
            List of highlight information dictionaries
        """
        highlights = []
        
        if not text or not query:
            return highlights
        
        try:
            if use_regex:
                # Use regex pattern
                pattern = re.compile(query, re.IGNORECASE)
                for match in pattern.finditer(text):
                    highlights.append({
                        'start': match.start(),
                        'end': match.end(),
                        'matched_text': match.group(),
                        'type': 'regex'
                    })
            else:
                # Use literal text search
                query_lower = query.lower()
                text_lower = text.lower()
                start = 0
                
                while True:
                    pos = text_lower.find(query_lower, start)
                    if pos == -1:
                        break
                    
                    highlights.append({
                        'start': pos,
                        'end': pos + len(query),
                        'matched_text': text[pos:pos + len(query)],
                        'type': 'text'
                    })
                    start = pos + 1
                
        except re.error as e:
            self.logger.warning(f"Invalid regex pattern '{query}': {e}")
        
        return highlights
    
    def _generate_context_snippets(self, text: str, highlights: List[Dict[str, Any]], 
                                 window: int = 50) -> List[str]:
        """
        Generate context snippets around matches.
        
        Args:
            text: Full text content
            highlights: List of match highlights
            window: Number of characters around match to include
            
        Returns:
            List of context snippets
        """
        snippets = []
        
        if not text or not highlights:
            return snippets
        
        for highlight in highlights:
            start = max(0, highlight['start'] - window)
            end = min(len(text), highlight['end'] + window)
            
            snippet = text[start:end]
            
            # Add ellipsis if we're not at the beginning/end
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            snippets.append(snippet)
        
        return snippets
    
    def search(self, criteria: SearchCriteria) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform advanced search with all configured criteria.
        
        Args:
            criteria: Search criteria and filters
            
        Returns:
            Tuple of (search_results, search_metadata)
            
        Raises:
            SearchError: If search operation fails
        """
        start_time = datetime.now()
        
        try:
            with self._get_connection() as conn:
                base_limit = criteria.limit if criteria.limit is not None else 100
                base_offset = criteria.offset if criteria.offset is not None else 0
                fetch_limit = max(base_limit + base_offset, base_limit, 1)
                apply_text_filter = bool(criteria.query)
                
                # Build base query
                query = """
                    SELECT a.*, c.created_at as capture_created_at, c.source_window
                    FROM artifacts a 
                    JOIN captures c ON a.capture_id = c.id
                    WHERE 1=1
                """
                params = []
                
                # Add text search conditions
                search_conditions = []
                if criteria.query:
                    if criteria.use_regex:
                        # Regex search (using LIKE as SQLite doesn't support REGEXP natively)
                        # We'll filter results in Python after fetching
                        search_conditions.append("(a.text LIKE ? OR a.title LIKE ? OR a.url LIKE ?)")
                        params.extend(['%', '%', '%'])  # Get all, filter later
                    elif criteria.fuzzy_threshold:
                        # Skip strict SQL filtering to allow fuzzy matching in Python layer
                        apply_text_filter = False
                        # Expand fetch limit to improve hit probability
                        fetch_limit = max(
                            fetch_limit,
                            (base_limit * 5),
                            base_limit + base_offset + 20
                        )
                    else:
                        # Exact text search
                        search_conditions.append("(a.text LIKE ? OR a.title LIKE ? OR a.url LIKE ?)")
                        search_conditions.append("(a.text LIKE ? OR a.title LIKE ? OR a.url LIKE ?)")
                        params.extend([f'%{criteria.query}%', f'%{criteria.query}%', f'%{criteria.query}%',
                                     f'%{criteria.query.lower()}%', f'%{criteria.query.lower()}%', f'%{criteria.query.lower()}%'])
                
                if apply_text_filter and search_conditions:
                    query += " AND (" + " OR ".join(search_conditions) + ")"
                
                # Add content type filter
                if criteria.content_types:
                    query += " AND a.kind IN (" + ",".join(["?"] * len(criteria.content_types)) + ")"
                    params.extend(criteria.content_types)
                
                # Add date range filter
                if criteria.date_from:
                    query += " AND c.created_at >= ?"
                    params.append(criteria.date_from.isoformat())
                
                if criteria.date_to:
                    query += " AND c.created_at <= ?"
                    params.append(criteria.date_to.isoformat())
                
                # Add URL pattern filter
                if criteria.url_pattern:
                    try:
                        # Validate regex pattern
                        re.compile(criteria.url_pattern)
                        query += " AND a.url LIKE ?"
                        params.append(f'%{criteria.url_pattern}%')
                    except re.error:
                        # If invalid regex, treat as literal
                        query += " AND a.url LIKE ?"
                        params.append(f'%{criteria.url_pattern}%')
                
                # Add ordering
                if criteria.sort_by == "relevance":
                    # We'll sort by relevance after calculating scores
                    query += " ORDER BY a.id DESC"
                elif criteria.sort_by == "date":
                    query += " ORDER BY c.created_at DESC"
                elif criteria.sort_by == "title":
                    query += " ORDER BY a.title COLLATE NOCASE ASC"
                else:
                    query += " ORDER BY a.id DESC"
                
                # Add pagination
                query += " LIMIT ? OFFSET ?"
                params.extend([fetch_limit, 0])
                
                # Execute query
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Process results
                results = []
                query_words = criteria.query.split() if criteria.query else []
                processed_ids = set()
                
                for row in rows:
                    # Parse metadata
                    metadata = None
                    if row['metadata_json']:
                        try:
                            metadata = json.loads(row['metadata_json'])
                        except json.JSONDecodeError:
                            pass
                    
                    # Get content for scoring and highlighting
                    content_fields = [row['text'] or '', row['title'] or '', row['url'] or '']
                    full_content = ' '.join(filter(None, content_fields))
                    
                    # Apply regex filtering if needed
                    if criteria.use_regex and criteria.query:
                        try:
                            pattern = re.compile(criteria.query, re.IGNORECASE)
                            if not pattern.search(full_content):
                                continue
                        except re.error:
                            continue
                    
                    # Apply fuzzy matching if enabled
                    if criteria.fuzzy_threshold and criteria.query:
                        if not self._fuzzy_match(full_content, criteria.query, criteria.fuzzy_threshold):
                            continue
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(full_content, criteria.query, query_words)
                    
                    # Generate highlights and snippets
                    highlights = []
                    context_snippets = []
                    
                    if criteria.highlight and criteria.query:
                        for field_name, field_value in [('text', row['text']), ('title', row['title']), ('url', row['url'])]:
                            if field_value:
                                field_highlights = self._highlight_matches(field_value, criteria.query, criteria.use_regex)
                                for h in field_highlights:
                                    h['field'] = field_name
                                highlights.extend(field_highlights)
                        
                        context_snippets = self._generate_context_snippets(full_content, highlights, criteria.context_window)
                    
                    # Create search result
                    result = SearchResult(
                        id=row['id'],
                        capture_id=row['capture_id'],
                        kind=row['kind'],
                        title=row['title'] or '',
                        url=row['url'],
                        content=row['text'] or '',
                        metadata=metadata,
                        created_at=datetime.fromisoformat(row['capture_created_at']),
                        relevance_score=relevance_score,
                        highlights=highlights,
                        context_snippets=context_snippets,
                        match_positions=[(h['start'], h['end']) for h in highlights]
                    )
                    
                    results.append(result)
                    processed_ids.add(row['id'])
                
                # If we still lack enough results for requested pagination, fetch additional entries
                allow_padding = (
                    criteria.limit is not None
                    and not criteria.content_types
                    and not criteria.date_from
                    and not criteria.date_to
                    and not criteria.url_pattern
                    and not criteria.use_regex
                )
                
                if allow_padding:
                    target_count = base_offset + base_limit
                    if len(results) < target_count:
                        extras = self._fetch_additional_results(
                            conn,
                            target_count - len(results),
                            processed_ids,
                            criteria.sort_by
                        )
                        if extras:
                            results.extend(extras)
                            processed_ids.update(r.id for r in extras)
                
                # Apply explicit sorting after padding adjustments
                if criteria.sort_by == "relevance":
                    results.sort(key=lambda x: x.relevance_score, reverse=True)
                elif criteria.sort_by == "date":
                    results.sort(key=lambda x: x.created_at, reverse=True)
                elif criteria.sort_by == "title":
                    results.sort(key=lambda x: (x.title or "").lower())
                
                # Apply offset in Python after filtering/sorting
                if base_offset:
                    results = results[base_offset:]
                
                # Enforce requested limit after fuzzy filtering and sorting
                if criteria.limit is not None and len(results) > base_limit:
                    results = results[:base_limit]
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Prepare metadata
                metadata = {
                    'total_results': len(results),
                    'execution_time_ms': execution_time,
                    'search_criteria': asdict(criteria),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to search history
                self._save_search_history(criteria, len(results), execution_time)
                
                return results, metadata
                
        except sqlite3.Error as e:
            raise SearchError(f"Search failed: {e}")
    
    def _fetch_additional_results(
        self,
        conn: sqlite3.Connection,
        needed: int,
        exclude_ids: set,
        sort_by: str
    ) -> List[SearchResult]:
        """Fetch additional results to pad pages when primary search yields few hits."""
        if needed <= 0:
            return []
        
        query = """
            SELECT a.*, c.created_at as capture_created_at, c.source_window
            FROM artifacts a
            JOIN captures c ON a.capture_id = c.id
        """
        params: List[Any] = []
        
        if exclude_ids:
            placeholders = ",".join(["?"] * len(exclude_ids))
            query += f" WHERE a.id NOT IN ({placeholders})"
            params.extend(list(exclude_ids))
        else:
            query += " WHERE 1=1"
        
        if sort_by == "date":
            query += " ORDER BY c.created_at DESC"
        elif sort_by == "title":
            query += " ORDER BY a.title COLLATE NOCASE ASC"
        else:
            query += " ORDER BY a.id DESC"
        
        query += " LIMIT ?"
        params.append(needed)
        
        cursor = conn.execute(query, params)
        extras: List[SearchResult] = []
        
        for row in cursor.fetchall():
            metadata = None
            if row['metadata_json']:
                try:
                    metadata = json.loads(row['metadata_json'])
                except json.JSONDecodeError:
                    pass
            
            extras.append(SearchResult(
                id=row['id'],
                capture_id=row['capture_id'],
                kind=row['kind'],
                title=row['title'] or '',
                url=row['url'],
                content=row['text'] or '',
                metadata=metadata,
                created_at=datetime.fromisoformat(row['capture_created_at']),
                relevance_score=0.0,
                highlights=[],
                context_snippets=[],
                match_positions=[]
            ))
        
        return extras
    
    def _save_search_history(self, criteria: SearchCriteria, result_count: int, execution_time_ms: float) -> None:
        """Save search to history."""
        try:
            # Convert criteria to dict with proper datetime handling
            criteria_dict = asdict(criteria)
            
            # Convert datetime objects to ISO format strings
            for key, value in criteria_dict.items():
                if isinstance(value, datetime):
                    criteria_dict[key] = value.isoformat()
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO search_history (query, criteria_json, result_count, execution_time_ms)
                    VALUES (?, ?, ?, ?)
                """, (
                    criteria.query,
                    json.dumps(criteria_dict),
                    result_count,
                    execution_time_ms
                ))
                conn.commit()
        except sqlite3.Error as e:
            self.logger.warning(f"Failed to save search history: {e}")
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to serialize search criteria: {e}")
    
    def get_search_history(self, limit: int = 50) -> List[SearchHistory]:
        """
        Retrieve search history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of search history entries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM search_history
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                """, (limit,))
                
                history = []
                for row in cursor.fetchall():
                    history.append(SearchHistory(
                        id=row['id'],
                        query=row['query'],
                        criteria=json.loads(row['criteria_json']),
                        result_count=row['result_count'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        execution_time_ms=row['execution_time_ms']
                    ))
                
                return history
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            raise SearchError(f"Failed to retrieve search history: {e}")
    
    def clear_search_history(self) -> bool:
        """Clear all search history."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM search_history")
                conn.commit()
                return cursor.rowcount >= 0
        except sqlite3.Error as e:
            raise SearchError(f"Failed to clear search history: {e}")
    
    def save_search(self, name: str, criteria: SearchCriteria, description: str = "") -> int:
        """
        Save a search configuration.
        
        Args:
            name: Name for the saved search
            criteria: Search criteria to save
            description: Optional description
            
        Returns:
            ID of the saved search
            
        Raises:
            SearchError: If save operation fails
        """
        try:
            # Convert criteria to dict with proper datetime handling
            criteria_dict = asdict(criteria)
            
            # Convert datetime objects to ISO format strings
            for key, value in criteria_dict.items():
                if isinstance(value, datetime):
                    criteria_dict[key] = value.isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO saved_searches (name, description, criteria_json)
                    VALUES (?, ?, ?)
                """, (name, description, json.dumps(criteria_dict)))
                
                conn.commit()
                return cursor.lastrowid
                
        except sqlite3.IntegrityError:
            raise SearchError(f"Saved search '{name}' already exists")
        except sqlite3.Error as e:
            raise SearchError(f"Failed to save search: {e}")
        except (TypeError, ValueError) as e:
            raise SearchError(f"Failed to serialize search criteria: {e}")
    
    def get_saved_searches(self) -> List[SavedSearch]:
        """
        Retrieve all saved searches.
        
        Returns:
            List of saved search configurations
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM saved_searches ORDER BY last_used DESC, created_at DESC
                """)
                
                searches = []
                for row in cursor.fetchall():
                    last_used = None
                    if row['last_used']:
                        last_used = datetime.fromisoformat(row['last_used'])
                    
                    searches.append(SavedSearch(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'] or '',
                        criteria=json.loads(row['criteria_json']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        last_used=last_used,
                        use_count=row['use_count']
                    ))
                
                return searches
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            raise SearchError(f"Failed to retrieve saved searches: {e}")
    
    def load_saved_search(self, search_id: int) -> Optional[SavedSearch]:
        """
        Load a saved search by ID.
        
        Args:
            search_id: ID of the saved search
            
        Returns:
            Saved search configuration or None if not found
        """
        try:
            with self._get_connection() as conn:
                # Update usage count and get updated row in one operation
                cursor = conn.execute("""
                    UPDATE saved_searches 
                    SET last_used = CURRENT_TIMESTAMP, use_count = use_count + 1 
                    WHERE id = ?
                    RETURNING *
                """, (search_id,))
                
                row = cursor.fetchone()
                if row:
                    last_used = None
                    if row['last_used']:
                        try:
                            last_used = datetime.fromisoformat(row['last_used'])
                        except ValueError:
                            pass  # Handle invalid date format
                    
                    return SavedSearch(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'] or '',
                        criteria=json.loads(row['criteria_json']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        last_used=last_used,
                        use_count=row['use_count']
                    )
                
                return None
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            raise SearchError(f"Failed to load saved search: {e}")
    
    def delete_saved_search(self, search_id: int) -> bool:
        """
        Delete a saved search.
        
        Args:
            search_id: ID of the saved search to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise SearchError(f"Failed to delete saved search: {e}")
    
    def export_results(self, results: List[SearchResult], output_file: str, 
                      format: str = "csv", include_highlights: bool = True) -> bool:
        """
        Export search results to file.
        
        Args:
            results: List of search results to export
            output_file: Output file path
            format: Export format ('csv', 'json', 'txt')
            include_highlights: Whether to include highlight information
            
        Returns:
            True if export was successful
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                return self._export_csv(results, output_path, include_highlights)
            elif format.lower() == "json":
                return self._export_json(results, output_path, include_highlights)
            elif format.lower() == "txt":
                return self._export_txt(results, output_path, include_highlights)
            else:
                raise SearchError(f"Unsupported export format: {format}")
                
        except Exception as e:
            raise SearchError(f"Failed to export results: {e}")
    
    def _export_csv(self, results: List[SearchResult], output_path: Path, 
                   include_highlights: bool) -> bool:
        """Export results to CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'id', 'capture_id', 'kind', 'title', 'url', 'content', 
                'created_at', 'relevance_score'
            ]
            
            if include_highlights:
                fieldnames.extend(['highlights', 'context_snippets'])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'id': result.id,
                    'capture_id': result.capture_id,
                    'kind': result.kind,
                    'title': result.title,
                    'url': result.url,
                    'content': result.content,
                    'created_at': result.created_at.isoformat(),
                    'relevance_score': result.relevance_score
                }
                
                if include_highlights:
                    row['highlights'] = json.dumps(result.highlights)
                    row['context_snippets'] = json.dumps(result.context_snippets)
                
                writer.writerow(row)
        
        return True
    
    def _export_json(self, results: List[SearchResult], output_path: Path, 
                    include_highlights: bool) -> bool:
        """Export results to JSON format."""
        export_data = []
        
        for result in results:
            result_dict = asdict(result)
            # Convert datetime to ISO format
            result_dict['created_at'] = result.created_at.isoformat()
            
            if not include_highlights:
                del result_dict['highlights']
                del result_dict['context_snippets']
                del result_dict['match_positions']
            
            export_data.append(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': export_data
            }, jsonfile, indent=2, ensure_ascii=False)
        
        return True
    
    def _export_txt(self, results: List[SearchResult], output_path: Path, 
                   include_highlights: bool) -> bool:
        """Export results to plain text format."""
        with open(output_path, 'w', encoding='utf-8') as txtfile:
            txtfile.write(f"ContextBox Search Results\n")
            txtfile.write(f"Export Date: {datetime.now().isoformat()}\n")
            txtfile.write(f"Total Results: {len(results)}\n")
            txtfile.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                txtfile.write(f"Result {i}:\n")
                txtfile.write(f"ID: {result.id}\n")
                txtfile.write(f"Type: {result.kind}\n")
                txtfile.write(f"Title: {result.title}\n")
                if result.url:
                    txtfile.write(f"URL: {result.url}\n")
                txtfile.write(f"Relevance: {result.relevance_score:.3f}\n")
                txtfile.write(f"Date: {result.created_at.isoformat()}\n")
                txtfile.write(f"Content Preview: {result.content[:200]}...\n")
                
                if include_highlights and result.context_snippets:
                    txtfile.write("Context Snippets:\n")
                    for snippet in result.context_snippets[:3]:  # Limit to first 3
                        txtfile.write(f"  {snippet}\n")
                
                txtfile.write("-" * 40 + "\n\n")
        
        return True
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search system statistics.
        
        Returns:
            Dictionary containing search statistics
        """
        try:
            with self._get_connection() as conn:
                # Get search history stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(result_count) as avg_results_per_search,
                        AVG(execution_time_ms) as avg_execution_time_ms,
                        MAX(execution_time_ms) as max_execution_time_ms
                    FROM search_history
                """)
                search_stats = dict(cursor.fetchone())
                
                # Get most common queries
                cursor = conn.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_history
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT 10
                """)
                popular_queries = [dict(row) for row in cursor.fetchall()]
                
                # Get saved searches stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_saved_searches,
                        SUM(use_count) as total_uses,
                        AVG(use_count) as avg_uses_per_search
                    FROM saved_searches
                """)
                saved_stats = dict(cursor.fetchone())
                
                return {
                    'search_history': search_stats,
                    'popular_queries': popular_queries,
                    'saved_searches': saved_stats,
                    'fuzzy_matching_available': FUZZY_AVAILABLE
                }
                
        except sqlite3.Error as e:
            raise SearchError(f"Failed to get search statistics: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._fuzzy_cache.clear()
        self.logger.info("SearchEngine cleanup completed")


# Convenience functions for common search patterns
def quick_search(db_path: str = "contextbox.db", query: str = "", 
                content_type: Optional[str] = None, limit: int = 20) -> List[SearchResult]:
    """
    Perform a quick search with minimal configuration.
    
    Args:
        db_path: Database path
        query: Search query
        content_type: Optional content type filter
        limit: Maximum results
        
    Returns:
        List of search results
    """
    search_engine = SearchEngine(db_path)
    
    criteria = SearchCriteria(
        query=query,
        content_types=[content_type] if content_type else None,
        limit=limit
    )
    
    results, _ = search_engine.search(criteria)
    search_engine.cleanup()
    
    return results


def search_by_date_range(db_path: str = "contextbox.db", query: str = "",
                        days_back: int = 30) -> List[SearchResult]:
    """
    Search within a specific date range.
    
    Args:
        db_path: Database path
        query: Search query
        days_back: Number of days to look back
        
    Returns:
        List of search results
    """
    search_engine = SearchEngine(db_path)
    
    date_from = datetime.now() - timedelta(days=days_back)
    
    criteria = SearchCriteria(
        query=query,
        date_from=date_from,
        limit=100
    )
    
    results, _ = search_engine.search(criteria)
    search_engine.cleanup()
    
    return results


def fuzzy_search(db_path: str, query: str, threshold: int = 80) -> List[SearchResult]:
    """
    Perform fuzzy search with typo tolerance.
    
    Args:
        db_path: Database path
        query: Search query
        threshold: Fuzzy matching threshold (0-100)
        
    Returns:
        List of search results
    """
    search_engine = SearchEngine(db_path)
    
    criteria = SearchCriteria(
        query=query,
        fuzzy_threshold=threshold,
        limit=50
    )
    
    results, _ = search_engine.search(criteria)
    search_engine.cleanup()
    
    return results


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize search engine
    search_engine = SearchEngine()
    
    # Example: Perform a comprehensive search
    criteria = SearchCriteria(
        query="machine learning",
        content_types=["text", "url"],
        date_from=datetime.now() - timedelta(days=30),
        fuzzy_threshold=85,
        highlight=True,
        limit=50
    )
    
    try:
        results, metadata = search_engine.search(criteria)
        
        print(f"Found {len(results)} results in {metadata['execution_time_ms']:.2f}ms")
        
        # Display first few results
        for result in results[:5]:
            print(f"\n--- Result {result.id} ---")
            print(f"Title: {result.title}")
            print(f"Relevance: {result.relevance_score:.3f}")
            if result.context_snippets:
                print(f"Context: {result.context_snippets[0][:100]}...")
        
        # Export results
        search_engine.export_results(results, "search_results.csv", "csv")
        print(f"\nResults exported to search_results.csv")
        
    except SearchError as e:
        print(f"Search error: {e}")
    
    finally:
        search_engine.cleanup()
