"""
Performance Optimizations for ContextBox

This module provides advanced performance optimizations including:
- Database indexing strategies
- Query optimization
- Caching systems
- Connection pooling
- Batch operations
- Performance monitoring
"""

import sqlite3
import logging
import json
import time
import threading
import weakref
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import hashlib
import pickle
import psutil
import os

try:
    import sqlite3
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class PerformanceOptimizationError(Exception):
    """Custom exception for performance optimization operations."""
    pass


class DatabaseIndexing:
    """
    Advanced database indexing for ContextBox.
    """
    
    def __init__(self, db_path: str):
        """Initialize database indexing."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Index definitions
        self.core_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_captures_created_at ON captures(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_capture_id ON artifacts(capture_id)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_url ON artifacts(url)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_text_fts ON artifacts(text)",
        ]
        
        self.advanced_indexes = [
            # Composite indexes
            "CREATE INDEX IF NOT EXISTS idx_artifacts_capture_kind ON artifacts(capture_id, kind)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_kind_text ON artifacts(kind, text)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_url_kind ON artifacts(url, kind)",
            
            # Partial indexes for better performance
            "CREATE INDEX IF NOT EXISTS idx_artifacts_url_not_null ON artifacts(url) WHERE url IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_text_not_null ON artifacts(text) WHERE text IS NOT NULL",
            
            # Covering indexes
            "CREATE INDEX IF NOT EXISTS idx_artifacts_covering ON artifacts(capture_id, kind, id, title, url) WHERE text IS NULL",
        ]
        
        self.semantic_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_semantic_embedding_artifact ON semantic_embeddings(artifact_id, last_used)",
            "CREATE INDEX IF NOT EXISTS idx_semantic_usage ON semantic_embeddings(usage_count, last_used)",
        ]
    
    def create_all_indexes(self) -> Dict[str, Any]:
        """Create all database indexes."""
        results = {
            'core_indexes': {'created': 0, 'errors': []},
            'advanced_indexes': {'created': 0, 'errors': []},
            'semantic_indexes': {'created': 0, 'errors': []},
            'total_created': 0,
            'total_errors': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable FTS for full-text search
                self._setup_full_text_search(conn)
                
                # Create core indexes
                for index_sql in self.core_indexes:
                    try:
                        conn.execute(index_sql)
                        results['core_indexes']['created'] += 1
                        results['total_created'] += 1
                    except sqlite3.Error as e:
                        error_msg = f"Core index creation failed: {e}"
                        results['core_indexes']['errors'].append(error_msg)
                        results['total_errors'] += 1
                
                # Create advanced indexes
                for index_sql in self.advanced_indexes:
                    try:
                        conn.execute(index_sql)
                        results['advanced_indexes']['created'] += 1
                        results['total_created'] += 1
                    except sqlite3.Error as e:
                        error_msg = f"Advanced index creation failed: {e}"
                        results['advanced_indexes']['errors'].append(error_msg)
                        results['total_errors'] += 1
                
                # Create semantic indexes
                for index_sql in self.semantic_indexes:
                    try:
                        conn.execute(index_sql)
                        results['semantic_indexes']['created'] += 1
                        results['total_created'] += 1
                    except sqlite3.Error as e:
                        error_msg = f"Semantic index creation failed: {e}"
                        results['semantic_indexes']['errors'].append(error_msg)
                        results['total_errors'] += 1
                
                # Analyze database for query optimization
                conn.execute("ANALYZE")
                
                conn.commit()
                self.logger.info(f"Database indexing completed: {results['total_created']} indexes created")
                
        except sqlite3.Error as e:
            error_msg = f"Database indexing failed: {e}"
            results['total_errors'] += 1
            results['core_indexes']['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results
    
    def _setup_full_text_search(self, conn: sqlite3.Connection) -> None:
        """Setup full-text search capabilities."""
        try:
            # Create FTS virtual table for artifacts text
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts 
                USING fts5(title, text, url, content='')
            """)
            
            # Create trigger to keep FTS index updated
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS artifacts_fts_insert 
                AFTER INSERT ON artifacts BEGIN
                    INSERT INTO artifacts_fts(rowid, title, text, url)
                    VALUES (new.id, new.title, new.text, new.url);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS artifacts_fts_delete 
                AFTER DELETE ON artifacts BEGIN
                    INSERT INTO artifacts_fts(artifacts_fts, rowid, title, text, url)
                    VALUES('delete', old.id, old.title, old.text, old.url);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS artifacts_fts_update 
                AFTER UPDATE ON artifacts BEGIN
                    INSERT INTO artifacts_fts(artifacts_fts, rowid, title, text, url)
                    VALUES('delete', old.id, old.title, old.text, old.url);
                    INSERT INTO artifacts_fts(rowid, title, text, url)
                    VALUES (new.id, new.title, new.text, new.url);
                END
            """)
            
            self.logger.info("Full-text search setup completed")
            
        except sqlite3.Error as e:
            self.logger.warning(f"Full-text search setup failed: {e}")
    
    def optimize_database_settings(self) -> None:
        """Apply database performance optimizations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode = WAL")
                
                # Set synchronous mode for performance
                conn.execute("PRAGMA synchronous = NORMAL")
                
                # Set cache size (in KB, negative for KB * 1024)
                conn.execute("PRAGMA cache_size = -20000")  # 20MB cache
                
                # Set temp store to memory
                conn.execute("PRAGMA temp_store = MEMORY")
                
                # Set memory map size (in bytes)
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
                
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Set busy timeout
                conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
                
                conn.commit()
                self.logger.info("Database performance settings optimized")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database optimization failed: {e}")
            raise PerformanceOptimizationError(f"Database optimization failed: {e}")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get database index statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get index list
                cursor = conn.execute("PRAGMA index_list(captures)")
                capture_indexes = [dict(row) for row in cursor.fetchall()]
                
                cursor = conn.execute("PRAGMA index_list(artifacts)")
                artifact_indexes = [dict(row) for row in cursor.fetchall()]
                
                # Get index details
                index_details = []
                for index_list in [capture_indexes, artifact_indexes]:
                    for index in index_list:
                        cursor = conn.execute(f"PRAGMA index_info({index['name']})")
                        index_info = cursor.fetchall()
                        index['columns'] = [col[2] for col in index_info]
                        index_details.append(index)
                
                return {
                    'capture_indexes': capture_indexes,
                    'artifact_indexes': artifact_indexes,
                    'index_details': index_details,
                    'total_indexes': len(index_details)
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get index statistics: {e}")
            return {}
    
    def rebuild_indexes(self) -> Dict[str, Any]:
        """Rebuild all indexes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Drop all indexes
                conn.execute("DROP INDEX IF EXISTS idx_captures_created_at")
                conn.execute("DROP INDEX IF NOT EXISTS idx_artifacts_capture_id")
                conn.execute("DROP INDEX IF NOT EXISTS idx_artifacts_kind")
                conn.execute("DROP INDEX IF NOT EXISTS idx_artifacts_url")
                
                # Recreate indexes
                return self.create_all_indexes()
                
        except sqlite3.Error as e:
            self.logger.error(f"Index rebuild failed: {e}")
            raise PerformanceOptimizationError(f"Index rebuild failed: {e}")


class QueryOptimizer:
    """
    Query optimization and performance monitoring.
    """
    
    def __init__(self, db_path: str):
        """Initialize query optimizer."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Query execution statistics
        self.query_stats = {}
        self.slow_queries = []
        self.performance_lock = threading.Lock()
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.max_query_stats = 1000
    
    def optimize_query(self, query: str, params: Optional[tuple] = None) -> str:
        """
        Optimize a SQL query using EXPLAIN QUERY PLAN.
        
        Args:
            query: SQL query to optimize
            params: Query parameters
            
        Returns:
            Optimized query and analysis
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get query plan
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                cursor = conn.execute(explain_query, params or ())
                plan = cursor.fetchall()
                
                # Analyze query plan
                analysis = self._analyze_query_plan(plan)
                
                # Suggest optimizations
                suggestions = self._suggest_optimizations(analysis, query)
                
                return {
                    'original_query': query,
                    'query_plan': plan,
                    'analysis': analysis,
                    'suggestions': suggestions,
                    'optimized': len(suggestions) > 0
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Query optimization failed: {e}")
            return {
                'original_query': query,
                'error': str(e),
                'optimized': False
            }
    
    def _analyze_query_plan(self, plan: List[Tuple]) -> Dict[str, Any]:
        """Analyze query execution plan."""
        analysis = {
            'uses_index': False,
            'full_table_scan': False,
            'order_by_costly': False,
            'complexity_score': 0,
            'scan_types': []
        }
        
        for step in plan:
            detail = step[3] if len(step) > 3 else ""
            
            if "USING INDEX" in detail:
                analysis['uses_index'] = True
                analysis['scan_types'].append('index_scan')
            elif "SCAN TABLE" in detail:
                analysis['full_table_scan'] = True
                analysis['scan_types'].append('table_scan')
            elif "ORDER BY" in detail:
                analysis['order_by_costly'] = True
            
            # Calculate complexity score
            if "SCAN TABLE" in detail:
                analysis['complexity_score'] += 10
            elif "USING INDEX" in detail:
                analysis['complexity_score'] += 2
            if "ORDER BY" in detail:
                analysis['complexity_score'] += 5
        
        return analysis
    
    def _suggest_optimizations(self, analysis: Dict[str, Any], query: str) -> List[str]:
        """Suggest query optimizations based on analysis."""
        suggestions = []
        
        if analysis['full_table_scan']:
            suggestions.append("Consider adding an index to avoid full table scan")
        
        if analysis['order_by_costly']:
            suggestions.append("ORDER BY clause may be costly - consider indexing order by columns")
        
        if analysis['complexity_score'] > 15:
            suggestions.append("Query is complex - consider breaking into smaller queries or using temporary tables")
        
        if "JOIN" in query.upper() and not analysis['uses_index']:
            suggestions.append("JOIN operations without indexes detected - add appropriate foreign key indexes")
        
        if "LIKE '%" in query.upper():
            suggestions.append("Wildcard LIKE patterns detected - consider using FTS or full-text indexes")
        
        return suggestions
    
    def execute_measured_query(self, query: str, params: Optional[tuple] = None, 
                              query_name: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute query with performance measurement.
        
        Args:
            query: SQL query
            params: Query parameters
            query_name: Name for tracking
            
        Returns:
            Tuple of (query_result, performance_metrics)
        """
        start_time = time.time()
        query_id = query_name or hashlib.md5(query.encode()).hexdigest()[:8]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params or ())
                results = cursor.fetchall()
                
                execution_time = time.time() - start_time
                
                # Record performance metrics
                metrics = {
                    'query_id': query_id,
                    'execution_time': execution_time,
                    'row_count': len(results),
                    'timestamp': datetime.now().isoformat(),
                    'slow_query': execution_time > self.slow_query_threshold
                }
                
                self._record_query_performance(query, metrics)
                
                return [dict(row) for row in results], metrics
                
        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            error_metrics = {
                'query_id': query_id,
                'execution_time': execution_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'slow_query': execution_time > self.slow_query_threshold
            }
            
            self._record_query_performance(query, error_metrics)
            raise
    
    def _record_query_performance(self, query: str, metrics: Dict[str, Any]) -> None:
        """Record query performance metrics."""
        with self.performance_lock:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    'query': query[:200] + "..." if len(query) > 200 else query,
                    'executions': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'errors': 0,
                    'last_execution': None
                }
            
            stats = self.query_stats[query_hash]
            stats['executions'] += 1
            stats['total_time'] += metrics['execution_time']
            stats['avg_time'] = stats['total_time'] / stats['executions']
            stats['min_time'] = min(stats['min_time'], metrics['execution_time'])
            stats['max_time'] = max(stats['max_time'], metrics['execution_time'])
            stats['last_execution'] = metrics['timestamp']
            
            if 'error' in metrics:
                stats['errors'] += 1
            
            # Record slow queries
            if metrics['slow_query']:
                self.slow_queries.append({
                    'query': query[:200] + "..." if len(query) > 200 else query,
                    'execution_time': metrics['execution_time'],
                    'timestamp': metrics['timestamp']
                })
                
                # Keep only last 100 slow queries
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]
            
            # Limit query stats size
            if len(self.query_stats) > self.max_query_stats:
                # Remove oldest entries
                sorted_stats = sorted(self.query_stats.items(), 
                                    key=lambda x: x[1]['last_execution'] or '')
                self.query_stats = dict(sorted_stats[-self.max_query_stats//2:])
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self.performance_lock:
            total_executions = sum(stats['executions'] for stats in self.query_stats.values())
            total_errors = sum(stats['errors'] for stats in self.query_stats.values())
            
            # Calculate overall statistics
            avg_execution_time = 0.0
            if total_executions > 0:
                avg_execution_time = sum(
                    stats['avg_time'] * stats['executions'] 
                    for stats in self.query_stats.values()
                ) / total_executions
            
            # Get top slowest queries
            top_slow_queries = sorted(
                self.query_stats.values(), 
                key=lambda x: x['avg_time'], 
                reverse=True
            )[:10]
            
            # Get most frequently executed queries
            top_frequent_queries = sorted(
                self.query_stats.values(),
                key=lambda x: x['executions'],
                reverse=True
            )[:10]
            
            return {
                'summary': {
                    'total_queries': len(self.query_stats),
                    'total_executions': total_executions,
                    'total_errors': total_errors,
                    'error_rate': total_errors / total_executions if total_executions > 0 else 0,
                    'avg_execution_time': avg_execution_time,
                    'slow_queries_count': len(self.slow_queries)
                },
                'top_slow_queries': top_slow_queries,
                'top_frequent_queries': top_frequent_queries,
                'recent_slow_queries': self.slow_queries[-10:],
                'slow_query_threshold': self.slow_query_threshold
            }
    
    def clear_performance_stats(self) -> None:
        """Clear performance statistics."""
        with self.performance_lock:
            self.query_stats.clear()
            self.slow_queries.clear()
            self.logger.info("Performance statistics cleared")


class DatabaseCache:
    """
    Database result caching system.
    """
    
    def __init__(self, cache_dir: str = './db_cache', max_size_mb: int = 100):
        """Initialize database cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'entries': 0,
            'total_size': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _generate_cache_key(self, query: str, params: Optional[tuple] = None) -> str:
        """Generate cache key for query."""
        key_data = f"{query}_{params or ()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        """
        Get cached query result.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Cached result or None
        """
        cache_key = self._generate_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        if not cache_file.exists():
            with self.lock:
                self.cache_stats['misses'] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            if self._is_cache_valid(cached_data):
                with self.lock:
                    self.cache_stats['hits'] += 1
                return cached_data['result']
            else:
                # Remove expired cache
                cache_file.unlink()
                with self.lock:
                    self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load cached result: {e}")
            return None
    
    def cache_result(self, query: str, result: Any, params: Optional[tuple] = None, 
                    ttl_seconds: int = 3600) -> None:
        """
        Cache query result.
        
        Args:
            query: SQL query
            result: Query result to cache
            params: Query parameters
            ttl_seconds: Time to live in seconds
        """
        cache_key = self._generate_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            cached_data = {
                'query': query,
                'params': params,
                'result': result,
                'created_at': datetime.now().isoformat(),
                'ttl_seconds': ttl_seconds
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Update cache size tracking
            with self.lock:
                self.cache_stats['entries'] += 1
                self.cache_stats['total_size'] += cache_file.stat().st_size
            
            # Clean up cache if too large
            self._cleanup_cache()
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        try:
            created_at = datetime.fromisoformat(cached_data['created_at'])
            ttl = cached_data.get('ttl_seconds', 3600)
            return datetime.now() - created_at < timedelta(seconds=ttl)
        except Exception:
            return False
    
    def _cleanup_cache(self) -> None:
        """Clean up cache to stay within size limits."""
        try:
            current_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            
            if current_size <= self.max_size_bytes:
                return
            
            # Get files sorted by access time (LRU)
            cache_files = []
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    stat = cache_file.stat()
                    cache_files.append((cache_file, stat.st_atime))
                except OSError:
                    continue
            
            cache_files.sort(key=lambda x: x[1])  # Sort by access time
            
            # Remove oldest files until under limit
            for cache_file, _ in cache_files:
                if current_size <= self.max_size_bytes:
                    break
                
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    current_size -= file_size
                    with self.lock:
                        self.cache_stats['entries'] -= 1
                        self.cache_stats['total_size'] -= file_size
                except OSError:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            with self.lock:
                self.cache_stats = {
                    'hits': 0,
                    'misses': 0,
                    'entries': 0,
                    'total_size': 0
                }
            
            self.logger.info("Database cache cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = 0
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_requests > 0:
                hit_rate = self.cache_stats['hits'] / total_requests
            
            return {
                **self.cache_stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'current_size_mb': self.cache_stats['total_size'] / (1024 * 1024)
            }


class PerformanceMonitor:
    """
    System performance monitoring for ContextBox.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.performance_data = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start system performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                metrics['timestamp'] = datetime.now().isoformat()
                
                self.performance_data.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.performance_data) > 1000:
                    self.performance_data = self.performance_data[-1000:]
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database size
            db_size = 0
            if Path('contextbox.db').exists():
                db_size = Path('contextbox.db').stat().st_size
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'database_size_mb': db_size / (1024**2),
                'process_count': len(psutil.pids())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.performance_data:
            return {'message': 'No performance data available'}
        
        recent_data = self.performance_data[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_cpu = sum(d.get('cpu_percent', 0) for d in recent_data) / len(recent_data)
        avg_memory = sum(d.get('memory_percent', 0) for d in recent_data) / len(recent_data)
        avg_disk = sum(d.get('disk_percent', 0) for d in recent_data) / len(recent_data)
        
        # Get latest values
        latest = self.performance_data[-1] if self.performance_data else {}
        
        return {
            'current_metrics': latest,
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'disk_percent': avg_disk
            },
            'data_points': len(self.performance_data),
            'monitoring_duration': self._get_monitoring_duration()
        }
    
    def _get_monitoring_duration(self) -> Optional[str]:
        """Get monitoring duration."""
        if len(self.performance_data) < 2:
            return None
        
        start_time = datetime.fromisoformat(self.performance_data[0]['timestamp'])
        end_time = datetime.fromisoformat(self.performance_data[-1]['timestamp'])
        duration = end_time - start_time
        
        return str(duration)


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logging.info(f"{function_name} executed in {execution_time:.3f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{function_name} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper


class ContextBoxPerformanceManager:
    """
    Main performance management system for ContextBox.
    """
    
    def __init__(self, db_path: str = 'contextbox.db', config: Dict[str, Any] = None):
        """Initialize performance manager."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_path = db_path
        self.indexing = DatabaseIndexing(db_path)
        self.query_optimizer = QueryOptimizer(db_path)
        
        cache_dir = self.config.get('cache_dir', './performance_cache')
        cache_size_mb = self.config.get('cache_size_mb', 100)
        self.cache = DatabaseCache(cache_dir, cache_size_mb)
        
        self.monitor = PerformanceMonitor()
        
        # Apply optimizations
        self._apply_initial_optimizations()
        
        self.logger.info("Performance manager initialized")
    
    def _apply_initial_optimizations(self) -> None:
        """Apply initial performance optimizations."""
        try:
            # Optimize database settings
            self.indexing.optimize_database_settings()
            
            # Create indexes
            indexing_results = self.indexing.create_all_indexes()
            
            if indexing_results['total_errors'] == 0:
                self.logger.info("Database optimizations applied successfully")
            else:
                self.logger.warning(f"Some optimizations failed: {indexing_results['total_errors']} errors")
            
        except Exception as e:
            self.logger.error(f"Failed to apply initial optimizations: {e}")
    
    def optimize_database(self) -> Dict[str, Any]:
        """Perform comprehensive database optimization."""
        results = {
            'indexing': {},
            'optimization': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Create/rebuild indexes
            results['indexing'] = self.indexing.create_all_indexes()
            
            # Apply database optimizations
            self.indexing.optimize_database_settings()
            
            # Get statistics
            results['statistics'] = self.indexing.get_index_statistics()
            
            self.logger.info("Database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'database_optimization': self.indexing.get_index_statistics(),
            'query_performance': self.query_optimizer.get_performance_report(),
            'cache_statistics': self.cache.get_cache_statistics(),
            'system_performance': self.monitor.get_performance_summary(),
            'generated_at': datetime.now().isoformat()
        }
    
    def clear_performance_data(self) -> None:
        """Clear all performance data."""
        self.query_optimizer.clear_performance_stats()
        self.cache.clear_cache()
        self.logger.info("Performance data cleared")
    
    def start_performance_monitoring(self, interval_seconds: int = 60) -> None:
        """Start performance monitoring."""
        self.monitor.start_monitoring(interval_seconds)
    
    def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitor.stop_monitoring()
    
    def execute_optimized_query(self, query: str, params: Optional[tuple] = None, 
                              use_cache: bool = True, cache_ttl: int = 3600) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute query with optimizations.
        
        Args:
            query: SQL query
            params: Query parameters
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Query results and performance metrics
        """
        # Check cache first
        if use_cache:
            cached_result = self.cache.get_cached_result(query, params)
            if cached_result is not None:
                return cached_result, {'cached': True}
        
        # Execute with performance monitoring
        results, metrics = self.query_optimizer.execute_measured_query(query, params)
        
        # Cache result if successful
        if use_cache and results:
            self.cache.cache_result(query, results, params, cache_ttl)
            metrics['cached'] = False
        
        return results, metrics


def create_performance_manager(db_path: str = 'contextbox.db', 
                              config: Optional[Dict[str, Any]] = None) -> ContextBoxPerformanceManager:
    """
    Factory function to create performance manager.
    
    Args:
        db_path: Path to ContextBox database
        config: Configuration dictionary
        
    Returns:
        ContextBoxPerformanceManager instance
    """
    return ContextBoxPerformanceManager(db_path, config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance manager
    config = {
        'cache_dir': './perf_cache',
        'cache_size_mb': 200
    }
    
    perf_manager = create_performance_manager('contextbox.db', config)
    
    # Optimize database
    print("Optimizing database...")
    optimization_results = perf_manager.optimize_database()
    print(f"Optimization completed: {optimization_results['indexing']['total_created']} indexes created")
    
    # Start monitoring
    print("Starting performance monitoring...")
    perf_manager.start_performance_monitoring(30)
    
    # Get performance report
    time.sleep(2)  # Wait for some monitoring data
    report = perf_manager.get_performance_report()
    
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Stop monitoring
    perf_manager.stop_performance_monitoring()
    print("Performance monitoring stopped")