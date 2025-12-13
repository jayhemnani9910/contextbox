"""
Semantic Search for ContextBox

This module provides semantic similarity search using sentence-transformers
for advanced contextual information retrieval.
"""

import logging
import json
import sqlite3
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import hashlib
import pickle

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available - using basic search")


class SemanticSearchError(Exception):
    """Custom exception for semantic search operations."""
    pass


class EmbeddingManager:
    """
    Manages text embeddings for semantic search.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./embeddings_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model if available
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                self.model = None
        else:
            self.logger.warning("Sentence transformers not available, using basic embeddings")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not self.model:
            # Fallback to simple TF-IDF-like embedding
            return self._fallback_embedding(text)
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Generate simple fallback embedding."""
        # Simple hash-based embedding as fallback
        words = text.lower().split()
        embedding = np.zeros(384)  # Default embedding size
        
        for word in words:
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(0, min(384, len(str(word_hash)) // 2), 2):
                if i < len(embedding):
                    embedding[i] = (word_hash >> (i // 2)) % 1000 / 1000.0
        
        return embedding.tolist()
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            return [self._fallback_embedding(text) for text in texts]
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def batch_calculate_similarities(self, query_embedding: List[float], 
                                   embeddings: List[List[float]]) -> List[float]:
        """Calculate similarities between query and multiple embeddings."""
        try:
            if not embeddings:
                return []
            
            query_vec = np.array(query_embedding).reshape(1, -1)
            embeddings_matrix = np.array(embeddings)
            similarities = cosine_similarity(query_vec, embeddings_matrix)[0]
            return similarities.tolist()
        except Exception as e:
            self.logger.error(f"Batch similarity calculation failed: {e}")
            return [0.0] * len(embeddings)


class EmbeddingCache:
    """
    Cache for text embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_dir: str = './embeddings_cache'):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / 'embedding_index.json'
        self.logger = logging.getLogger(__name__)
        
        # Load existing index
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load embedding index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load embedding index: {e}")
        
        return {
            'texts': {},
            'embeddings': {},
            'metadata': {},
            'version': '1.0'
        }
    
    def _save_index(self) -> None:
        """Save embedding index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save embedding index: {e}")
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding_file(self, text_hash: str) -> Path:
        """Get file path for embedding storage."""
        return self.cache_dir / f"{text_hash}.embedding"
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        text_hash = self.get_text_hash(text)
        
        # Check if we have this text cached
        if text_hash not in self.index['embeddings']:
            return None
        
        # Check if embedding file exists
        embedding_file = self.get_embedding_file(text_hash)
        if not embedding_file.exists():
            return None
        
        try:
            with open(embedding_file, 'rb') as f:
                embedding = pickle.load(f)
            
            # Update access metadata
            self.index['metadata'][text_hash]['last_accessed'] = datetime.now().isoformat()
            self.index['metadata'][text_hash]['access_count'] += 1
            
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to load cached embedding: {e}")
            return None
    
    def cache_embedding(self, text: str, embedding: List[float]) -> str:
        """Cache embedding for text."""
        text_hash = self.get_text_hash(text)
        
        try:
            # Save embedding to file
            embedding_file = self.get_embedding_file(text_hash)
            with open(embedding_file, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update index
            self.index['embeddings'][text_hash] = {
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'cached_at': datetime.now().isoformat(),
                'file_path': str(embedding_file)
            }
            
            self.index['metadata'][text_hash] = {
                'last_accessed': datetime.now().isoformat(),
                'access_count': 1
            }
            
            # Save index
            self._save_index()
            
            return text_hash
        except Exception as e:
            self.logger.error(f"Failed to cache embedding: {e}")
            raise SemanticSearchError(f"Embedding caching failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_embeddings': len(self.index['embeddings']),
            'total_size_mb': self._calculate_cache_size(),
            'index_size': len(self.index),
            'last_updated': self._get_last_updated()
        }
    
    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for embedding_hash in self.index['embeddings']:
            embedding_file = self.get_embedding_file(embedding_hash)
            if embedding_file.exists():
                total_size += embedding_file.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def _get_last_updated(self) -> Optional[str]:
        """Get last updated timestamp."""
        if self.index['metadata']:
            last_updated = max(
                metadata.get('last_accessed', '') 
                for metadata in self.index['metadata'].values()
            )
            return last_updated if last_updated else None
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        try:
            # Remove all embedding files
            for embedding_file in self.cache_dir.glob("*.embedding"):
                embedding_file.unlink()
            
            # Reset index
            self.index = {
                'texts': {},
                'embeddings': {},
                'metadata': {},
                'version': '1.0'
            }
            
            self._save_index()
            self.logger.info("Embedding cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")


class SemanticSearchIndex:
    """
    Semantic search index for ContextBox data.
    """
    
    def __init__(self, db_path: str = 'contextbox.db', cache_dir: Optional[str] = None):
        """
        Initialize semantic search index.
        
        Args:
            db_path: Path to ContextBox database
            cache_dir: Directory for embedding cache
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(cache_dir=cache_dir)
        self.embedding_cache = EmbeddingCache(cache_dir=cache_dir)
        
        # Search configuration
        self.similarity_threshold = 0.6
        self.max_results = 50
        
        self.logger.info("Semantic search index initialized")
    
    def initialize_database_schema(self, connection: sqlite3.Connection) -> None:
        """
        Initialize database schema for semantic search.
        
        Args:
            connection: SQLite connection
        """
        try:
            # Create embeddings table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS semantic_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artifact_id INTEGER NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    content_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_used DATETIME,
                    usage_count INTEGER DEFAULT 0,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for performance
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_content_hash 
                ON semantic_embeddings(content_hash)
            """)
            
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_artifact_id 
                ON semantic_embeddings(artifact_id)
            """)
            
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_usage_count 
                ON semantic_embeddings(usage_count)
            """)
            
            self.logger.info("Semantic search database schema initialized")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database schema initialization failed: {e}")
            raise SemanticSearchError(f"Database schema error: {e}")
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def create_embedding_for_artifact(self, artifact_data: Dict[str, Any]) -> Optional[str]:
        """
        Create and store embedding for artifact.
        
        Args:
            artifact_data: Artifact data dictionary
            
        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
            # Extract text content from artifact
            content_text = self._extract_artifact_text(artifact_data)
            if not content_text or len(content_text.strip()) < 3:
                return None
            
            # Check if embedding already exists
            content_hash = self.generate_content_hash(content_text)
            existing_embedding = self.embedding_cache.get_cached_embedding(content_text)
            
            if existing_embedding is not None:
                # Use cached embedding
                embedding_data = existing_embedding
            else:
                # Generate new embedding
                embedding_data = self.embedding_manager.generate_embedding(content_text)
                if embedding_data:
                    # Cache the embedding
                    self.embedding_cache.cache_embedding(content_text, embedding_data)
            
            if not embedding_data:
                return None
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO semantic_embeddings 
                    (artifact_id, content_hash, content_text, embedding, last_used, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    artifact_data['id'],
                    content_hash,
                    content_text,
                    pickle.dumps(embedding_data),
                    datetime.now().isoformat(),
                    1 if existing_embedding is None else 0
                ))
                
                embedding_id = cursor.lastrowid
                conn.commit()
            
            self.logger.debug(f"Created embedding for artifact {artifact_data['id']}")
            return str(embedding_id)
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding for artifact: {e}")
            return None
    
    def _extract_artifact_text(self, artifact_data: Dict[str, Any]) -> str:
        """Extract text content from artifact data."""
        text_parts = []
        
        # Add title if available
        if artifact_data.get('title'):
            text_parts.append(artifact_data['title'])
        
        # Add text content
        if artifact_data.get('text'):
            text_parts.append(artifact_data['text'])
        
        # Add URL as text
        if artifact_data.get('url'):
            text_parts.append(artifact_data['url'])
        
        # Add metadata as text
        if artifact_data.get('metadata'):
            metadata_text = json.dumps(artifact_data['metadata'])
            text_parts.append(metadata_text)
        
        return ' '.join(text_parts)
    
    def search_similar_content(self, 
                             query: str, 
                             similarity_threshold: Optional[float] = None,
                             max_results: Optional[int] = None,
                             artifact_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content using semantic similarity.
        
        Args:
            query: Search query text
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results
            artifact_types: Optional list of artifact types to filter by
            
        Returns:
            List of similar artifacts with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Get all embeddings from cache
            cached_embeddings = self._get_all_cached_embeddings()
            if not cached_embeddings:
                return []
            
            # Calculate similarities
            similarities = self.embedding_manager.batch_calculate_similarities(
                query_embedding, cached_embeddings['embeddings']
            )
            
            # Build results with similarity scores
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= (similarity_threshold or self.similarity_threshold):
                    embedding_info = cached_embeddings['info'][i]
                    
                    result = {
                        'artifact_id': embedding_info['artifact_id'],
                        'content_text': embedding_info['content_text'],
                        'similarity_score': similarity,
                        'artifact_type': embedding_info['artifact_type'],
                        'content_hash': embedding_info['content_hash']
                    }
                    
                    # Add additional artifact info if available
                    if 'artifact_data' in embedding_info:
                        result['artifact_data'] = embedding_info['artifact_data']
                    
                    results.append(result)
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            limit = max_results or self.max_results
            
            # Update usage statistics
            self._update_usage_statistics([r['content_hash'] for r in results[:limit]])
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _get_all_cached_embeddings(self) -> Dict[str, Any]:
        """Get all cached embeddings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT se.artifact_id, se.content_text, se.embedding, se.content_hash,
                           a.kind as artifact_type, a.title, a.url, a.metadata
                    FROM semantic_embeddings se
                    LEFT JOIN artifacts a ON se.artifact_id = a.id
                    ORDER BY se.last_used DESC
                """)
                
                embeddings = []
                info = []
                
                for row in cursor.fetchall():
                    try:
                        embedding = pickle.loads(row[2])  # embedding blob
                        embeddings.append(embedding)
                        
                        info.append({
                            'artifact_id': row[0],
                            'content_text': row[1],
                            'content_hash': row[3],
                            'artifact_type': row[4],
                            'artifact_data': {
                                'title': row[5],
                                'url': row[6],
                                'metadata': json.loads(row[7]) if row[7] else None
                            }
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to process embedding row: {e}")
                        continue
                
                return {'embeddings': embeddings, 'info': info}
                
        except Exception as e:
            self.logger.error(f"Failed to get cached embeddings: {e}")
            return {'embeddings': [], 'info': []}
    
    def _update_usage_statistics(self, content_hashes: List[str]) -> None:
        """Update usage statistics for embeddings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                current_time = datetime.now().isoformat()
                
                for content_hash in content_hashes:
                    conn.execute("""
                        UPDATE semantic_embeddings 
                        SET last_used = ?, usage_count = usage_count + 1
                        WHERE content_hash = ?
                    """, (current_time, content_hash))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update usage statistics: {e}")
    
    def rebuild_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Rebuild embeddings for all artifacts.
        
        Args:
            batch_size: Number of artifacts to process at once
            
        Returns:
            Rebuild statistics
        """
        stats = {
            'total_artifacts': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all artifacts
                cursor = conn.execute("""
                    SELECT * FROM artifacts 
                    WHERE text IS NOT NULL AND LENGTH(TRIM(text)) > 0
                """)
                
                artifacts = cursor.fetchall()
                stats['total_artifacts'] = len(artifacts)
                
                column_names = [description[0] for description in cursor.description]
                
                for i in range(0, len(artifacts), batch_size):
                    batch = artifacts[i:i + batch_size]
                    
                    for artifact_row in batch:
                        try:
                            # Convert row to dictionary
                            artifact_data = dict(zip(column_names, artifact_row))
                            artifact_data['metadata'] = json.loads(artifact_data['metadata_json']) if artifact_data.get('metadata_json') else None
                            
                            # Create embedding
                            embedding_id = self.create_embedding_for_artifact(artifact_data)
                            
                            if embedding_id:
                                stats['successful'] += 1
                            else:
                                stats['skipped'] += 1
                            
                            stats['processed'] += 1
                            
                        except Exception as e:
                            stats['failed'] += 1
                            stats['errors'].append(f"Artifact {artifact_data.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Embedding rebuild completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Embedding rebuild failed: {e}")
            stats['errors'].append(f"Rebuild failed: {e}")
            return stats
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get semantic search statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get embedding statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT artifact_id) as unique_artifacts,
                        SUM(usage_count) as total_usage,
                        AVG(usage_count) as avg_usage
                    FROM semantic_embeddings
                """)
                
                row = cursor.fetchone()
                embedding_stats = dict(row) if row else {}
                
                # Get cache statistics
                cache_stats = self.embedding_cache.get_cache_stats()
                
                return {
                    'embedding_stats': embedding_stats,
                    'cache_stats': cache_stats,
                    'model_available': self.embedding_manager.model is not None,
                    'similarity_threshold': self.similarity_threshold,
                    'max_results': self.max_results
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            return {}


class SemanticSearchEngine:
    """
    Main semantic search engine that combines all components.
    """
    
    def __init__(self, db_path: str = 'contextbox.db', config: Dict[str, Any] = None):
        """
        Initialize semantic search engine.
        
        Args:
            db_path: Path to ContextBox database
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize search index
        self.search_index = SemanticSearchIndex(db_path, self.config.get('cache_dir'))
        
        # Configuration
        self.auto_indexing = self.config.get('auto_indexing', True)
        self.default_similarity_threshold = self.config.get('similarity_threshold', 0.6)
        
        # Initialize database schema
        self._initialize_database()
        
        self.logger.info("Semantic search engine initialized")
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.search_index.db_path) as conn:
                self.search_index.initialize_database_schema(conn)
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise SemanticSearchError(f"Database initialization failed: {e}")
    
    def search(self, 
               query: str,
               similarity_threshold: Optional[float] = None,
               max_results: Optional[int] = None,
               artifact_types: Optional[List[str]] = None,
               include_captures: bool = True) -> Dict[str, Any]:
        """
        Perform semantic search with rich metadata.
        
        Args:
            query: Search query
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            artifact_types: Filter by artifact types
            include_captures: Whether to include full capture data
            
        Returns:
            Search results with metadata
        """
        try:
            # Perform semantic search
            results = self.search_index.search_similar_content(
                query=query,
                similarity_threshold=similarity_threshold or self.default_similarity_threshold,
                max_results=max_results,
                artifact_types=artifact_types
            )
            
            # Enrich results with additional data
            enriched_results = []
            for result in results:
                enriched_result = self._enrich_search_result(result, include_captures)
                enriched_results.append(enriched_result)
            
            # Prepare response
            search_stats = self.search_index.get_search_statistics()
            
            return {
                'query': query,
                'results': enriched_results,
                'result_count': len(enriched_results),
                'search_time': datetime.now().isoformat(),
                'search_metadata': {
                    'similarity_threshold': similarity_threshold or self.default_similarity_threshold,
                    'max_results': max_results or self.search_index.max_results,
                    'artifact_types': artifact_types,
                    'include_captures': include_captures
                },
                'statistics': search_stats
            }
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e),
                'result_count': 0,
                'search_time': datetime.now().isoformat()
            }
    
    def _enrich_search_result(self, result: Dict[str, Any], include_captures: bool) -> Dict[str, Any]:
        """Enrich search result with additional metadata."""
        enriched = result.copy()
        
        try:
            with sqlite3.connect(self.search_index.db_path) as conn:
                # Get full artifact data
                cursor = conn.execute("""
                    SELECT a.*, c.created_at as capture_date, c.source_window
                    FROM artifacts a
                    LEFT JOIN captures c ON a.capture_id = c.id
                    WHERE a.id = ?
                """, (result['artifact_id'],))
                
                row = cursor.fetchone()
                if row:
                    column_names = [description[0] for description in cursor.description]
                    artifact_data = dict(zip(column_names, row))
                    
                    # Parse metadata
                    if artifact_data.get('metadata_json'):
                        artifact_data['metadata'] = json.loads(artifact_data['metadata_json'])
                        del artifact_data['metadata_json']
                    
                    enriched['full_artifact'] = artifact_data
                    
                    # Add capture context
                    if include_captures and artifact_data.get('capture_id'):
                        capture_cursor = conn.execute("""
                            SELECT * FROM captures WHERE id = ?
                        """, (artifact_data['capture_id'],))
                        
                        capture_row = capture_cursor.fetchone()
                        if capture_row:
                            capture_columns = [desc[0] for desc in capture_cursor.description]
                            capture_data = dict(zip(capture_columns, capture_row))
                            enriched['capture_context'] = capture_data
        
        except Exception as e:
            self.logger.warning(f"Failed to enrich search result: {e}")
        
        return enriched
    
    def index_artifact(self, artifact_data: Dict[str, Any]) -> bool:
        """Index a single artifact."""
        try:
            embedding_id = self.search_index.create_embedding_for_artifact(artifact_data)
            return embedding_id is not None
        except Exception as e:
            self.logger.error(f"Failed to index artifact: {e}")
            return False
    
    def rebuild_index(self, batch_size: int = 100) -> Dict[str, Any]:
        """Rebuild the entire semantic search index."""
        return self.search_index.rebuild_embeddings(batch_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return self.search_index.get_search_statistics()
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update search engine configuration."""
        self.config.update(config_updates)
        
        # Update search index configuration
        if 'similarity_threshold' in config_updates:
            self.search_index.similarity_threshold = config_updates['similarity_threshold']
        
        if 'max_results' in config_updates:
            self.search_index.max_results = config_updates['max_results']
        
        self.logger.info("Semantic search configuration updated")


def create_semantic_search_engine(db_path: str = 'contextbox.db', 
                                 config: Optional[Dict[str, Any]] = None) -> SemanticSearchEngine:
    """
    Factory function to create semantic search engine.
    
    Args:
        db_path: Path to ContextBox database
        config: Configuration dictionary
        
    Returns:
        SemanticSearchEngine instance
    """
    return SemanticSearchEngine(db_path, config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create semantic search engine
    config = {
        'cache_dir': './semantic_search_cache',
        'auto_indexing': True,
        'similarity_threshold': 0.6,
        'max_results': 20
    }
    
    search_engine = create_semantic_search_engine('contextbox.db', config)
    
    # Test search
    query = "email contact information"
    results = search_engine.search(query, max_results=10)
    
    print(f"Search results for '{query}':")
    print(f"Found {results['result_count']} results")
    
    for i, result in enumerate(results['results'][:5]):
        print(f"{i+1}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Content: {result['content_text'][:100]}...")
        print()
    
    # Get statistics
    stats = search_engine.get_statistics()
    print("Search statistics:")
    print(json.dumps(stats, indent=2))