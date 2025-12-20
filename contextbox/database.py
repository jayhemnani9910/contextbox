"""
ContextBox database module for SQLite operations.

This module provides database connection, schema management, and CRUD operations
for ContextBox captures and artifacts.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class ContextDatabase:
    """
    SQLite database manager for ContextBox.
    
    Handles database connections, schema creation, and CRUD operations
    for captures and artifacts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the database manager.
        
        Args:
            config: Database configuration dictionary
                - db_path: Path to SQLite database file
                - timeout: Connection timeout in seconds
                - backup_interval: Interval for automatic backups (optional)
        """
        self.config = config or {}
        self.db_path = self.config.get('db_path') or self.config.get('path') or 'contextbox.db'
        self.timeout = self.config.get('timeout', 30.0)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"ContextDatabase initialized with database: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Create and return a database connection.
        
        Returns:
            SQLite connection object
            
        Raises:
            DatabaseError: If connection fails
        """
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            
            # Enable foreign keys and row factory
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            
            return conn
            
        except sqlite3.Error as e:
            self.logger.error(f"Database connection failed: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    def _initialize_database(self) -> None:
        """
        Initialize database schema and tables.
        
        Raises:
            DatabaseError: If initialization fails
        """
        try:
            with self._get_connection() as conn:
                # Create captures table
                self._create_captures_table(conn)
                
                # Create artifacts table
                self._create_artifacts_table(conn)
                
                # Create indexes
                self._create_indexes(conn)
                
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode = WAL")
                
                conn.commit()
                self.logger.info("Database schema initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    def _create_captures_table(self, conn: sqlite3.Connection) -> None:
        """Create the captures table."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_window TEXT,
            screenshot_path TEXT,
            clipboard_text TEXT,
            notes TEXT
        )
        """
        conn.execute(create_sql)
    
    def _create_artifacts_table(self, conn: sqlite3.Connection) -> None:
        """Create the artifacts table."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            url TEXT,
            title TEXT,
            text TEXT,
            metadata_json TEXT,
            FOREIGN KEY (capture_id) REFERENCES captures(id) ON DELETE CASCADE
        )
        """
        conn.execute(create_sql)
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for better performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_captures_created_at ON captures(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_capture_id ON artifacts(capture_id)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_url ON artifacts(url)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def create_capture(
        self,
        source_window: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        clipboard_text: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new capture record.
        
        Args:
            source_window: Source window name/title
            screenshot_path: Path to screenshot file
            clipboard_text: Text from clipboard
            notes: Additional notes
            
        Returns:
            ID of the created capture
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO captures (source_window, screenshot_path, clipboard_text, notes)
                    VALUES (?, ?, ?, ?)
                """, (source_window, screenshot_path, clipboard_text, notes))
                
                capture_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Created capture with ID: {capture_id}")
                return capture_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create capture: {e}")
            raise DatabaseError(f"Failed to create capture: {e}")
    
    def get_capture(self, capture_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a capture by ID.
        
        Args:
            capture_id: ID of the capture to retrieve
            
        Returns:
            Capture dictionary or None if not found
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM captures WHERE id = ?
                """, (capture_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve capture {capture_id}: {e}")
            raise DatabaseError(f"Failed to retrieve capture: {e}")
    
    def update_capture(
        self,
        capture_id: int,
        source_window: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        clipboard_text: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update a capture record.
        
        Args:
            capture_id: ID of the capture to update
            source_window: Source window name/title
            screenshot_path: Path to screenshot file
            clipboard_text: Text from clipboard
            notes: Additional notes
            
        Returns:
            True if update was successful, False if capture not found
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            with self._get_connection() as conn:
                # Build dynamic update query
                updates = []
                params = []
                
                if source_window is not None:
                    updates.append("source_window = ?")
                    params.append(source_window)
                if screenshot_path is not None:
                    updates.append("screenshot_path = ?")
                    params.append(screenshot_path)
                if clipboard_text is not None:
                    updates.append("clipboard_text = ?")
                    params.append(clipboard_text)
                if notes is not None:
                    updates.append("notes = ?")
                    params.append(notes)
                
                if not updates:
                    return False
                
                params.append(capture_id)
                
                cursor = conn.execute(f"""
                    UPDATE captures 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    self.logger.info(f"Updated capture with ID: {capture_id}")
                else:
                    self.logger.warning(f"Failed to find capture with ID: {capture_id}")
                
                return success
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update capture {capture_id}: {e}")
            raise DatabaseError(f"Failed to update capture: {e}")
    
    def delete_capture(self, capture_id: int) -> bool:
        """
        Delete a capture and all its artifacts.
        
        Args:
            capture_id: ID of the capture to delete
            
        Returns:
            True if deletion was successful, False if capture not found
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM captures WHERE id = ?
                """, (capture_id,))
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    self.logger.info(f"Deleted capture with ID: {capture_id}")
                else:
                    self.logger.warning(f"Failed to find capture with ID: {capture_id}")
                
                return success
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete capture {capture_id}: {e}")
            raise DatabaseError(f"Failed to delete capture: {e}")
    
    def list_captures(
        self,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        List captures with optional filtering.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            since: Only return captures created after this datetime
            
        Returns:
            List of capture dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM captures"
                params = []
                
                if since:
                    query += " WHERE created_at >= ?"
                    params.append(since.isoformat())
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    raw = dict(row)
                    normalized = {
                        key: (str(value) if key == 'id' and value is not None else value)
                        for key, value in raw.items()
                    }
                    results.append(normalized)
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to list captures: {e}")
            raise DatabaseError(f"Failed to list captures: {e}")
    
    def create_artifact(
        self,
        capture_id: int,
        kind: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create a new artifact record.
        
        Args:
            capture_id: ID of the associated capture
            kind: Type of artifact (e.g., 'url', 'text', 'image')
            url: URL if applicable
            title: Title or description
            text: Text content
            metadata: Additional metadata as dictionary
            
        Returns:
            ID of the created artifact
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (capture_id, kind, url, title, text, metadata_json))
                
                artifact_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Created artifact with ID: {artifact_id}")
                return artifact_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create artifact: {e}")
            raise DatabaseError(f"Failed to create artifact: {e}")
    
    def get_artifact(self, artifact_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an artifact by ID.
        
        Args:
            artifact_id: ID of the artifact to retrieve
            
        Returns:
            Artifact dictionary or None if not found
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM artifacts WHERE id = ?
                """, (artifact_id,))
                
                row = cursor.fetchone()
                if row:
                    artifact = dict(row)
                    
                    # Parse metadata JSON
                    if artifact['metadata_json']:
                        artifact['metadata'] = json.loads(artifact['metadata_json'])
                        del artifact['metadata_json']
                    else:
                        artifact['metadata'] = None
                        del artifact['metadata_json']
                    
                    return artifact
                return None
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            raise DatabaseError(f"Failed to retrieve artifact: {e}")
    
    def get_artifacts_by_capture(self, capture_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all artifacts for a capture.
        
        Args:
            capture_id: ID of the capture
            
        Returns:
            List of artifact dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM artifacts WHERE capture_id = ? ORDER BY id
                """, (capture_id,))
                
                artifacts = []
                for row in cursor.fetchall():
                    artifact = dict(row)
                    
                    # Parse metadata JSON
                    if artifact['metadata_json']:
                        artifact['metadata'] = json.loads(artifact['metadata_json'])
                        del artifact['metadata_json']
                    else:
                        artifact['metadata'] = None
                        del artifact['metadata_json']
                    
                    artifacts.append(artifact)
                
                return artifacts
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to retrieve artifacts for capture {capture_id}: {e}")
            raise DatabaseError(f"Failed to retrieve artifacts: {e}")
    
    def update_artifact(
        self,
        artifact_id: int,
        kind: Optional[str] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an artifact record.
        
        Args:
            artifact_id: ID of the artifact to update
            kind: Type of artifact
            url: URL if applicable
            title: Title or description
            text: Text content
            metadata: Additional metadata as dictionary
            
        Returns:
            True if update was successful, False if artifact not found
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            with self._get_connection() as conn:
                # Build dynamic update query
                updates = []
                params = []
                
                if kind is not None:
                    updates.append("kind = ?")
                    params.append(kind)
                if url is not None:
                    updates.append("url = ?")
                    params.append(url)
                if title is not None:
                    updates.append("title = ?")
                    params.append(title)
                if text is not None:
                    updates.append("text = ?")
                    params.append(text)
                if metadata is not None:
                    updates.append("metadata_json = ?")
                    params.append(json.dumps(metadata))
                
                if not updates:
                    return False
                
                params.append(artifact_id)
                
                cursor = conn.execute(f"""
                    UPDATE artifacts 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    self.logger.info(f"Updated artifact with ID: {artifact_id}")
                else:
                    self.logger.warning(f"Failed to find artifact with ID: {artifact_id}")
                
                return success
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update artifact {artifact_id}: {e}")
            raise DatabaseError(f"Failed to update artifact: {e}")
    
    def delete_artifact(self, artifact_id: int) -> bool:
        """
        Delete an artifact.
        
        Args:
            artifact_id: ID of the artifact to delete
            
        Returns:
            True if deletion was successful, False if artifact not found
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM artifacts WHERE id = ?
                """, (artifact_id,))
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    self.logger.info(f"Deleted artifact with ID: {artifact_id}")
                else:
                    self.logger.warning(f"Failed to find artifact with ID: {artifact_id}")
                
                return success
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            raise DatabaseError(f"Failed to delete artifact: {e}")

    def search_contexts(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search captures and related artifacts for a term.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of captures to return
            
        Returns:
            List of capture dictionaries matching the term
        """
        like_term = f"%{search_term}%"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT c.*, CAST(c.id AS TEXT) AS id_text
                    FROM captures c
                    LEFT JOIN artifacts a ON a.capture_id = c.id
                    WHERE c.source_window LIKE ?
                       OR c.clipboard_text LIKE ?
                       OR c.notes LIKE ?
                       OR a.title LIKE ?
                       OR a.text LIKE ?
                       OR a.url LIKE ?
                    ORDER BY c.created_at DESC
                    LIMIT ?
                    """,
                    (
                        like_term,
                        like_term,
                        like_term,
                        like_term,
                        like_term,
                        like_term,
                        limit,
                    )
                )
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    if 'id_text' in row_dict and row_dict['id_text'] is not None:
                        row_dict['id'] = row_dict['id_text']
                        del row_dict['id_text']
                    results.append(row_dict)
                return results
        except sqlite3.Error as e:
            self.logger.error(f"Failed to search contexts: {e}")
            raise DatabaseError(f"Failed to search contexts: {e}")
    
    def store(self, context: Dict[str, Any]) -> str:
        """
        Store context data in the database (for ContextBox integration).
        
        Args:
            context: Context data dictionary containing capture and artifacts
            
        Returns:
            Stored capture ID as string
            
        Raises:
            DatabaseError: If storage fails
        """
        try:
            legacy_payload = not (
                isinstance(context, dict)
                and 'capture' in context
                and 'artifacts' in context
            )
            
            if not isinstance(context, dict):
                raise DatabaseError("Context payload must be a dictionary")
            
            if legacy_payload:
                capture_data = {
                    'source_window': context.get('source_window'),
                    'screenshot_path': context.get('screenshot_path'),
                    'clipboard_text': (
                        context.get('clipboard_text')
                        or context.get('clipboard')
                        or context.get('clipboard_content')
                    ),
                }
                
                try:
                    serialized = json.dumps(context, ensure_ascii=False, default=str)
                except Exception:
                    serialized = str(context)
                
                capture_data['notes'] = serialized
                
                existing_artifacts = context.get('artifacts')
                artifacts_data = existing_artifacts if isinstance(existing_artifacts, list) else []
                artifacts_data = list(artifacts_data)
                artifacts_data.append({
                    'kind': 'context_data',
                    'title': 'Context Payload',
                    'text': serialized,
                    'metadata': {
                        'legacy_payload': True,
                        'stored_at': datetime.now().isoformat()
                    }
                })
            else:
                capture_data = context.get('capture') or {}
                artifacts_data = context.get('artifacts') or []
            
            if not isinstance(artifacts_data, list):
                artifacts_data = []
            
            # Create capture
            capture_id = self.create_capture(
                source_window=capture_data.get('source_window'),
                screenshot_path=capture_data.get('screenshot_path'),
                clipboard_text=capture_data.get('clipboard_text'),
                notes=capture_data.get('notes')
            )
            
            # Create artifacts
            for artifact_data in artifacts_data:
                self.create_artifact(
                    capture_id=capture_id,
                    kind=artifact_data.get('kind'),
                    url=artifact_data.get('url'),
                    title=artifact_data.get('title'),
                    text=artifact_data.get('text'),
                    metadata=artifact_data.get('metadata')
                )
            
            self.logger.info(f"Stored context with capture ID: {capture_id}")
            return str(capture_id)
            
        except DatabaseError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to store context: {e}")
            raise DatabaseError(f"Failed to store context: {e}")
    
    def retrieve(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context data from the database (for ContextBox integration).
        
        Args:
            context_id: Context/capture ID
            
        Returns:
            Context data dictionary or None if not found
        """
        try:
            capture_id = int(context_id)
            
            # Get capture
            capture = self.get_capture(capture_id)
            if not capture:
                return None
            
            # Get artifacts
            artifacts = self.get_artifacts_by_capture(capture_id)
            
            result: Dict[str, Any] = {
                'capture': capture,
                'artifacts': artifacts
            }
            
            # Attempt to restore legacy payload for backwards compatibility
            notes = capture.get('notes')
            if notes:
                try:
                    legacy_payload = json.loads(notes)
                    if isinstance(legacy_payload, dict) and 'capture' not in legacy_payload:
                        merged_result = legacy_payload.copy()
                        merged_result['capture'] = capture
                        merged_result['artifacts'] = artifacts
                        return merged_result
                except (json.JSONDecodeError, TypeError):
                    pass
            
            return result
            
        except ValueError:
            self.logger.error(f"Invalid context ID: {context_id}")
            return None
        except DatabaseError as e:
            self.logger.error(f"Failed to retrieve context {context_id}: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Close database connections and perform cleanup.
        """
        self.logger.info("ContextDatabase cleanup completed")
    
    def close(self) -> None:
        """
        Compatibility close method for legacy callers.
        """
        self.logger.info("ContextDatabase close called")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM captures) as total_captures,
                        (SELECT COUNT(*) FROM artifacts) as total_artifacts
                """)
                
                row = cursor.fetchone()
                if not row:
                    row = {'total_captures': 0, 'total_artifacts': 0}
                stats = dict(row)
                stats.setdefault('total_captures', 0)
                stats.setdefault('total_artifacts', 0)
                stats['context_count'] = stats['total_captures']
                stats['extraction_count'] = stats['total_artifacts']
                return stats
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {
                'total_captures': 0,
                'total_artifacts': 0,
                'context_count': 0,
                'extraction_count': 0
            }
    
    def backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a database backup.
        
        Args:
            backup_path: Path for backup file. If None, creates timestamped backup
            
        Returns:
            True if backup was successful
            
        Raises:
            DatabaseError: If backup fails
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_contextbox_{timestamp}.db"
            
            with self._get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to backup database: {e}")
            raise DatabaseError(f"Failed to backup database: {e}")
    
    def store_extraction_result(self, 
                               capture_id: int,
                               extraction_result: Dict[str, Any],
                               auto_create_capture: bool = True) -> bool:
        """
        Store content extraction results in the database.
        
        This method stores extraction results by creating appropriate artifacts
        for different types of extracted content including URLs, text, OCR results,
        and metadata.
        
        Args:
            capture_id: ID of the associated capture
            extraction_result: Extraction results dictionary
            auto_create_capture: Whether to create capture if it doesn't exist
            
        Returns:
            True if storage was successful
            
        Raises:
            DatabaseError: If storage fails
        """
        try:
            # Verify capture exists or create it
            capture = self.get_capture(capture_id)
            if not capture and auto_create_capture:
                capture_id = self.create_capture(
                    source_window="Content Extraction",
                    notes=f"Auto-created for extraction result: {extraction_result.get('extraction_id', 'unknown')}"
                )
            elif not capture:
                raise DatabaseError(f"Capture with ID {capture_id} not found")
            
            stored_artifacts = 0
            
            # Store URL analysis results
            if 'url_analysis' in extraction_result:
                url_analysis = extraction_result['url_analysis']
                
                # Store high confidence URLs
                if url_analysis.get('high_confidence'):
                    for url in url_analysis['high_confidence']:
                        self.create_artifact(
                            capture_id=capture_id,
                            kind='url',
                            url=url,
                            title='High Confidence URL',
                            metadata={
                                'confidence': 'high',
                                'extraction_method': 'automatic',
                                'extraction_id': extraction_result.get('extraction_id')
                            }
                        )
                        stored_artifacts += 1
                
                # Store URL analysis summary
                analysis_summary = {
                    'total_count': url_analysis.get('total_count', 0),
                    'by_type': url_analysis.get('by_type', {}),
                    'top_domains': dict(list(url_analysis.get('by_domain', {}).items())[:10])
                }
                
                self.create_artifact(
                    capture_id=capture_id,
                    kind='url_analysis',
                    title='URL Analysis Summary',
                    text=json.dumps(analysis_summary),
                    metadata={
                        'analysis_type': 'url_summary',
                        'extraction_id': extraction_result.get('extraction_id')
                    }
                )
                stored_artifacts += 1
            
            # Store context extraction results
            if 'context_extraction' in extraction_result:
                context_data = extraction_result['context_extraction']
                
                # Store enhanced features
                if 'enhanced_features' in context_data:
                    features = context_data['enhanced_features']
                    
                    # OCR extraction results
                    if 'ocr_extraction' in features:
                        ocr_data = features['ocr_extraction']
                        self.create_artifact(
                            capture_id=capture_id,
                            kind='ocr_text',
                            title='OCR Extracted Text',
                            text=ocr_data.get('text', ''),
                            metadata={
                                'confidence': ocr_data.get('confidence', 0),
                                'bbox': ocr_data.get('bbox'),
                                'extraction_method': 'ocr',
                                'word_count': len(ocr_data.get('text', '').split()),
                                'extraction_id': extraction_result.get('extraction_id')
                            }
                        )
                        stored_artifacts += 1
                    
                    # URL extraction from enhanced features
                    for url_key in ['ocr_urls', 'enhanced_url_extraction', 'clipboard_urls']:
                        if url_key in features:
                            url_data = features[url_key]
                            if url_key == 'clipboard_urls':
                                # Handle clipboard URLs (list of dicts)
                                urls_list = url_data
                            else:
                                # Handle structured URL data
                                urls_list = []
                                if 'direct_urls' in url_data:
                                    urls_list.extend(url_data['direct_urls'])
                                if 'inferred_domains' in url_data:
                                    urls_list.extend(url_data['inferred_domains'])
                            
                            for url_item in urls_list:
                                if isinstance(url_item, dict):
                                    url_text = url_item.get('normalized_url', url_item.get('url', ''))
                                else:
                                    url_text = str(url_item)
                                
                                if url_text:
                                    self.create_artifact(
                                        capture_id=capture_id,
                                        kind='extracted_url',
                                        url=url_text,
                                        title=f'Extracted URL ({url_key})',
                                        metadata={
                                            'extraction_method': url_key,
                                            'extraction_id': extraction_result.get('extraction_id'),
                                            'confidence': url_item.get('confidence', 0) if isinstance(url_item, dict) else None
                                        }
                                    )
                                    stored_artifacts += 1
            
            # Store extraction summary
            if 'summary' in extraction_result:
                summary = extraction_result['summary']
                self.create_artifact(
                    capture_id=capture_id,
                    kind='extraction_summary',
                    title='Content Extraction Summary',
                    text=json.dumps(summary, indent=2),
                    metadata={
                        'extraction_method': extraction_result.get('type', 'unknown'),
                        'extraction_id': extraction_result.get('extraction_id'),
                        'timestamp': extraction_result.get('timestamp'),
                        'success_count': summary.get('successful_extractions', 0),
                        'failure_count': summary.get('failed_extractions', 0)
                    }
                )
                stored_artifacts += 1
            
            # Store metadata if available
            if 'metadata' in extraction_result:
                metadata = extraction_result['metadata']
                self.create_artifact(
                    capture_id=capture_id,
                    kind='extraction_metadata',
                    title='Extraction Metadata',
                    text=json.dumps(metadata, indent=2),
                    metadata={
                        'metadata_type': 'extraction_info',
                        'extraction_id': extraction_result.get('extraction_id')
                    }
                )
                stored_artifacts += 1
            
            self.logger.info(f"Stored {stored_artifacts} artifacts for extraction result {extraction_result.get('extraction_id', 'unknown')}")
            return True
            
        except DatabaseError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to store extraction result: {e}")
            raise DatabaseError(f"Failed to store extraction result: {e}")
    
    def get_extraction_results(self, capture_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all extraction-related artifacts for a capture.
        
        Args:
            capture_id: ID of the capture
            
        Returns:
            List of extraction-related artifacts
        """
        try:
            all_artifacts = self.get_artifacts_by_capture(capture_id)
            
            # Filter for extraction-related artifacts
            extraction_kinds = [
                'url', 'url_analysis', 'ocr_text', 'extracted_url', 
                'extraction_summary', 'extraction_metadata'
            ]
            
            extraction_artifacts = [
                artifact for artifact in all_artifacts 
                if artifact['kind'] in extraction_kinds
            ]
            
            return extraction_artifacts
            
        except DatabaseError as e:
            self.logger.error(f"Failed to retrieve extraction results: {e}")
            raise
    
    def search_extraction_artifacts(self, 
                                   search_term: str,
                                   artifact_kind: Optional[str] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search through extraction artifacts for specific terms.
        
        Args:
            search_term: Term to search for
            artifact_kind: Optional filter by artifact type
            limit: Maximum number of results
            
        Returns:
            List of matching artifacts
        """
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM artifacts 
                    WHERE (text LIKE ? OR url LIKE ? OR title LIKE ?)
                """
                params = [f'%{search_term}%', f'%{search_term}%', f'%{search_term}%']
                
                if artifact_kind:
                    query += " AND kind = ?"
                    params.append(artifact_kind)
                
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    artifact = dict(row)
                    
                    # Parse metadata JSON
                    if artifact['metadata_json']:
                        artifact['metadata'] = json.loads(artifact['metadata_json'])
                        del artifact['metadata_json']
                    else:
                        artifact['metadata'] = None
                        del artifact['metadata_json']
                    
                    results.append(artifact)
                
                return results
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to search extraction artifacts: {e}")
            raise DatabaseError(f"Failed to search artifacts: {e}")
