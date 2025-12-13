"""
Main ContextBox application module.
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from .capture import ContextCapture
from .database import ContextDatabase
from .extractors.enhanced_extractors import ContentExtractor, ContentExtractionError
# Import the actual ContextExtractor directly from extractors.py
try:
    from .extractors import EnhancedContextExtractor as ContextExtractorClass
except (ImportError, TypeError):
    # Fallback if the package import fails
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "contextbox.extractors_module",
        os.path.join(os.path.dirname(__file__), 'extractors.py')
    )
    if spec and spec.loader:
        extractors_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractors_module)
        ContextExtractorClass = extractors_module.EnhancedContextExtractor
    else:
        ContextExtractorClass = None
from .utils import setup_logging
from .config import get_config, ContextBoxConfig


class ContextBox:
    """
    Main ContextBox application class.
    
    Orchestrates context capture, extraction, and storage operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, profile: str = "default"):
        """
        Initialize ContextBox application.
        
        Args:
            config: Configuration dictionary (legacy)
            profile: Configuration profile to use
        """
        # Load enhanced configuration
        if config is None:
            # Use enhanced config system
            try:
                self.config_obj = get_config(profile)
                self.config = self.config_obj.to_dict()
            except Exception as e:
                logging.warning(f"Could not load enhanced config, using defaults: {e}")
                self.config_obj = ContextBoxConfig()
                self.config = self.config_obj.to_dict()
        else:
            # Legacy config support
            self.config_obj = None
            self.config = config
        
        setup_logging(self.config.get('log_level', 'INFO'))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components  
        capture_config = self.config.get('capture', {})
        database_config = self.config.get('database', {})
        
        self.database = ContextDatabase(database_config)
        self.capture = ContextCapture(capture_config, database=self.database)
        
        # Initialize the legacy/compatibility extractor
        self.extractor = None
        extractor_config = self.config.get('extractors', {})
        if ContextExtractorClass is not None:
            try:
                self.extractor = ContextExtractorClass(extractor_config)
                self.logger.info("Legacy context extractor initialized")
            except Exception as exc:
                self.logger.warning(f"Primary extractor initialization failed: {exc}")
        else:
            self.logger.warning("ContextExtractor class not available")
            self.extractor = None
        
        # Initialize content extractor for advanced content extraction
        self.content_extractor = None
        # Check both legacy and new config formats
        content_extraction_enabled = self.config.get('enable_content_extraction', True)
        content_extraction_config = self.config.get('content_extraction', {})
        if content_extraction_enabled and content_extraction_config.get('enabled', True):
            try:
                self.content_extractor = ContentExtractor(content_extraction_config)
                self.logger.info("Content extraction enabled")
            except Exception as e:
                self.logger.warning(f"Content extraction initialization failed: {e}")
                self.content_extractor = None
        
        self.logger.info("ContextBox initialized")
    
    def start_capture(self) -> None:
        """Start context capture operations."""
        self.logger.info("Starting context capture")
        self.capture.start()
    
    def stop_capture(self) -> None:
        """Stop context capture operations."""
        self.logger.info("Stopping context capture")
        self.capture.stop()
    
    def extract_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context from captured data.
        
        Args:
            data: Raw captured data
            
        Returns:
            Extracted context information
        """
        self.logger.info("Extracting context from data")
        
        if self.extractor:
            return self.extractor.extract(data)
        
        if self.content_extractor:
            self.logger.debug("Falling back to content extractor for basic extraction")
            try:
                return self.content_extractor.extractor.extract(data)
            except Exception as exc:
                self.logger.error(f"Fallback extraction failed: {exc}")
        
        self.logger.warning("No extractor available, returning original data as context")
        return {'raw_context': data}
    
    def store_context(self, context: Dict[str, Any]) -> str:
        """
        Store extracted context in database.
        
        Args:
            context: Extracted context information
            
        Returns:
            Context ID in database
        """
        self.logger.info("Storing context in database")
        prepared_context = self._prepare_context_for_storage(context)
        return self.database.store(prepared_context)
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context from database.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Stored context information or None
        """
        self.logger.info(f"Retrieving context: {context_id}")
        return self.database.retrieve(context_id)
    
    def capture_context(self, capture_data: Dict[str, Any]) -> str:
        """
        Capture and process context data end-to-end.
        
        Args:
            capture_data: Raw capture data (screenshot, system info, etc.)
            
        Returns:
            Context ID in database
        """
        self.logger.info("Starting complete context capture workflow")
        
        try:
            # Step 1: Extract context from captured data
            extracted_context = self.extract_context(capture_data)
            
            # Step 2: Combine captured and extracted data
            full_context = {
                **capture_data,
                'extracted_context': extracted_context,
                'capture_workflow': 'completed'
            }
            
            # Step 3: Store in database
            context_id = self.store_context(full_context)
            
            self.logger.info(f"Context capture workflow completed: {context_id}")
            return context_id
            
        except Exception as e:
            self.logger.error(f"Error in context capture workflow: {e}")
            raise
    
    def list_contexts(self, limit: int = 50) -> list:
        """
        List stored contexts from database.
        
        Args:
            limit: Maximum number of contexts to return
            
        Returns:
            List of context summaries
        """
        self.logger.info("Listing stored contexts")
        
        try:
            # Get database statistics
            stats = self.database.get_stats()
            
            # For now, return database stats since full listing requires more complex queries
            return {
                'stats': stats,
                'message': 'Use database query methods for detailed listings'
            }
            
        except Exception as e:
            self.logger.error(f"Error listing contexts: {e}")
            raise
    
    def extract_content_from_capture(self, capture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from context capture with enhanced capabilities.
        
        This method provides automatic content extraction when URLs are found during
        context capture. It uses the ContentExtractor to perform advanced content
        extraction including OCR, URL analysis, and content processing.
        
        Args:
            capture_data: Context capture data
            
        Returns:
            Enhanced extraction results with rich metadata
        """
        self.logger.info("Starting enhanced content extraction from capture")
        
        if not self.content_extractor:
            self.logger.warning("Content extractor not initialized, falling back to basic extraction")
            return self.extract_context(capture_data)
        
        try:
            # Perform automatic content extraction
            extraction_result = self.content_extractor.extract_from_capture(capture_data)
            
            # Store extracted content in database if configured
            if self.content_extractor.store_in_database:
                try:
                    self._store_extracted_content(capture_data, extraction_result)
                except Exception as e:
                    self.logger.warning(f"Failed to store extracted content: {e}")
            
            return extraction_result
            
        except ContentExtractionError as e:
            self.logger.error(f"Content extraction error: {e}")
            # Fallback to basic extraction
            return self.extract_context(capture_data)
        except Exception as e:
            self.logger.error(f"Unexpected error in content extraction: {e}")
            raise
    
    def extract_content_manually(self, 
                                input_data: Dict[str, Any],
                                extract_urls: bool = True,
                                extract_text: bool = True,
                                extract_images: bool = True,
                                output_format: str = 'json') -> str:
        """
        Manually extract content from provided data.
        
        This method provides manual content extraction capabilities for CLI and API use.
        
        Args:
            input_data: Input data for extraction (text, images, URLs, etc.)
            extract_urls: Whether to extract URLs from the data
            extract_text: Whether to extract text content
            extract_images: Whether to extract and process images
            output_format: Output format ('json', 'pretty', 'summary', 'detailed')
            
        Returns:
            Formatted extraction results
        """
        self.logger.info("Starting manual content extraction")
        
        if not self.content_extractor:
            raise ContentExtractionError("Content extractor not initialized")
        
        try:
            # Perform manual extraction
            extraction_result = self.content_extractor.extract_content_manually(
                input_data, extract_urls, extract_text, extract_images
            )
            
            # Format output
            formatted_output = self.content_extractor.format_extraction_result(
                extraction_result, output_format
            )
            
            return formatted_output
            
        except ContentExtractionError:
            raise
        except Exception as e:
            self.logger.error(f"Manual content extraction failed: {e}")
            raise ContentExtractionError(f"Manual extraction failed: {e}")
    
    def extract_urls_only(self, text: str, include_clipboard: bool = True) -> list:
        """
        Extract only URLs from text with optional clipboard integration.
        
        Args:
            text: Text to extract URLs from
            include_clipboard: Whether to also check clipboard for URLs
            
        Returns:
            List of extracted URLs
        """
        if not self.content_extractor:
            # Fallback to basic URL extraction - use regex as fallback
            import re
            # Find URLs with or without protocol
            url_pattern = r'(?:https?://)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s<>"\'{}|\\^`\[\]]*)?'
            urls = re.findall(url_pattern, text)
            # Add https:// prefix if missing
            formatted_urls = []
            for url in urls:
                if not url.startswith(('http://', 'https://')):
                    formatted_urls.append('https://' + url)
                else:
                    formatted_urls.append(url)
            return formatted_urls
        
        try:
            urls = self.content_extractor.extract_urls_only(text, include_clipboard)
            return [url.normalized_url for url in urls]
        except Exception as e:
            self.logger.error(f"URL extraction failed: {e}")
            return []
    
    def extract_ocr_only(self, image_path: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract only OCR text from image.
        
        Args:
            image_path: Path to image file
            config: OCR configuration options
            
        Returns:
            Extracted text
        """
        if not self.content_extractor:
            raise ContentExtractionError("Content extractor not initialized")
        
        try:
            ocr_result = self.content_extractor.extract_ocr_only(image_path, config)
            return ocr_result.text
        except ContentExtractionError:
            raise
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise ContentExtractionError(f"OCR extraction failed: {e}")
    
    def get_content_extraction_config(self) -> Dict[str, Any]:
        """
        Get current content extraction configuration.
        
        Returns:
            Current configuration dictionary
        """
        if not self.content_extractor:
            return {
                'enabled': False,
                'reason': 'Content extractor not initialized'
            }
        
        config = self.content_extractor.get_extraction_config()
        # Add 'enabled' key for compatibility with tests
        config['enabled'] = True
        return config
    
    def update_content_extraction_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update content extraction configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        if not self.content_extractor:
            self.logger.warning("Content extractor not initialized, cannot update config")
            return
        
        try:
            self.content_extractor.update_extraction_config(config_updates)
            self.logger.info(f"Updated content extraction configuration: {config_updates}")
        except Exception as e:
            self.logger.error(f"Failed to update content extraction config: {e}")
            raise ContentExtractionError(f"Config update failed: {e}")
    
    def _store_extracted_content(self, capture_data: Dict[str, Any], extraction_result: Dict[str, Any]) -> None:
        """
        Store extracted content in the database.
        
        Args:
            capture_data: Original capture data
            extraction_result: Extraction results
        """
        try:
            # Get capture ID or create new capture
            capture_id = None
            if 'capture_id' in capture_data:
                capture_id = int(capture_data['capture_id'])
            else:
                # Create a new capture record
                source_window = capture_data.get('active_window', {}).get('title', 'Unknown')
                screenshot_path = capture_data.get('screenshot_path')
                clipboard_text = capture_data.get('clipboard_text')
                
                capture_id = self.database.create_capture(
                    source_window=source_window,
                    screenshot_path=screenshot_path,
                    clipboard_text=clipboard_text
                )
                self.logger.debug(f"Created new capture record with ID: {capture_id}")
            
            # Use the new database method to store extraction results
            self.database.store_extraction_result(capture_id, extraction_result)
            
        except Exception as e:
            self.logger.error(f"Failed to store extracted content: {e}")
            raise

    def _prepare_context_for_storage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize context payloads for database storage.
        
        Args:
            context: Raw context payload from extractors or callers
            
        Returns:
            Dictionary conforming to ContextDatabase.store expectations
        """
        if isinstance(context, dict) and 'capture' in context and 'artifacts' in context:
            return context
        
        capture_source = None
        screenshot_path = None
        clipboard_text = None
        
        if isinstance(context, dict):
            capture_source = (
                context.get('active_window', {}).get('title')
                if isinstance(context.get('active_window'), dict)
                else None
            )
            screenshot_path = context.get('screenshot_path')
            clipboard_text = (
                context.get('clipboard')
                or context.get('clipboard_text')
                or context.get('clipboard_content')
            )
        
        try:
            serialized_context = json.dumps(context, ensure_ascii=False, default=str)
        except Exception:
            serialized_context = str(context)
        
        artifacts: List[Dict[str, Any]] = []
        if isinstance(context, dict):
            embedded_artifacts = context.get('artifacts')
            if isinstance(embedded_artifacts, list):
                artifacts.extend(embedded_artifacts)  # type: ignore[arg-type]
        
        artifacts.append({
            'kind': 'context_snapshot',
            'title': 'Context Data',
            'text': serialized_context,
            'metadata': {
                'stored_at': datetime.now().isoformat(),
                'source': 'ContextBox.store_context',
            }
        })
        
        capture_payload = {
            'source_window': capture_source,
            'screenshot_path': screenshot_path,
            'clipboard_text': clipboard_text,
            'notes': serialized_context
        }
        
        return {
            'capture': capture_payload,
            'artifacts': artifacts
        }
