"""
Data Export and Import System for ContextBox

This module provides comprehensive data export and import capabilities
for backup, migration, and data management purposes.
"""

import json
import csv
import sqlite3
import zipfile
import logging
import hashlib
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class DataExportImportError(Exception):
    """Custom exception for export/import operations."""
    pass


class DataExporter:
    """
    Comprehensive data exporter for ContextBox.
    """
    
    def __init__(self, db_path: str = 'contextbox.db'):
        """Initialize data exporter."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Export configuration
        self.default_formats = ['json', 'csv', 'xml']
        self.supported_formats = self.default_formats + (['xlsx'] if PANDAS_AVAILABLE else [])
        
        self.logger.info(f"Data exporter initialized for {self.supported_formats}")
    
    def export_all_data(self, 
                       export_path: str,
                       formats: Optional[List[str]] = None,
                       include_artifacts: bool = True,
                       include_metadata: bool = True,
                       compression: bool = True) -> Dict[str, Any]:
        """
        Export all ContextBox data in multiple formats.
        
        Args:
            export_path: Path for exported data
            formats: List of formats to export (default: all supported)
            include_artifacts: Whether to include artifacts data
            include_metadata: Whether to include metadata
            compression: Whether to compress the export
            
        Returns:
            Export results and statistics
        """
        formats = formats or self.supported_formats
        export_path = Path(export_path)
        
        # Create export directory
        export_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'export_timestamp': datetime.now().isoformat(),
            'source_database': str(self.db_path),
            'formats_exported': [],
            'files_created': [],
            'statistics': {},
            'errors': []
        }
        
        try:
            # Get all data
            data = self._extract_all_data(include_artifacts, include_metadata)
            
            # Export in requested formats
            for format_type in formats:
                try:
                    if format_type.lower() == 'json':
                        result = self._export_json(data, export_path, 'contextbox_data')
                        results['formats_exported'].append('json')
                    elif format_type.lower() == 'csv':
                        result = self._export_csv(data, export_path, 'contextbox_data')
                        results['formats_exported'].append('csv')
                    elif format_type.lower() == 'xml':
                        result = self._export_xml(data, export_path, 'contextbox_data')
                        results['formats_exported'].append('xml')
                    elif format_type.lower() == 'xlsx' and PANDAS_AVAILABLE:
                        result = self._export_xlsx(data, export_path, 'contextbox_data')
                        results['formats_exported'].append('xlsx')
                    else:
                        results['errors'].append(f"Unsupported format: {format_type}")
                        continue
                    
                    results['files_created'].extend(result['files'])
                    results['statistics'][format_type] = result['statistics']
                    
                except Exception as e:
                    error_msg = f"Failed to export {format_type}: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Create backup manifest
            manifest = self._create_export_manifest(results)
            manifest_file = export_path / 'export_manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            results['files_created'].append(str(manifest_file))
            
            # Compress if requested
            if compression and len(results['files_created']) > 1:
                compression_result = self._compress_export(export_path, results['files_created'])
                if compression_result['success']:
                    # Update file list with compressed archive
                    compressed_file = compression_result['archive_path']
                    results['files_created'] = [compressed_file]
                    results['compression'] = compression_result
            
            results['total_files'] = len(results['files_created'])
            results['success'] = len(results['errors']) == 0
            
            self.logger.info(f"Data export completed: {results['total_files']} files created")
            return results
            
        except Exception as e:
            error_msg = f"Data export failed: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            results['success'] = False
            return results
    
    def _extract_all_data(self, include_artifacts: bool, include_metadata: bool) -> Dict[str, Any]:
        """Extract all data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Extract captures
                captures_cursor = conn.execute("SELECT * FROM captures ORDER BY created_at DESC")
                captures = [dict(row) for row in captures_cursor.fetchall()]
                
                data = {
                    'export_info': {
                        'source_database': str(self.db_path),
                        'export_timestamp': datetime.now().isoformat(),
                        'database_version': '1.0',
                        'contextbox_version': '1.0.0'
                    },
                    'captures': captures
                }
                
                # Extract artifacts if requested
                if include_artifacts:
                    artifacts_cursor = conn.execute("""
                        SELECT a.*, c.created_at as capture_created_at 
                        FROM artifacts a 
                        LEFT JOIN captures c ON a.capture_id = c.id 
                        ORDER BY a.id
                    """)
                    artifacts = []
                    
                    for row in artifacts_cursor.fetchall():
                        artifact = dict(row)
                        if not include_metadata and 'metadata_json' in artifact:
                            del artifact['metadata_json']
                        artifacts.append(artifact)
                    
                    data['artifacts'] = artifacts
                
                # Add statistics
                stats_cursor = conn.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM captures) as total_captures,
                        (SELECT COUNT(*) FROM artifacts) as total_artifacts
                """)
                
                row = stats_cursor.fetchone()
                if row:
                    data['statistics'] = dict(row)
                else:
                    data['statistics'] = {'total_captures': 0, 'total_artifacts': 0}
                
                return data
                
        except sqlite3.Error as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise DataExportImportError(f"Data extraction failed: {e}")
    
    def _export_json(self, data: Dict[str, Any], export_path: Path, base_name: str) -> Dict[str, Any]:
        """Export data as JSON."""
        json_file = export_path / f"{base_name}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            file_size = json_file.stat().st_size
            return {
                'files': [str(json_file)],
                'statistics': {
                    'file_size_mb': file_size / (1024 * 1024),
                    'format': 'json'
                }
            }
            
        except Exception as e:
            raise DataExportImportError(f"JSON export failed: {e}")
    
    def _export_csv(self, data: Dict[str, Any], export_path: Path, base_name: str) -> Dict[str, Any]:
        """Export data as CSV files."""
        files_created = []
        
        try:
            # Export captures
            captures_file = export_path / f"{base_name}_captures.csv"
            with open(captures_file, 'w', newline='', encoding='utf-8') as f:
                if data['captures']:
                    writer = csv.DictWriter(f, fieldnames=data['captures'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['captures'])
            files_created.append(str(captures_file))
            
            # Export artifacts if available
            if 'artifacts' in data and data['artifacts']:
                artifacts_file = export_path / f"{base_name}_artifacts.csv"
                with open(artifacts_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data['artifacts'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['artifacts'])
                files_created.append(str(artifacts_file))
            
            total_size = sum(Path(f).stat().st_size for f in files_created)
            return {
                'files': files_created,
                'statistics': {
                    'file_size_mb': total_size / (1024 * 1024),
                    'format': 'csv',
                    'files_count': len(files_created)
                }
            }
            
        except Exception as e:
            raise DataExportImportError(f"CSV export failed: {e}")
    
    def _export_xml(self, data: Dict[str, Any], export_path: Path, base_name: str) -> Dict[str, Any]:
        """Export data as XML."""
        xml_file = export_path / f"{base_name}.xml"
        
        try:
            root = ET.Element("ContextBoxData")
            root.set("export_timestamp", data['export_info']['export_timestamp'])
            root.set("source_database", data['export_info']['source_database'])
            
            # Add export info
            info_elem = ET.SubElement(root, "ExportInfo")
            for key, value in data['export_info'].items():
                elem = ET.SubElement(info_elem, key)
                elem.text = str(value)
            
            # Add captures
            captures_elem = ET.SubElement(root, "Captures")
            for capture in data['captures']:
                capture_elem = ET.SubElement(captures_elem, "Capture")
                for key, value in capture.items():
                    elem = ET.SubElement(capture_elem, key)
                    elem.text = str(value) if value is not None else ""
            
            # Add artifacts if available
            if 'artifacts' in data and data['artifacts']:
                artifacts_elem = ET.SubElement(root, "Artifacts")
                for artifact in data['artifacts']:
                    artifact_elem = ET.SubElement(artifacts_elem, "Artifact")
                    for key, value in artifact.items():
                        elem = ET.SubElement(artifact_elem, key)
                        elem.text = str(value) if value is not None else ""
            
            # Add statistics
            if 'statistics' in data:
                stats_elem = ET.SubElement(root, "Statistics")
                for key, value in data['statistics'].items():
                    elem = ET.SubElement(stats_elem, key)
                    elem.text = str(value)
            
            # Write XML
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
            file_size = xml_file.stat().st_size
            return {
                'files': [str(xml_file)],
                'statistics': {
                    'file_size_mb': file_size / (1024 * 1024),
                    'format': 'xml'
                }
            }
            
        except Exception as e:
            raise DataExportImportError(f"XML export failed: {e}")
    
    def _export_xlsx(self, data: Dict[str, Any], export_path: Path, base_name: str) -> Dict[str, Any]:
        """Export data as Excel file."""
        if not PANDAS_AVAILABLE:
            raise DataExportImportError("Pandas not available for Excel export")
        
        xlsx_file = export_path / f"{base_name}.xlsx"
        
        try:
            with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
                # Export captures sheet
                if data['captures']:
                    captures_df = pd.DataFrame(data['captures'])
                    captures_df.to_excel(writer, sheet_name='Captures', index=False)
                
                # Export artifacts sheet
                if 'artifacts' in data and data['artifacts']:
                    artifacts_df = pd.DataFrame(data['artifacts'])
                    artifacts_df.to_excel(writer, sheet_name='Artifacts', index=False)
                
                # Add statistics sheet
                if 'statistics' in data:
                    stats_df = pd.DataFrame([data['statistics']])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            file_size = xlsx_file.stat().st_size
            return {
                'files': [str(xlsx_file)],
                'statistics': {
                    'file_size_mb': file_size / (1024 * 1024),
                    'format': 'xlsx'
                }
            }
            
        except Exception as e:
            raise DataExportImportError(f"Excel export failed: {e}")
    
    def _create_export_manifest(self, export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create export manifest."""
        manifest = {
            'export_info': {
                'timestamp': export_results['export_timestamp'],
                'source_database': export_results['source_database'],
                'total_files': export_results['total_files'],
                'formats_exported': export_results['formats_exported'],
                'success': export_results['success']
            },
            'files': []
        }
        
        for file_path in export_results['files_created']:
            file_path = Path(file_path)
            if file_path.exists():
                file_info = {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size_bytes': file_path.stat().st_size,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'hash': self._calculate_file_hash(file_path)
                }
                manifest['files'].append(file_info)
        
        return manifest
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for integrity verification."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def _compress_export(self, export_path: Path, files: List[str]) -> Dict[str, Any]:
        """Compress export files into archive."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"contextbox_export_{timestamp}.zip"
            archive_path = export_path / archive_name
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    file_path = Path(file_path)
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)
            
            return {
                'success': True,
                'archive_path': str(archive_path),
                'archive_size_mb': archive_path.stat().st_size / (1024 * 1024),
                'compressed_files': len(files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_captures_only(self, export_path: str, format_type: str = 'json') -> Dict[str, Any]:
        """Export only captures data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM captures ORDER BY created_at DESC")
                captures = [dict(row) for row in cursor.fetchall()]
            
            data = {
                'export_info': {
                    'type': 'captures_only',
                    'export_timestamp': datetime.now().isoformat(),
                    'total_captures': len(captures)
                },
                'captures': captures
            }
            
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'json':
                return self._export_json(data, export_path, 'captures_only')
            elif format_type.lower() == 'csv':
                # Create individual CSV files for captures
                return self._export_csv(data, export_path, 'captures_only')
            else:
                raise DataExportImportError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise DataExportImportError(f"Captures export failed: {e}")
    
    def export_artifacts_only(self, export_path: str, format_type: str = 'json', 
                            capture_id: Optional[int] = None) -> Dict[str, Any]:
        """Export only artifacts data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if capture_id:
                    cursor = conn.execute("""
                        SELECT a.*, c.created_at as capture_created_at 
                        FROM artifacts a 
                        LEFT JOIN captures c ON a.capture_id = c.id 
                        WHERE a.capture_id = ?
                        ORDER BY a.id
                    """, (capture_id,))
                else:
                    cursor = conn.execute("""
                        SELECT a.*, c.created_at as capture_created_at 
                        FROM artifacts a 
                        LEFT JOIN captures c ON a.capture_id = c.id 
                        ORDER BY a.id
                    """)
                
                artifacts = [dict(row) for row in cursor.fetchall()]
            
            data = {
                'export_info': {
                    'type': 'artifacts_only',
                    'export_timestamp': datetime.now().isoformat(),
                    'total_artifacts': len(artifacts),
                    'capture_filter': capture_id
                },
                'artifacts': artifacts
            }
            
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'json':
                return self._export_json(data, export_path, 'artifacts_only')
            elif format_type.lower() == 'csv':
                return self._export_csv(data, export_path, 'artifacts_only')
            else:
                raise DataExportImportError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise DataExportImportError(f"Artifacts export failed: {e}")


class DataImporter:
    """
    Comprehensive data importer for ContextBox.
    """
    
    def __init__(self, db_path: str = 'contextbox.db'):
        """Initialize data importer."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Import configuration
        self.supported_formats = ['json', 'xml', 'csv', 'xlsx']
        self.merge_mode = False  # Whether to merge with existing data
        
        self.logger.info("Data importer initialized")
    
    def import_data(self, 
                   import_path: str,
                   format_type: Optional[str] = None,
                   merge_mode: bool = False,
                   backup_before_import: bool = True) -> Dict[str, Any]:
        """
        Import data from various formats.
        
        Args:
            import_path: Path to imported data
            format_type: Format type (auto-detected if None)
            merge_mode: Whether to merge with existing data
            backup_before_import: Whether to backup existing data
            
        Returns:
            Import results and statistics
        """
        import_path = Path(import_path)
        results = {
            'import_timestamp': datetime.now().isoformat(),
            'import_path': str(import_path),
            'format_detected': None,
            'success': False,
            'records_imported': {
                'captures': 0,
                'artifacts': 0
            },
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create backup if requested
            if backup_before_import:
                backup_result = self._create_backup()
                results['backup_created'] = backup_result
            
            # Detect format
            if not format_type:
                format_type = self._detect_format(import_path)
            
            results['format_detected'] = format_type
            
            # Load data based on format
            if format_type.lower() == 'json':
                data = self._load_json_data(import_path)
            elif format_type.lower() == 'xml':
                data = self._load_xml_data(import_path)
            elif format_type.lower() == 'csv':
                data = self._load_csv_data(import_path)
            elif format_type.lower() == 'xlsx':
                data = self._load_xlsx_data(import_path)
            else:
                raise DataExportImportError(f"Unsupported import format: {format_type}")
            
            # Validate data structure
            validation_result = self._validate_import_data(data)
            if not validation_result['valid']:
                results['errors'].extend(validation_result['errors'])
                return results
            
            # Import data
            import_stats = self._import_data_to_database(data, merge_mode)
            results['records_imported'] = import_stats
            
            # Verify import
            verification_result = self._verify_import(import_stats)
            results['verification'] = verification_result
            
            results['success'] = len(results['errors']) == 0
            
            self.logger.info(f"Data import completed: {import_stats}")
            return results
            
        except Exception as e:
            error_msg = f"Data import failed: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            results['success'] = False
            return results
    
    def _detect_format(self, import_path: Path) -> str:
        """Detect file format from extension or content."""
        # Check file extension first
        if import_path.suffix.lower() == '.json':
            return 'json'
        elif import_path.suffix.lower() == '.xml':
            return 'xml'
        elif import_path.suffix.lower() == '.csv':
            return 'csv'
        elif import_path.suffix.lower() in ['.xlsx', '.xls']:
            return 'xlsx'
        
        # Check content if extension is not clear
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
                
                if content.strip().startswith('{') or content.strip().startswith('['):
                    return 'json'
                elif '<ContextBoxData' in content or '<?xml' in content:
                    return 'xml'
                elif ',' in content and '\n' in content:
                    return 'csv'
                    
        except Exception:
            pass
        
        raise DataExportImportError("Could not detect file format")
    
    def _load_json_data(self, import_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise DataExportImportError(f"Failed to load JSON data: {e}")
    
    def _load_xml_data(self, import_path: Path) -> Dict[str, Any]:
        """Load data from XML file."""
        try:
            tree = ET.parse(import_path)
            root = tree.getroot()
            
            data = {'export_info': {}, 'captures': [], 'artifacts': []}
            
            # Parse export info
            info_elem = root.find('ExportInfo')
            if info_elem is not None:
                for child in info_elem:
                    data['export_info'][child.tag] = child.text
            
            # Parse captures
            captures_elem = root.find('Captures')
            if captures_elem is not None:
                for capture_elem in captures_elem:
                    capture = {}
                    for child in capture_elem:
                        capture[child.tag] = child.text
                    data['captures'].append(capture)
            
            # Parse artifacts
            artifacts_elem = root.find('Artifacts')
            if artifacts_elem is not None:
                for artifact_elem in artifacts_elem:
                    artifact = {}
                    for child in artifact_elem:
                        artifact[child.tag] = child.text
                    data['artifacts'].append(artifact)
            
            return data
            
        except Exception as e:
            raise DataExportImportError(f"Failed to load XML data: {e}")
    
    def _load_csv_data(self, import_path: Path) -> Dict[str, Any]:
        """Load data from CSV files."""
        try:
            data = {'captures': [], 'artifacts': []}
            
            # Look for captures file
            captures_file = import_path / 'contextbox_data_captures.csv'
            if captures_file.exists():
                with open(captures_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['captures'] = list(reader)
            
            # Look for artifacts file
            artifacts_file = import_path / 'contextbox_data_artifacts.csv'
            if artifacts_file.exists():
                with open(artifacts_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['artifacts'] = list(reader)
            
            # If no specific files, assume the CSV file itself contains the data
            if not data['captures'] and not data['artifacts'] and import_path.suffix == '.csv':
                with open(import_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Assume it's captures data
                    data['captures'] = list(reader)
            
            return data
            
        except Exception as e:
            raise DataExportImportError(f"Failed to load CSV data: {e}")
    
    def _load_xlsx_data(self, import_path: Path) -> Dict[str, Any]:
        """Load data from Excel file."""
        if not PANDAS_AVAILABLE:
            raise DataExportImportError("Pandas not available for Excel import")
        
        try:
            data = {'captures': [], 'artifacts': []}
            
            # Read captures sheet
            try:
                captures_df = pd.read_excel(import_path, sheet_name='Captures')
                data['captures'] = captures_df.to_dict('records')
            except Exception:
                pass  # Captures sheet might not exist
            
            # Read artifacts sheet
            try:
                artifacts_df = pd.read_excel(import_path, sheet_name='Artifacts')
                data['artifacts'] = artifacts_df.to_dict('records')
            except Exception:
                pass  # Artifacts sheet might not exist
            
            return data
            
        except Exception as e:
            raise DataExportImportError(f"Failed to load Excel data: {e}")
    
    def _validate_import_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imported data structure."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required data
        if not data.get('captures') and not data.get('artifacts'):
            validation_result['errors'].append("No captures or artifacts found in data")
            validation_result['valid'] = False
        
        # Validate capture data
        if data.get('captures'):
            for i, capture in enumerate(data['captaces'][:5]):  # Validate first 5
                if 'created_at' not in capture:
                    validation_result['warnings'].append(f"Capture {i+1} missing created_at")
        
        # Validate artifact data
        if data.get('artifacts'):
            for i, artifact in enumerate(data['artifacts'][:5]):  # Validate first 5
                if 'kind' not in artifact:
                    validation_result['errors'].append(f"Artifact {i+1} missing kind field")
                    validation_result['valid'] = False
        
        return validation_result
    
    def _import_data_to_database(self, data: Dict[str, Any], merge_mode: bool) -> Dict[str, int]:
        """Import data to database."""
        stats = {'captures': 0, 'artifacts': 0}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = OFF")  # Disable for faster import
                conn.execute("BEGIN TRANSACTION")
                
                # Import captures
                if data.get('captures'):
                    for capture in data['captaces']:
                        try:
                            conn.execute("""
                                INSERT INTO captures (created_at, source_window, screenshot_path, clipboard_text, notes)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                capture.get('created_at'),
                                capture.get('source_window'),
                                capture.get('screenshot_path'),
                                capture.get('clipboard_text'),
                                capture.get('notes')
                            ))
                            stats['captures'] += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to import capture: {e}")
                
                # Import artifacts
                if data.get('artifacts'):
                    for artifact in data['artifacts']:
                        try:
                            # Map capture_id if available
                            capture_id = None
                            if artifact.get('capture_id'):
                                capture_id = int(artifact['capture_id'])
                            
                            conn.execute("""
                                INSERT INTO artifacts (capture_id, kind, url, title, text, metadata_json)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                capture_id,
                                artifact.get('kind'),
                                artifact.get('url'),
                                artifact.get('title'),
                                artifact.get('text'),
                                json.dumps(artifact.get('metadata')) if artifact.get('metadata') else None
                            ))
                            stats['artifacts'] += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to import artifact: {e}")
                
                conn.execute("COMMIT")
                conn.execute("PRAGMA foreign_keys = ON")
                
        except Exception as e:
            self.logger.error(f"Database import failed: {e}")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("ROLLBACK")
            except:
                pass
            raise DataExportImportError(f"Database import failed: {e}")
        
        return stats
    
    def _verify_import(self, import_stats: Dict[str, int]) -> Dict[str, Any]:
        """Verify import success."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current counts
                cursor = conn.execute("SELECT COUNT(*) FROM captures")
                current_captures = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM artifacts")
                current_artifacts = cursor.fetchone()[0]
                
                return {
                    'current_captures': current_captures,
                    'current_artifacts': current_artifacts,
                    'imported_captures': import_stats['captures'],
                    'imported_artifacts': import_stats['artifacts']
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _create_backup(self) -> Dict[str, Any]:
        """Create backup of current database."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_contextbox_{timestamp}.db"
            
            with sqlite3.connect(self.db_path) as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            backup_size = Path(backup_path).stat().st_size / (1024 * 1024)
            
            return {
                'success': True,
                'backup_path': backup_path,
                'backup_size_mb': backup_size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def import_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """Import from a database backup file."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise DataExportImportError(f"Backup file not found: {backup_path}")
        
        try:
            # Backup current database first
            current_backup = self._create_backup()
            
            # Replace current database with backup
            shutil.copy2(backup_path, self.db_path)
            
            return {
                'success': True,
                'backup_restored': str(backup_path),
                'previous_backup': current_backup
            }
            
        except Exception as e:
            raise DataExportImportError(f"Backup restore failed: {e}")


class ContextBoxDataManager:
    """
    Main data management system combining export and import capabilities.
    """
    
    def __init__(self, db_path: str = 'contextbox.db'):
        """Initialize data manager."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.exporter = DataExporter(db_path)
        self.importer = DataImporter(db_path)
        
        # Export/import directories
        self.export_dir = Path('./exports')
        self.import_dir = Path('./imports')
        self.backup_dir = Path('./backups')
        
        # Create directories
        for directory in [self.export_dir, self.import_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Data manager initialized")
    
    def export_data(self, 
                   export_name: str,
                   formats: Optional[List[str]] = None,
                   include_artifacts: bool = True,
                   compress: bool = True) -> Dict[str, Any]:
        """
        Export data with automatic directory management.
        
        Args:
            export_name: Name for the export
            formats: List of formats to export
            include_artifacts: Whether to include artifacts
            compress: Whether to compress the export
            
        Returns:
            Export results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"{export_name}_{timestamp}"
        
        return self.exporter.export_all_data(
            export_path=str(export_path),
            formats=formats,
            include_artifacts=include_artifacts,
            compression=compress
        )
    
    def import_data(self, 
                   import_path: str,
                   format_type: Optional[str] = None,
                   backup_first: bool = True) -> Dict[str, Any]:
        """
        Import data with automatic backup.
        
        Args:
            import_path: Path to import data
            format_type: Data format
            backup_first: Whether to backup before import
            
        Returns:
            Import results
        """
        return self.importer.import_data(
            import_path=import_path,
            format_type=format_type,
            backup_before_import=backup_first
        )
    
    def create_backup(self) -> Dict[str, Any]:
        """Create a database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"contextbox_backup_{timestamp}.db"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            backup_size = backup_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(f"Backup created: {backup_path}")
            return {
                'success': True,
                'backup_path': str(backup_path),
                'backup_size_mb': backup_size,
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': timestamp
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("contextbox_backup_*.db"):
            try:
                stat = backup_file.stat()
                backup_info = {
                    'path': str(backup_file),
                    'name': backup_file.name,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                backups.append(backup_info)
            except Exception as e:
                self.logger.warning(f"Failed to get backup info for {backup_file}: {e}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backups
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore from a backup file."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise DataExportImportError(f"Backup file not found: {backup_path}")
        
        try:
            # Create backup of current state
            current_backup = self.create_backup()
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            self.logger.info(f"Database restored from backup: {backup_path}")
            return {
                'success': True,
                'restored_from': str(backup_path),
                'previous_state_backup': current_backup
            }
            
        except Exception as e:
            self.logger.error(f"Backup restore failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'restored_from': str(backup_path)
            }
    
    def cleanup_old_backups(self, keep_count: int = 10) -> Dict[str, Any]:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return {
                'cleaned': 0,
                'message': f'Only {len(backups)} backups exist, nothing to clean'
            }
        
        backups_to_delete = backups[keep_count:]
        deleted_count = 0
        total_size_freed = 0
        
        for backup in backups_to_delete:
            try:
                backup_path = Path(backup['path'])
                file_size = backup['size_mb']
                backup_path.unlink()
                deleted_count += 1
                total_size_freed += file_size
                self.logger.info(f"Deleted old backup: {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete backup {backup['path']}: {e}")
        
        return {
            'cleaned': deleted_count,
            'size_freed_mb': total_size_freed,
            'remaining_backups': len(backups) - deleted_count
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM captures) as total_captures,
                        (SELECT COUNT(*) FROM artifacts) as total_artifacts,
                        (SELECT COUNT(DISTINCT capture_id) FROM artifacts) as captures_with_artifacts,
                        (SELECT MAX(created_at) FROM captures) as latest_capture,
                        (SELECT MIN(created_at) FROM captures) as earliest_capture
                """)
                
                row = cursor.fetchone()
                if row:
                    stats = {
                        'total_captures': row[0],
                        'total_artifacts': row[1],
                        'captures_with_artifacts': row[2],
                        'latest_capture': row[3],
                        'earliest_capture': row[4]
                    }
                else:
                    stats = {'total_captures': 0, 'total_artifacts': 0}
                
                # Get database file size
                db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
                stats['database_size_mb'] = db_size_mb
                
                # Get backup information
                backups = self.list_backups()
                stats['available_backups'] = len(backups)
                if backups:
                    stats['latest_backup'] = backups[0]['created_at']
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get data statistics: {e}")
            return {'error': str(e)}


def create_data_manager(db_path: str = 'contextbox.db') -> ContextBoxDataManager:
    """
    Factory function to create data manager.
    
    Args:
        db_path: Path to ContextBox database
        
    Returns:
        ContextBoxDataManager instance
    """
    return ContextBoxDataManager(db_path)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data manager
    data_manager = create_data_manager('contextbox.db')
    
    # Export data
    print("Exporting data...")
    export_result = data_manager.export_data(
        export_name="test_export",
        formats=['json', 'csv'],
        compress=True
    )
    print(f"Export completed: {export_result['success']}")
    print(f"Files created: {export_result['total_files']}")
    
    # Get statistics
    print("\nData statistics:")
    stats = data_manager.get_data_statistics()
    print(json.dumps(stats, indent=2, default=str))
    
    # Create backup
    print("\nCreating backup...")
    backup_result = data_manager.create_backup()
    print(f"Backup created: {backup_result['success']}")
    
    # List backups
    print("\nAvailable backups:")
    backups = data_manager.list_backups()
    for backup in backups[:3]:  # Show first 3
        print(f"- {backup['name']}: {backup['size_mb']:.1f} MB")
    
    print("\nData management system test completed!")