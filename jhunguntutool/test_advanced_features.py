"""
Advanced Features Integration Test for ContextBox

This module tests all advanced features together to ensure they integrate properly:
- Notification system with desktop notifications and system tray
- Privacy mode with encryption and PII redaction
- Semantic search using sentence-transformers
- Performance optimizations with indexing and caching
- Data export/import system
"""

import json
import time
import tempfile
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import all advanced features
import sys
sys.path.append('/workspace/contextbox')

# Import advanced feature modules
import notification_system
import privacy_mode
import semantic_search
import performance_optimizations
import data_export_import

from notification_system import NotificationSystem
from privacy_mode import PrivacyMode, PIIRedactor
from semantic_search import SemanticSearchEngine
from performance_optimizations import ContextBoxPerformanceManager, performance_monitor
from data_export_import import ContextBoxDataManager

# Import ContextBox core components
from contextbox.database import ContextDatabase


class AdvancedFeaturesTester:
    """
    Comprehensive test suite for ContextBox advanced features.
    """
    
    def __init__(self, test_db_path: str = 'test_contextbox.db'):
        """Initialize the test suite."""
        self.test_db_path = test_db_path
        self.logger = logging.getLogger(__name__)
        
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp(prefix='contextbox_test_'))
        
        # Test configuration
        self.test_config = {
            'database': {
                'db_path': str(self.test_dir / test_db_path)
            },
            'notification': {
                'enable_desktop': False,  # Disable for testing
                'enable_tray': False,     # Disable for testing
                'notification_timeout': 1000,
                'enable_sounds': False
            },
            'privacy': {
                'enable_encryption': True,
                'enable_redaction': True,
                'master_password': 'test_password_123',
                'auto_protect': True
            },
            'semantic_search': {
                'cache_dir': str(self.test_dir / 'semantic_cache'),
                'auto_indexing': False,  # Disable for testing
                'similarity_threshold': 0.6
            },
            'performance': {
                'cache_dir': str(self.test_dir / 'perf_cache'),
                'cache_size_mb': 50
            },
            'data_export': {}
        }
        
        # Initialize components
        self.database = None
        self.notification_system = None
        self.privacy_mode = None
        self.semantic_search = None
        self.performance_manager = None
        self.data_manager = None
        
        # Test results
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        self.logger.info(f"Advanced features test suite initialized with test directory: {self.test_dir}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all advanced features tests."""
        print("\n" + "="*60)
        print("CONTEXTBOX ADVANCED FEATURES INTEGRATION TEST")
        print("="*60)
        
        try:
            # Initialize all components
            self._initialize_components()
            
            # Run test suites
            self._test_database_operations()
            self._test_notification_system()
            self._test_privacy_mode()
            self._test_semantic_search()
            self._test_performance_optimizations()
            self._test_data_export_import()
            self._test_integration_scenarios()
            
            # Print summary
            self._print_test_summary()
            
        except Exception as e:
            print(f"\nCritical test failure: {e}")
            self.logger.error(f"Critical test failure: {e}")
        
        finally:
            # Cleanup
            self._cleanup_test_environment()
        
        return self.test_results
    
    def _initialize_components(self) -> None:
        """Initialize all ContextBox components."""
        print("\n1. Initializing Components...")
        
        # Database
        self.database = ContextDatabase(self.test_config['database'])
        self._log_test("Database initialization", True)
        
        # Notification System
        try:
            self.notification_system = NotificationSystem(self.test_config['notification'])
            self.notification_system.start_notification_system()
            self._log_test("Notification system initialization", True)
        except Exception as e:
            self._log_test("Notification system initialization", False, str(e))
        
        # Privacy Mode
        try:
            self.privacy_mode = PrivacyMode(self.test_config['privacy'])
            self._log_test("Privacy mode initialization", True)
        except Exception as e:
            self._log_test("Privacy mode initialization", False, str(e))
        
        # Semantic Search
        try:
            self.semantic_search = SemanticSearchEngine(
                self.test_config['database']['db_path'], 
                self.test_config['semantic_search']
            )
            self._log_test("Semantic search initialization", True)
        except Exception as e:
            self._log_test("Semantic search initialization", False, str(e))
        
        # Performance Manager
        try:
            self.performance_manager = ContextBoxPerformanceManager(
                self.test_config['database']['db_path'],
                self.test_config['performance']
            )
            self._log_test("Performance manager initialization", True)
        except Exception as e:
            self._log_test("Performance manager initialization", False, str(e))
        
        # Data Manager
        try:
            self.data_manager = ContextBoxDataManager(self.test_config['database']['db_path'])
            self._log_test("Data manager initialization", True)
        except Exception as e:
            self._log_test("Data manager initialization", False, str(e))
    
    def _test_database_operations(self) -> None:
        """Test database operations with advanced features."""
        print("\n2. Testing Database Operations...")
        
        try:
            # Create test capture
            capture_id = self.database.create_capture(
                source_window="Test Window",
                screenshot_path="/path/to/test.png",
                clipboard_text="Test clipboard content with URL: https://example.com",
                notes="Test notes with email: test@example.com"
            )
            self._log_test("Database capture creation", True)
            
            # Create test artifacts
            artifact1_id = self.database.create_artifact(
                capture_id=capture_id,
                kind='url',
                url='https://example.com',
                title='Test URL',
                text='Example website content'
            )
            
            artifact2_id = self.database.create_artifact(
                capture_id=capture_id,
                kind='text',
                text='Important phone number: (555) 123-4567'
            )
            
            self._log_test("Database artifact creation", True)
            
            # Test database statistics
            stats = self.database.get_stats()
            self._log_test("Database statistics retrieval", True, f"Stats: {stats}")
            
        except Exception as e:
            self._log_test("Database operations", False, str(e))
    
    def _test_notification_system(self) -> None:
        """Test notification system functionality."""
        print("\n3. Testing Notification System...")
        
        if not self.notification_system:
            self._log_test("Notification system", False, "Not initialized")
            return
        
        try:
            # Test basic notifications
            self.notification_system.notify(
                "Test Notification", 
                "This is a test notification", 
                "info"
            )
            self._log_test("Basic notification", True)
            
            # Test capture success notification
            self.notification_system.notify_capture_success({
                'source_window': 'Test Window',
                'artifact_count': 2
            })
            self._log_test("Capture success notification", True)
            
            # Test error notification
            self.notification_system.notify_error("Test error", "Test error details")
            self._log_test("Error notification", True)
            
            # Test notification statistics
            notification_stats = self.notification_system.get_statistics()
            self._log_test("Notification statistics", True, f"Stats: {notification_stats}")
            
        except Exception as e:
            self._log_test("Notification system", False, str(e))
    
    def _test_privacy_mode(self) -> None:
        """Test privacy mode functionality."""
        print("\n4. Testing Privacy Mode...")
        
        if not self.privacy_mode:
            self._log_test("Privacy mode", False, "Not initialized")
            return
        
        try:
            # Test PII detection
            test_text = "Contact me at john.doe@company.com or call (555) 123-4567"
            pii_items = self.privacy_mode.redactor.detect_pii(test_text)
            self._log_test("PII detection", len(pii_items) > 0, f"Found {len(pii_items)} PII items")
            
            # Test PII redaction
            redacted_text, report = self.privacy_mode.redactor.redact_pii(test_text)
            self._log_test("PII redaction", '[REDACTED]' in redacted_text)
            
            # Test encryption
            original_text = "Secret information"
            encrypted_text = self.privacy_mode.encryption.encrypt_text(original_text)
            decrypted_text = self.privacy_mode.encryption.decrypt_text(encrypted_text)
            self._log_test("Text encryption/decryption", original_text == decrypted_text)
            
            # Test capture protection
            capture_data = {
                'source_window': 'Email - user@domain.com',
                'clipboard_text': test_text,
                'notes': 'Meeting scheduled for 12/25/2023'
            }
            
            protected_data = self.privacy_mode.protect_capture(capture_data)
            self._log_test("Capture protection", 'privacy_protection' in protected_data)
            
            # Test sensitivity analysis
            analysis = self.privacy_mode.analyze_data_sensitivity(capture_data)
            self._log_test("Data sensitivity analysis", analysis['sensitivity_score'] >= 0)
            
        except Exception as e:
            self._log_test("Privacy mode", False, str(e))
    
    def _test_semantic_search(self) -> None:
        """Test semantic search functionality."""
        print("\n5. Testing Semantic Search...")
        
        if not self.semantic_search:
            self._log_test("Semantic search", False, "Not initialized")
            return
        
        try:
            # Test single artifact indexing
            artifact_data = {
                'id': 1,
                'title': 'Contact Information',
                'text': 'Email addresses and phone numbers for business contacts',
                'url': 'https://contacts.example.com',
                'kind': 'url'
            }
            
            indexed = self.semantic_search.index_artifact(artifact_data)
            self._log_test("Artifact indexing", indexed)
            
            # Test search functionality
            search_results = self.semantic_search.search(
                query="email contact information",
                max_results=10
            )
            
            self._log_test("Semantic search", 'results' in search_results)
            
            # Test search statistics
            search_stats = self.semantic_search.get_statistics()
            self._log_test("Search statistics", isinstance(search_stats, dict))
            
        except Exception as e:
            self._log_test("Semantic search", False, str(e))
    
    def _test_performance_optimizations(self) -> None:
        """Test performance optimization functionality."""
        print("\n6. Testing Performance Optimizations...")
        
        if not self.performance_manager:
            self._log_test("Performance manager", False, "Not initialized")
            return
        
        try:
            # Test database optimization
            optimization_results = self.performance_manager.optimize_database()
            self._log_test("Database optimization", optimization_results['success'])
            
            # Test performance report
            perf_report = self.performance_manager.get_performance_report()
            self._log_test("Performance report", isinstance(perf_report, dict))
            
            # Test query optimization
            query = "SELECT * FROM captures LIMIT 10"
            optimization_analysis = self.performance_manager.query_optimizer.optimize_query(query)
            self._log_test("Query optimization", 'analysis' in optimization_analysis)
            
            # Test monitored query execution
            results, metrics = self.performance_manager.execute_optimized_query(
                query, use_cache=True, cache_ttl=60
            )
            self._log_test("Monitored query execution", isinstance(results, list))
            
        except Exception as e:
            self._log_test("Performance optimizations", False, str(e))
    
    def _test_data_export_import(self) -> None:
        """Test data export and import functionality."""
        print("\n7. Testing Data Export/Import...")
        
        if not self.data_manager:
            self._log_test("Data manager", False, "Not initialized")
            return
        
        try:
            # Test data statistics
            data_stats = self.data_manager.get_data_statistics()
            self._log_test("Data statistics", isinstance(data_stats, dict))
            
            # Test data export
            export_result = self.data_manager.export_data(
                export_name="test_export",
                formats=['json'],
                compress=False
            )
            self._log_test("Data export", export_result['success'])
            
            # Test backup creation
            backup_result = self.data_manager.create_backup()
            self._log_test("Backup creation", backup_result['success'])
            
            # Test backup listing
            backups = self.data_manager.list_backups()
            self._log_test("Backup listing", len(backups) > 0)
            
        except Exception as e:
            self._log_test("Data export/import", False, str(e))
    
    def _test_integration_scenarios(self) -> None:
        """Test integration scenarios between features."""
        print("\n8. Testing Integration Scenarios...")
        
        try:
            # Scenario 1: Privacy-protected capture with notification
            capture_data = {
                'source_window': 'Banking Website',
                'clipboard_text': 'Account balance: $1,234.56, SSN: 123-45-6789',
                'notes': 'Financial information'
            }
            
            # Apply privacy protection
            protected_data = self.privacy_mode.protect_capture(capture_data)
            
            # Store in database
            protected_capture_id = self.database.create_capture(
                source_window=protected_data['source_window'],
                clipboard_text=protected_data['clipboard_text'],
                notes=protected_data['notes']
            )
            
            # Send notification
            if self.notification_system:
                self.notification_system.notify_capture_success({
                    'source_window': 'Privacy-protected capture',
                    'artifact_count': 0
                })
            
            self._log_test("Privacy-protected capture with notification", True)
            
            # Scenario 2: Semantic search with performance monitoring
            if self.semantic_search:
                # Create test artifact
                test_artifact = {
                    'id': 2,
                    'title': 'Project Documentation',
                    'text': 'Technical specifications and requirements for the new system',
                    'kind': 'text'
                }
                
                self.semantic_search.index_artifact(test_artifact)
                
                # Perform search with performance monitoring
                start_time = time.time()
                search_results = self.semantic_search.search("technical specifications")
                search_time = time.time() - start_time
                
                self._log_test("Semantic search with performance monitoring", search_time < 5.0)
            
            # Scenario 3: Complete data workflow
            if self.data_manager:
                # Export all data with performance tracking
                export_start = time.time()
                export_result = self.data_manager.export_data(
                    export_name="integration_test",
                    formats=['json'],
                    compress=True
                )
                export_time = time.time() - export_start
                
                self._log_test("Complete data workflow", export_result['success'] and export_time < 10.0)
            
        except Exception as e:
            self._log_test("Integration scenarios", False, str(e))
    
    def _log_test(self, test_name: str, success: bool, details: str = "") -> None:
        """Log test result."""
        self.test_results['total_tests'] += 1
        
        if success:
            self.test_results['passed_tests'] += 1
            print(f"✓ {test_name}")
        else:
            self.test_results['failed_tests'] += 1
            print(f"✗ {test_name}")
            if details:
                print(f"  Error: {details}")
        
        # Store test details
        self.test_results['test_details'].append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def _print_test_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        print(f"Total Tests: {self.test_results['total_tests']}")
        print(f"Passed: {self.test_results['passed_tests']}")
        print(f"Failed: {self.test_results['failed_tests']}")
        
        if self.test_results['total_tests'] > 0:
            success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Feature-specific summary
        print("\nFeature Status:")
        features_tested = {}
        for test_detail in self.test_results['test_details']:
            feature_name = test_detail['test_name'].split(' ')[0]
            if feature_name not in features_tested:
                features_tested[feature_name] = {'passed': 0, 'total': 0}
            features_tested[feature_name]['total'] += 1
            if test_detail['success']:
                features_tested[feature_name]['passed'] += 1
        
        for feature, stats in features_tested.items():
            status = "✓ PASSED" if stats['passed'] == stats['total'] else "✗ FAILED"
            print(f"  {feature}: {stats['passed']}/{stats['total']} {status}")
        
        # Integration summary
        integration_tests = [t for t in self.test_results['test_details'] if 'Integration' in t['test_name']]
        if integration_tests:
            integration_passed = sum(1 for t in integration_tests if t['success'])
            print(f"\nIntegration Tests: {integration_passed}/{len(integration_tests)} passed")
        
        # Final verdict
        print("\n" + "="*60)
        if self.test_results['failed_tests'] == 0:
            print("🎉 ALL TESTS PASSED! Advanced features are working correctly.")
        else:
            print(f"⚠️  {self.test_results['failed_tests']} tests failed. Check details above.")
        print("="*60)
        
        # Save detailed results
        results_file = self.test_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed test results saved to: {results_file}")
    
    def _cleanup_test_environment(self) -> None:
        """Clean up test environment."""
        try:
            # Stop notification system
            if self.notification_system:
                self.notification_system.stop_notification_system()
            
            # Stop performance monitoring
            if self.performance_manager:
                self.performance_manager.stop_performance_monitoring()
            
            # Close database connections
            if self.database:
                self.database.cleanup()
            
            # Clean up test directory
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                print(f"\nTest directory cleaned up: {self.test_dir}")
            
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed test results."""
        return self.test_results


def run_advanced_features_test() -> Dict[str, Any]:
    """
    Main function to run advanced features test.
    
    Returns:
        Test results dictionary
    """
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive test
    tester = AdvancedFeaturesTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    # Run the test suite
    results = run_advanced_features_test()
    
    # Exit with appropriate code
    if results['failed_tests'] == 0:
        exit(0)  # Success
    else:
        exit(1)  # Failure