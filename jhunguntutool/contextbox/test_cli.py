#!/usr/bin/env python3
"""
Test script for ContextBox CLI capture functionality
"""

import sys
import os
import json

# Add contextbox to path
sys.path.insert(0, '/workspace/contextbox')

def test_contextbox_imports():
    """Test that all ContextBox components can be imported."""
    print("Testing ContextBox imports...")
    
    try:
        from contextbox.main import ContextBox
        from contextbox.cli import create_parser, capture_command
        from contextbox.utils import setup_logging, get_platform_info
        from contextbox.database import ContextDatabase
        from contextbox.extractors import ContextExtractor
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_capture_command():
    """Test the capture command functionality."""
    print("\nTesting capture command...")
    
    try:
        from contextbox.main import ContextBox
        from contextbox.cli import capture_command
        import argparse
        
        # Create a mock args object
        class MockArgs:
            def __init__(self):
                self.output = '/tmp/capture_test.json'
                self.artifact_dir = '/tmp/contextbox_artifacts'
                self.no_screenshot = True  # Skip screenshot for test
                self.extract_text = True
                self.extract_urls = True
        
        # Initialize ContextBox
        config = {'log_level': 'INFO'}
        app = ContextBox(config)
        
        # Test capture command
        args = MockArgs()
        capture_command(args, app)
        
        # Check if output was created
        if os.path.exists(args.output):
            print("✅ Capture command executed successfully")
            with open(args.output, 'r') as f:
                result = json.load(f)
            print(f"   Generated context ID: {result.get('context_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            return True
        else:
            print("❌ Capture command did not generate output file")
            return False
            
    except Exception as e:
        print(f"❌ Capture command failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_integration():
    """Test database integration."""
    print("\nTesting database integration...")
    
    try:
        from contextbox.database import ContextDatabase
        
        # Initialize database with temporary path
        config = {'path': '/tmp/test_contextbox.db'}
        db = ContextDatabase(config)
        
        # Test storing and retrieving data
        test_data = {
            'test': True,
            'message': 'Database integration test',
            'timestamp': '2023-01-01T00:00:00'
        }
        
        context_id = db.store(test_data)
        retrieved = db.retrieve(context_id)
        
        if retrieved and retrieved.get('test') == True:
            print("✅ Database integration working")
            return True
        else:
            print("❌ Database integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_extraction():
    """Test text extraction functionality."""
    print("\nTesting extraction functionality...")
    
    try:
        from contextbox.extractors import ContextExtractor
        
        config = {}
        extractor = ContextExtractor(config)
        
        test_data = {
            'text': 'Visit https://example.com for more info',
            'window_title': 'Test Window - Example Application'
        }
        
        result = extractor.extract(test_data)
        
        if result and 'extractions' in result:
            print("✅ Extraction functionality working")
            return True
        else:
            print("❌ Extraction functionality failed")
            return False
            
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        return False

def test_platform_info():
    """Test platform information gathering."""
    print("\nTesting platform information...")
    
    try:
        from contextbox.utils import get_platform_info
        
        info = get_platform_info()
        
        if info and 'system' in info:
            print("✅ Platform info gathering working")
            print(f"   Platform: {info.get('system', 'Unknown')}")
            return True
        else:
            print("❌ Platform info gathering failed")
            return False
            
    except Exception as e:
        print(f"❌ Platform info test failed: {e}")
        return False

def main():
    """Main test function."""
    print("ContextBox CLI Test Suite")
    print("=" * 50)
    
    tests = [
        test_contextbox_imports,
        test_database_integration,
        test_extraction,
        test_platform_info,
        test_capture_command
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"  {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLI is ready to use.")
        print("\nUsage: python contextbox/cli.py capture")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
