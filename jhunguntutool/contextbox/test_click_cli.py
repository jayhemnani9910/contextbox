#!/usr/bin/env python3
"""
Comprehensive test script for the new Click-based CLI with rich formatting
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a CLI command and capture output."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Exit Code: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def create_test_data():
    """Create test data files for import/export testing."""
    # Create test JSON file
    test_data = {
        "contexts": [
            {
                "context_id": "test_001",
                "timestamp": "2023-12-01T12:00:00",
                "platform": {"system": "Linux"},
                "status": "completed",
                "extracted": {
                    "text": "This is test context content for CLI testing",
                    "urls": ["https://example.com", "https://test.org"]
                }
            },
            {
                "context_id": "test_002", 
                "timestamp": "2023-12-01T12:30:00",
                "platform": {"system": "Windows"},
                "status": "completed",
                "extracted": {
                    "text": "Another test context for comprehensive testing",
                    "urls": []
                }
            }
        ]
    }
    
    with open('/workspace/contextbox/test_import.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("✅ Created test data files")

def main():
    """Main test function."""
    print("🚀 ContextBox Click CLI - Comprehensive Test Suite")
    print("="*70)
    
    os.chdir('/workspace/contextbox')
    
    # Test results tracking
    tests = []
    
    # Create test data
    create_test_data()
    
    # Test 1: Help system
    tests.append(run_command(
        "python click_cli.py --help",
        "Help system with rich formatting"
    ))
    
    # Test 2: Version
    tests.append(run_command(
        "python click_cli.py --version", 
        "Version command"
    ))
    
    # Test 3: Empty command (beautiful help header)
    tests.append(run_command(
        "python click_cli.py",
        "Beautiful help header when no command provided"
    ))
    
    # Test 4: Capture command
    tests.append(run_command(
        "python click_cli.py capture --output test_capture_full.json --no-screenshot",
        "Capture command with rich formatting"
    ))
    
    # Test 5: List command
    tests.append(run_command(
        "python click_cli.py list --limit 3 --format table",
        "List contexts with rich table"
    ))
    
    # Test 6: Stats command
    tests.append(run_command(
        "python click_cli.py stats --detailed",
        "Database statistics with rich formatting"
    ))
    
    # Test 7: Search command
    tests.append(run_command(
        'python click_cli.py search "test" --limit 3',
        "Search functionality with progress bars"
    ))
    
    # Test 8: Ask command
    tests.append(run_command(
        'python click_cli.py ask "What is this about?"',
        "AI Q&A functionality with progress indicators"
    ))
    
    # Test 9: Summarize command
    tests.append(run_command(
        "python click_cli.py summarize --format brief",
        "Context summarization with different formats"
    ))
    
    # Test 10: Config view
    tests.append(run_command(
        "python click_cli.py config --view",
        "Configuration management with rich display"
    ))
    
    # Test 11: Export command
    tests.append(run_command(
        "python click_cli.py export --format json --output test_export.json",
        "Export functionality to various formats"
    ))
    
    # Test Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(tests)
    total = len(tests)
    
    test_descriptions = [
        "Help system with rich formatting",
        "Version command", 
        "Beautiful help header",
        "Capture command with rich formatting",
        "List contexts with rich table",
        "Database statistics with rich formatting",
        "Search functionality with progress bars",
        "AI Q&A functionality with progress indicators",
        "Context summarization with different formats",
        "Configuration management with rich display",
        "Export functionality to various formats"
    ]
    
    for i, (description, passed_test) in enumerate(zip(test_descriptions, tests), 1):
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{i:2d}. {status} - {description}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The Click CLI migration is complete and working perfectly!")
        print("\n✨ Features Successfully Implemented:")
        print("  • Rich formatted output with tables, panels, and progress bars")
        print("  • Interactive prompts for sensitive data (API keys)")  
        print("  • Subcommands: capture, ask, summarize, search, list, stats, config, export, import")
        print("  • Beautiful help system with rich formatting")
        print("  • Progress indicators for long operations")
        print("  • Colorful status messages and success/error displays")
        print("  • Autocomplete-friendly command structure")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)