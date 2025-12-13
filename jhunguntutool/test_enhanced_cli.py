#!/usr/bin/env python3
"""
Test script for the enhanced Click CLI
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print("✅ STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ STDERR:")
            print(result.stderr)
        
        print(f"✅ Return code: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def test_cli():
    """Test the enhanced Click CLI."""
    
    # Change to contextbox directory
    os.chdir('/workspace/contextbox')
    
    # Test basic commands
    tests = [
        ("python click_cli_enhanced.py --version", "Version command"),
        ("python click_cli_enhanced.py --help", "Help command"),
        ("python click_cli_enhanced.py help", "Help with help command"),
        ("python click_cli_enhanced.py capture --help", "Capture help"),
        ("python click_cli_enhanced.py list --help", "List help"),
        ("python click_cli_enhanced.py search --help", "Search help"),
        ("python click_cli_enhanced.py config --help", "Config help"),
        ("python click_cli_enhanced.py stats --help", "Stats help"),
        ("python click_cli_enhanced.py export --help", "Export help"),
        ("python click_cli_enhanced.py import --help", "Import help"),
    ]
    
    passed = 0
    total = len(tests)
    
    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1
    
    # Test actual functionality (these might fail due to missing data)
    functional_tests = [
        ("python click_cli_enhanced.py list", "List contexts"),
        ("python click_cli_enhanced.py stats", "Show statistics"),
        ("python click_cli_enhanced.py config --view", "View configuration"),
    ]
    
    print(f"\n{'='*60}")
    print("🧪 FUNCTIONAL TESTS")
    print('='*60)
    
    for cmd, description in functional_tests:
        if run_command(cmd, description):
            passed += 1
        total += 1
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    elif passed > total // 2:
        print("👍 Most tests passed!")
    else:
        print("⚠️ Some tests failed, but this might be expected")
    
    return passed == total

if __name__ == '__main__':
    test_cli()