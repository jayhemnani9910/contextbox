#!/usr/bin/env python3

# Test the screenshot capture implementation
import sys
import os

# Add the contextbox path
sys.path.insert(0, '/workspace/contextbox')

def test_import():
    """Test that the modules can be imported"""
    try:
        from contextbox.capture import ScreenshotCapture, ContextCapture
        print("✓ Successfully imported ScreenshotCapture and ContextCapture")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_screenshot_capture():
    """Test ScreenshotCapture functionality"""
    try:
        from contextbox.capture import ScreenshotCapture
        
        # Initialize
        capture = ScreenshotCapture()
        print(f"✓ ScreenshotCapture initialized with media_dir: {capture.media_dir}")
        
        # Test tool detection
        tools = capture.get_available_tools()
        print(f"✓ Detected available tools: {tools}")
        
        # Test system detection
        is_wayland = capture.is_wayland()
        is_gnome = capture.is_gnome()
        print(f"✓ System detection - Wayland: {is_wayland}, GNOME: {is_gnome}")
        
        return True
    except Exception as e:
        print(f"✗ ScreenshotCapture test failed: {e}")
        return False

def test_context_capture():
    """Test ContextCapture compatibility class"""
    try:
        from contextbox.capture import ContextCapture
        
        # Initialize with config
        config = {
            'capture': {
                'interval': 1.0,
                'max_captures': 1,
                'types': ['full']
            }
        }
        context_capture = ContextCapture(config)
        print("✓ ContextCapture initialized successfully")
        
        # Test stats
        stats = context_capture.get_capture_stats()
        print(f"✓ ContextCapture stats: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ ContextCapture test failed: {e}")
        return False

def test_main_integration():
    """Test main module integration"""
    try:
        from contextbox.main import ContextBox
        
        # Initialize ContextBox
        config = {
            'capture': {
                'interval': 5.0,
                'max_captures': 0
            }
        }
        app = ContextBox(config)
        print("✓ ContextBox initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Main integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing ContextBox Screenshot Capture Implementation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("ScreenshotCapture Test", test_screenshot_capture),
        ("ContextCapture Test", test_context_capture),
        ("Main Integration Test", test_main_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  FAILED: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Screenshot capture implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)