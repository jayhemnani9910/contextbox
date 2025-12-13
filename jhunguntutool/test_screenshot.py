#!/usr/bin/env python3
"""
Test script for ContextBox screenshot capture functionality
"""

import sys
import os
sys.path.insert(0, '/workspace/contextbox')

from contextbox.capture import ScreenshotCapture

def test_screenshot_functionality():
    """Test screenshot capture functionality"""
    print("Testing ScreenshotCapture functionality...")
    
    # Initialize screenshot capture
    capture = ScreenshotCapture()
    print(f"Media directory: {capture.media_dir}")
    
    # Test tool detection
    tools = capture.get_available_tools()
    print(f"Available screenshot tools: {tools}")
    
    # Test system detection
    print(f"Running on Wayland: {capture.is_wayland()}")
    print(f"Running on GNOME: {capture.is_gnome()}")
    
    # Test clipboard reading (may fail if no clipboard content)
    clipboard = capture.read_clipboard_content()
    print(f"Clipboard content: {clipboard}")
    
    # Test compatibility ContextCapture class
    from contextbox.capture import ContextCapture
    
    config = {
        'capture': {
            'interval': 1.0,
            'max_captures': 1,
            'types': ['full']
        }
    }
    
    context_capture = ContextCapture(config)
    print(f"ContextCapture stats: {context_capture.get_capture_stats()}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_screenshot_functionality()