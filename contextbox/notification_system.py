"""
Advanced Notification System for ContextBox

This module provides desktop notifications and system tray functionality
for the ContextBox application.
"""

import logging
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import queue
import tkinter as tk
from tkinter import messagebox

# Try to import notification libraries
try:
    import notify2
    NOTIFY2_AVAILABLE = True
except ImportError:
    NOTIFY2_AVAILABLE = False

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

try:
    import pystray
    from PIL import Image
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False


class NotificationError(Exception):
    """Custom exception for notification operations."""
    pass


class NotificationSystem:
    """
    Advanced notification system for ContextBox.
    
    Features:
    - Desktop notifications (notify2, plyer)
    - System tray icon with quick capture
    - Notification queue management
    - Sound notifications
    - Custom notification types
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize notification system.
        
        Args:
            config: Configuration dictionary
                - enable_desktop: Enable desktop notifications
                - enable_tray: Enable system tray icon
                - notification_timeout: Notification display timeout
                - enable_sounds: Enable notification sounds
                - max_notifications: Maximum notification history
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.enable_desktop = self.config.get('enable_desktop', True)
        self.enable_tray = self.config.get('enable_tray', True)
        self.notification_timeout = self.config.get('notification_timeout', 5000)
        self.enable_sounds = self.config.get('enable_sounds', True)
        self.max_notifications = self.config.get('max_notifications', 100)
        
        # Notification queue and history
        self.notification_queue = queue.Queue()
        self.notification_history: List[Dict[str, Any]] = []
        self.notification_count = 0
        
        # Callback functions
        self.quick_capture_callback: Optional[Callable] = None
        self.quit_callback: Optional[Callable] = None
        
        # System tray
        self.tray_icon = None
        self.tray_running = False
        
        # Threading
        self.notification_thread = None
        self.notification_thread_running = False
        
        # Initialize notification system
        self.notification_backend = None  # Always initialize
        self._initialize_desktop_notifications()
        self._initialize_tray_system()
        
        self.logger.info("Notification system initialized")
    
    def _initialize_desktop_notifications(self) -> None:
        """Initialize desktop notification system."""
        if not self.enable_desktop:
            self.logger.info("Desktop notifications disabled")
            return
        
        if NOTIFY2_AVAILABLE:
            try:
                notify2.init("ContextBox")
                self.notification_backend = "notify2"
                self.logger.info("Using notify2 for desktop notifications")
            except Exception as e:
                self.logger.warning(f"notify2 initialization failed: {e}")
                self.notification_backend = None
        elif PLYER_AVAILABLE:
            self.notification_backend = "plyer"
            self.logger.info("Using plyer for desktop notifications")
        else:
            self.notification_backend = None
            self.logger.warning("No desktop notification backend available")
    
    def _initialize_tray_system(self) -> None:
        """Initialize system tray icon."""
        if not self.enable_tray or not PYSTRAY_AVAILABLE:
            if self.enable_tray:
                self.logger.warning("System tray unavailable - pystray not installed")
            return
        
        try:
            self._create_tray_icon()
            self.logger.info("System tray icon initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize system tray: {e}")
    
    def _create_tray_icon(self) -> None:
        """Create system tray icon with menu."""
        # Create a simple icon (blue square)
        icon_size = (64, 64)
        icon_image = Image.new('RGB', icon_size, color='blue')
        
        # Define tray menu items
        menu = pystray.Menu(
            pystray.MenuItem("Quick Capture", self._quick_capture_action),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Notifications", self._show_notifications_action),
            pystray.MenuItem("Clear History", self._clear_history_action),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self._settings_action),
            pystray.MenuItem("Quit", self._quit_action)
        )
        
        # Create tray icon
        self.tray_icon = pystray.Icon(
            "ContextBox",
            icon_image,
            "ContextBox - Desktop Assistant",
            menu
        )
        
        # Store references to menu actions
        self.tray_icon._menu_actions = {
            'quick_capture': self._quick_capture_action,
            'show_notifications': self._show_notifications_action,
            'clear_history': self._clear_history_action,
            'settings': self._settings_action,
            'quit': self._quit_action
        }
    
    def _quick_capture_action(self, icon=None, item=None) -> None:
        """Handle quick capture tray menu action."""
        if self.quick_capture_callback:
            try:
                self.quick_capture_callback()
            except Exception as e:
                self.logger.error(f"Quick capture failed: {e}")
        else:
            self.logger.info("Quick capture requested (no callback configured)")
    
    def _show_notifications_action(self, icon=None, item=None) -> None:
        """Show notification history in a window."""
        try:
            self._show_notification_history_window()
        except Exception as e:
            self.logger.error(f"Failed to show notification history: {e}")
    
    def _clear_history_action(self, icon=None, item=None) -> None:
        """Clear notification history."""
        self.notification_history.clear()
        self.notification_count = 0
        self.logger.info("Notification history cleared")
    
    def _settings_action(self, icon=None, item=None) -> None:
        """Show notification settings window."""
        try:
            self._show_settings_window()
        except Exception as e:
            self.logger.error(f"Failed to show settings: {e}")
    
    def _quit_action(self, icon=None, item=None) -> None:
        """Handle quit action."""
        if self.quit_callback:
            try:
                self.quit_callback()
            except Exception as e:
                self.logger.error(f"Quit callback failed: {e}")
        self.stop_tray()
    
    def _show_notification_history_window(self) -> None:
        """Show notification history in a tkinter window."""
        root = tk.Tk()
        root.title("ContextBox - Notification History")
        root.geometry("600x400")
        
        # Create scrollable text widget
        import tkinter.scrolledtext as scrolledtext
        
        text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display notification history
        history_text = ""
        for notification in self.notification_history[-50:]:  # Show last 50
            timestamp = notification.get('timestamp', 'Unknown')
            title = notification.get('title', 'No Title')
            message = notification.get('message', '')
            notification_type = notification.get('type', 'info')
            
            history_text += f"[{timestamp}] [{notification_type.upper()}] {title}\n"
            history_text += f"{message}\n"
            history_text += "-" * 60 + "\n\n"
        
        text_widget.insert(tk.INSERT, history_text or "No notifications yet")
        text_widget.config(state=tk.DISABLED)
        
        root.mainloop()
    
    def _show_settings_window(self) -> None:
        """Show notification settings window."""
        root = tk.Tk()
        root.title("ContextBox - Notification Settings")
        root.geometry("400x300")
        
        # Settings variables
        desktop_var = tk.BooleanVar(value=self.enable_desktop)
        tray_var = tk.BooleanVar(value=self.enable_tray)
        sounds_var = tk.BooleanVar(value=self.enable_sounds)
        timeout_var = tk.IntVar(value=self.notification_timeout)
        
        # Create settings UI
        tk.Label(root, text="Notification Settings", font=("Arial", 14, "bold")).pack(pady=10)
        
        tk.Checkbutton(root, text="Desktop Notifications", variable=desktop_var).pack(anchor=tk.W, padx=20)
        tk.Checkbutton(root, text="System Tray", variable=tray_var).pack(anchor=tk.W, padx=20)
        tk.Checkbutton(root, text="Notification Sounds", variable=sounds_var).pack(anchor=tk.W, padx=20)
        
        tk.Label(root, text="Display Timeout (ms):").pack(anchor=tk.W, padx=20, pady=(20, 5))
        timeout_scale = tk.Scale(root, from_=1000, to=10000, orient=tk.HORIZONTAL, variable=timeout_var)
        timeout_scale.pack(fill=tk.X, padx=20)
        
        def save_settings():
            self.enable_desktop = desktop_var.get()
            self.enable_tray = tray_var.get()
            self.enable_sounds = sounds_var.get()
            self.notification_timeout = timeout_var.get()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            root.destroy()
        
        tk.Button(root, text="Save Settings", command=save_settings, bg="green", fg="white").pack(pady=20)
        
        root.mainloop()
    
    def start_notification_system(self) -> None:
        """Start the notification system."""
        # Start notification processing thread
        if not self.notification_thread_running:
            self.notification_thread_running = True
            self.notification_thread = threading.Thread(target=self._notification_processor, daemon=True)
            self.notification_thread.start()
            self.logger.info("Notification processing started")
        
        # Start system tray
        if self.tray_icon and not self.tray_running:
            self.tray_running = True
            threading.Thread(target=self.tray_icon.run, daemon=True).start()
            self.logger.info("System tray started")
    
    def stop_notification_system(self) -> None:
        """Stop the notification system."""
        self.notification_thread_running = False
        
        if self.tray_running:
            self.stop_tray()
    
    def stop_tray(self) -> None:
        """Stop system tray icon."""
        if self.tray_icon and self.tray_running:
            self.tray_running = False
            self.tray_icon.stop()
            self.logger.info("System tray stopped")
    
    def _notification_processor(self) -> None:
        """Process notification queue in background thread."""
        while self.notification_thread_running:
            try:
                # Get notification from queue (with timeout)
                notification_data = self.notification_queue.get(timeout=1)
                
                # Process notification
                self._send_notification(notification_data)
                
                # Add to history
                self._add_to_history(notification_data)
                
                self.notification_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Notification processing error: {e}")
    
    def _send_notification(self, notification_data: Dict[str, Any]) -> None:
        """Send a single notification."""
        title = notification_data.get('title', 'ContextBox')
        message = notification_data.get('message', '')
        notification_type = notification_data.get('type', 'info')
        
        # Map notification types to icons
        type_to_icon = {
            'capture': 'capture',
            'error': 'error',
            'success': 'success',
            'warning': 'warning',
            'info': 'info'
        }
        
        icon_name = type_to_icon.get(notification_type, 'info')
        
        # Send notification based on backend
        if self.notification_backend == "notify2":
            try:
                notification_obj = notify2.Notification(title, message, icon_name)
                notification_obj.set_timeout(self.notification_timeout)
                notification_obj.show()
            except Exception as e:
                self.logger.error(f"notify2 notification failed: {e}")
                
        elif self.notification_backend == "plyer":
            try:
                notification.notify(
                    title=title,
                    message=message,
                    timeout=self.notification_timeout // 1000  # plyer expects seconds
                )
            except Exception as e:
                self.logger.error(f"plyer notification failed: {e}")
    
    def _add_to_history(self, notification_data: Dict[str, Any]) -> None:
        """Add notification to history."""
        # Add timestamp if not present
        if 'timestamp' not in notification_data:
            notification_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        self.notification_history.append(notification_data.copy())
        
        # Limit history size
        if len(self.notification_history) > self.max_notifications:
            self.notification_history = self.notification_history[-self.max_notifications:]
        
        self.notification_count += 1
    
    def notify(self, 
               title: str, 
               message: str, 
               notification_type: str = 'info',
               priority: str = 'normal') -> None:
        """
        Send a notification.
        
        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification ('capture', 'error', 'success', 'warning', 'info')
            priority: Priority level ('low', 'normal', 'high')
        """
        notification_data = {
            'title': title,
            'message': message,
            'type': notification_type,
            'priority': priority,
            'id': self.notification_count + 1
        }
        
        # Add to notification queue
        try:
            self.notification_queue.put(notification_data, timeout=1)
        except queue.Full:
            self.logger.warning("Notification queue full, dropping notification")
    
    def notify_capture_success(self, capture_info: Dict[str, Any]) -> None:
        """Notify successful capture."""
        title = "Context Capture Successful"
        message = f"Captured context from {capture_info.get('source_window', 'Unknown window')}"
        
        if 'artifact_count' in capture_info:
            message += f" - {capture_info['artifact_count']} artifacts found"
        
        self.notify(title, message, 'capture', 'normal')
    
    def notify_extraction_complete(self, extraction_result: Dict[str, Any]) -> None:
        """Notify extraction completion."""
        title = "Content Extraction Complete"
        message = f"Extraction completed with {extraction_result.get('success_count', 0)} successes"
        
        self.notify(title, message, 'success', 'normal')
    
    def notify_error(self, error_message: str, error_details: str = "") -> None:
        """Notify error."""
        title = "ContextBox Error"
        message = error_message
        if error_details:
            message += f"\nDetails: {error_details}"
        
        self.notify(title, message, 'error', 'high')
    
    def notify_search_complete(self, search_results: Dict[str, Any]) -> None:
        """Notify search completion."""
        title = "Search Complete"
        count = search_results.get('result_count', 0)
        message = f"Found {count} results"
        
        self.notify(title, message, 'info', 'low')
    
    def set_quick_capture_callback(self, callback: Callable) -> None:
        """Set callback function for quick capture."""
        self.quick_capture_callback = callback
    
    def set_quit_callback(self, callback: Callable) -> None:
        """Set callback function for quit action."""
        self.quit_callback = callback
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notification history."""
        return self.notification_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear notification history."""
        self.notification_history.clear()
        self.notification_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        return {
            'total_notifications': self.notification_count,
            'backend_used': self.notification_backend,
            'tray_enabled': self.enable_tray,
            'desktop_enabled': self.enable_desktop,
            'queue_size': self.notification_queue.qsize(),
            'history_size': len(self.notification_history)
        }


def create_notification_system(config: Optional[Dict[str, Any]] = None) -> NotificationSystem:
    """
    Factory function to create a notification system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NotificationSystem instance
    """
    return NotificationSystem(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create notification system
    config = {
        'enable_desktop': True,
        'enable_tray': True,
        'notification_timeout': 3000,
        'enable_sounds': False,
        'max_notifications': 100
    }
    
    notification_system = create_notification_system(config)
    
    # Test notifications
    notification_system.notify("Test Notification", "This is a test notification", "info")
    notification_system.notify("Capture Success", "Successfully captured screen content", "capture")
    notification_system.notify("Error", "Something went wrong", "error")
    
    # Start the system
    notification_system.start_notification_system()
    
    # Keep running for testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        notification_system.stop_notification_system()
        print("Notification system stopped")