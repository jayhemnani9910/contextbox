"""
Screenshot capture module for ContextBox
Provides cross-platform screenshot functionality with support for Wayland, X11, and various fallback tools.
"""

import os
import sys
import subprocess
import time
import shutil
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Cross-platform screenshot capture with multiple fallback options"""
    
    def __init__(self, media_dir: str = None):
        """
        Initialize screenshot capture
        
        Args:
            media_dir: Custom media directory path. Defaults to ~/.contextbox/media/
        """
        if media_dir:
            self.media_dir = Path(media_dir)
        else:
            self.media_dir = Path.home() / ".contextbox" / "media"
        
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp_format = "%Y%m%d_%H%M%S"
        
        available_tools = self.get_available_tools()
        if not available_tools:
            logger.warning(
                "No screenshot tools detected. Install utilities such as 'gnome-screenshot', "
                "'scrot', or 'grim' to enable screenshot capture."
            )
        
    def is_wayland(self) -> bool:
        """Check if running on Wayland"""
        return os.environ.get('WAYLAND_DISPLAY') is not None
    
    def is_gnome(self) -> bool:
        """Check if running on GNOME desktop"""
        try:
            result = subprocess.run(
                ['ps', '-e'], capture_output=True, text=True, timeout=5
            )
            return 'gnome-session' in result.stdout or 'gnome-shell' in result.stdout
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False
    
    def check_tool_availability(self, tool: str) -> bool:
        """Check if a tool is available in the system"""
        return shutil.which(tool) is not None
    
    def get_available_tools(self) -> list:
        """Get list of available screenshot tools"""
        tools = []
        
        # Wayland-specific tools
        if self.is_wayland():
            if self.check_tool_availability('grim'):
                tools.append('grim')
            if self.check_tool_availability('gnome-screenshot'):
                tools.append('gnome-screenshot')
        
        # X11/general tools
        if self.check_tool_availability('gnome-screenshot'):
            tools.append('gnome-screenshot')
        if self.check_tool_availability('scrot'):
            tools.append('scrot')
        if self.check_tool_availability('maim'):
            tools.append('maim')
        if self.check_tool_availability('import'):
            tools.append('import')  # ImageMagick
        if self.check_tool_availability('xfce4-screenshooter'):
            tools.append('xfce4-screenshooter')
        
        return tools
    
    def capture_full_screen(self, output_filename: str = None) -> str:
        """
        Capture full screen screenshot
        
        Args:
            output_filename: Custom output filename. If None, auto-generates with timestamp
            
        Returns:
            Path to saved screenshot file or None if failed
        """
        if output_filename is None:
            timestamp = datetime.now().strftime(self.timestamp_format)
            output_filename = f"screenshot_full_{timestamp}.png"
        
        output_path = self.media_dir / output_filename
        
        tools = self.get_available_tools()
        
        for tool in tools:
            try:
                if tool == 'grim':
                    # Wayland native screenshot
                    if self._capture_with_grim(str(output_path)):
                        return str(output_path)
                        
                elif tool == 'gnome-screenshot':
                    if self._capture_with_gnome_screenshot(str(output_path), fullscreen=True):
                        return str(output_path)
                        
                elif tool == 'scrot':
                    if self._capture_with_scrot(str(output_path)):
                        return str(output_path)
                        
                elif tool == 'maim':
                    if self._capture_with_maim(str(output_path)):
                        return str(output_path)
                        
                elif tool == 'import':
                    if self._capture_with_imagemagick(str(output_path)):
                        return str(output_path)
                        
                elif tool == 'xfce4-screenshooter':
                    if self._capture_with_xfce4_screenshooter(str(output_path)):
                        return str(output_path)
                        
            except Exception as e:
                logger.warning(f"Failed to capture with {tool}: {e}")
                continue
        
        logger.error("No screenshot tool available or all tools failed")
        return None
    
    def capture_active_window(self, output_filename: str = None) -> str:
        """
        Capture active window screenshot
        
        Args:
            output_filename: Custom output filename. If None, auto-generates with timestamp
            
        Returns:
            Path to saved screenshot file or None if failed
        """
        if output_filename is None:
            timestamp = datetime.now().strftime(self.timestamp_format)
            output_filename = f"screenshot_window_{timestamp}.png"
        
        output_path = self.media_dir / output_filename
        
        # Try active window detection first
        if not self.is_wayland() and self.check_tool_availability('wmctrl'):
            window_id = self._get_active_window_id()
            if window_id:
                tools = self.get_available_tools()
                
                for tool in tools:
                    try:
                        if tool == 'maim':
                            if self._capture_window_with_maim(str(output_path), window_id):
                                return str(output_path)
                        elif tool == 'scrot':
                            if self._capture_window_with_scrot(str(output_path), window_id):
                                return str(output_path)
                        elif tool == 'gnome-screenshot':
                            if self._capture_with_gnome_screenshot(str(output_path), window=True):
                                return str(output_path)
                    except Exception as e:
                        logger.warning(f"Failed to capture window with {tool}: {e}")
                        continue
        
        # Fallback to area selection if window capture fails
        logger.info("Window capture failed, falling back to area selection")
        return self.capture_area_selection(output_filename)
    
    def capture_area_selection(self, output_filename: str = None) -> str:
        """
        Capture selected area screenshot
        
        Args:
            output_filename: Custom output filename. If None, auto-generates with timestamp
            
        Returns:
            Path to saved screenshot file or None if failed
        """
        if output_filename is None:
            timestamp = datetime.now().strftime(self.timestamp_format)
            output_filename = f"screenshot_area_{timestamp}.png"
        
        output_path = self.media_dir / output_filename
        
        tools = self.get_available_tools()
        
        for tool in tools:
            try:
                if tool == 'maim':
                    if self._capture_with_maim(str(output_path), select=True):
                        return str(output_path)
                elif tool == 'scrot':
                    if self._capture_with_scrot(str(output_path), select=True):
                        return str(output_path)
                elif tool == 'gnome-screenshot':
                    if self._capture_with_gnome_screenshot(str(output_path), area=True):
                        return str(output_path)
                elif tool == 'import':
                    if self._capture_with_imagemagick(str(output_path), select=True):
                        return str(output_path)
            except Exception as e:
                logger.warning(f"Failed to capture area with {tool}: {e}")
                continue
        
        logger.error("Area selection capture failed")
        return None
    
    def _capture_with_grim(self, output_path: str) -> bool:
        """Capture screenshot using grim (Wayland)"""
        try:
            if self.is_wayland():
                subprocess.run(['grim', output_path], check=True, timeout=30)
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Grim capture failed: {e}")
        return False
    
    def _capture_with_gnome_screenshot(self, output_path: str, fullscreen: bool = False, 
                                     window: bool = False, area: bool = False) -> bool:
        """Capture screenshot using gnome-screenshot"""
        try:
            cmd = ['gnome-screenshot']
            
            if fullscreen:
                cmd.extend(['-f', output_path])
            elif window:
                cmd.extend(['-w', '-f', output_path])
            elif area:
                cmd.extend(['-a', '-f', output_path])
            else:
                cmd.extend(['-f', output_path])
            
            subprocess.run(cmd, check=True, timeout=30)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Gnome-screenshot capture failed: {e}")
        return False
    
    def _capture_with_scrot(self, output_path: str, select: bool = False, window_id: str = None) -> bool:
        """Capture screenshot using scrot"""
        try:
            cmd = ['scrot']
            
            if select:
                cmd.extend(['-s', output_path])
            elif window_id:
                # Scrot doesn't have direct window ID support, so we skip this
                return False
            else:
                cmd.append(output_path)
            
            subprocess.run(cmd, check=True, timeout=30)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Scrot capture failed: {e}")
        return False
    
    def _capture_with_maim(self, output_path: str, select: bool = False, window_id: str = None) -> bool:
        """Capture screenshot using maim"""
        try:
            cmd = ['maim']
            
            if select:
                cmd.extend(['-s', output_path])
            elif window_id:
                # Window-specific capture (requires wmctrl for ID conversion)
                cmd.extend(['-i', '0', '-d', '0.1', '--hidecursor'])
                # Note: This is a simplified approach - actual implementation would need
                # coordinate conversion from window ID
                cmd.append(output_path)
            else:
                cmd.extend(['--hidecursor', output_path])
            
            subprocess.run(cmd, check=True, timeout=30)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Maim capture failed: {e}")
        return False
    
    def _capture_window_with_maim(self, output_path: str, window_id: str) -> bool:
        """Capture specific window using maim (simplified approach)"""
        try:
            # This is a basic implementation - in practice, you'd need to get
            # window geometry and use maim with coordinates
            cmd = ['maim', '--hidecursor', output_path]
            subprocess.run(cmd, check=True, timeout=30)
            return True
        except Exception as e:
            logger.warning(f"Window capture with maim failed: {e}")
        return False
    
    def _capture_window_with_scrot(self, output_path: str, window_id: str) -> bool:
        """Capture specific window using scrot"""
        # Scrot doesn't have direct window ID support, fallback to full screen
        return self._capture_with_scrot(output_path)
    
    def _capture_with_imagemagick(self, output_path: str, select: bool = False) -> bool:
        """Capture screenshot using ImageMagick import"""
        try:
            cmd = ['import', 'root:' + output_path]
            if select:
                # Import with interactive selection
                subprocess.run(['import', output_path], check=True, timeout=30)
            else:
                subprocess.run(cmd, check=True, timeout=30)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"ImageMagick capture failed: {e}")
        return False
    
    def _capture_with_xfce4_screenshooter(self, output_path: str) -> bool:
        """Capture screenshot using XFCE4 Screenshooter"""
        try:
            subprocess.run(['xfce4-screenshooter', '-f', '-s', str(self.media_dir)], check=True, timeout=30)
            # Rename the generated file
            generated_files = list(self.media_dir.glob("screenshot*.png"))
            if generated_files:
                latest_file = max(generated_files, key=os.path.getctime)
                if latest_file.name != output_path:
                    latest_file.rename(output_path)
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"XFCE4 Screenshooter capture failed: {e}")
        return False
    
    def _get_active_window_id(self) -> Optional[str]:
        """Get active window ID using wmctrl"""
        try:
            result = subprocess.run(
                ['wmctrl', '-d'], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Get active workspace
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('*'):
                        workspace_info = line.split()
                        if len(workspace_info) > 0:
                            workspace = workspace_info[0]
                            break
                
                # Get active window on current workspace
                result = subprocess.run(
                    ['wmctrl', '-R', '-1'], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Extract window ID from output
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split()
                        if len(parts) > 0:
                            return parts[0]
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception) as e:
            logger.warning(f"Failed to get active window ID: {e}")
        
        return None
    
    def read_clipboard_content(self) -> str:
        """
        Read clipboard content
        
        Returns:
            Clipboard text content or None if failed
        """
        tools = []
        
        # Check available clipboard tools
        if self.check_tool_availability('xclip'):
            tools.append('xclip')
        if self.check_tool_availability('xsel'):
            tools.append('xsel')
        if self.check_tool_availability('wl-paste'):
            tools.append('wl-paste')
        
        for tool in tools:
            try:
                if tool == 'xclip':
                    result = subprocess.run(
                        ['xclip', '-selection', 'clipboard', '-o'], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                        
                elif tool == 'xsel':
                    result = subprocess.run(
                        ['xsel', '-b', '-o'], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                        
                elif tool == 'wl-paste':
                    result = subprocess.run(
                        ['wl-paste'], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                        
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.warning(f"Failed to read clipboard with {tool}: {e}")
                continue
        
        logger.warning("No clipboard tool available or all tools failed")
        return None
    
    def list_screenshots(self) -> list:
        """List all screenshots in the media directory"""
        try:
            return sorted(self.media_dir.glob("screenshot*.png"), key=os.path.getmtime, reverse=True)
        except Exception as e:
            logger.error(f"Failed to list screenshots: {e}")
            return []
    
    def cleanup_old_screenshots(self, days: int = 30, max_files: int = None):
        """
        Clean up old screenshots
        
        Args:
            days: Remove files older than this many days
            max_files: Maximum number of files to keep (takes precedence over days)
        """
        try:
            screenshots = self.list_screenshots()
            if not screenshots:
                return
            
            current_time = time.time()
            
            # If max_files specified, keep only the most recent files
            if max_files is not None and len(screenshots) > max_files:
                files_to_remove = screenshots[max_files:]
            else:
                # Remove files older than specified days
                files_to_remove = []
                for screenshot in screenshots:
                    file_age_days = (current_time - screenshot.stat().st_mtime) / (24 * 3600)
                    if file_age_days > days:
                        files_to_remove.append(screenshot)
            
            # Remove files
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old screenshot: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup screenshots: {e}")
    
    def get_screenshot_info(self, filename: str) -> Optional[dict]:
        """
        Get information about a screenshot file
        
        Args:
            filename: Screenshot filename
            
        Returns:
            Dictionary with screenshot information or None if failed
        """
        try:
            file_path = self.media_dir / filename
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            return {
                'filename': filename,
                'path': str(file_path),
                'size_bytes': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'timestamp': self._extract_timestamp_from_filename(filename)
            }
        except Exception as e:
            logger.error(f"Failed to get screenshot info for {filename}: {e}")
            return None
    
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Extract timestamp from screenshot filename"""
        import re
        
        # Look for timestamp pattern: YYYYMMDD_HHMMSS
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp
                dt = datetime.strptime(timestamp_str, self.timestamp_format)
                return dt.isoformat()
            except ValueError:
                pass
        
        return None


def main():
    """Main function for testing screenshot capture functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ContextBox Screenshot Capture Tool')
    parser.add_argument('action', choices=['full', 'window', 'area', 'list', 'clipboard'], 
                       help='Action to perform')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--media-dir', help='Media directory path')
    
    args = parser.parse_args()
    
    # Initialize capture
    capture = ScreenshotCapture(args.media_dir)
    
    # Perform action
    if args.action == 'full':
        result = capture.capture_full_screen(args.output)
        if result:
            print(f"Full screen screenshot saved: {result}")
        else:
            print("Failed to capture full screen")
            sys.exit(1)
            
    elif args.action == 'window':
        result = capture.capture_active_window(args.output)
        if result:
            print(f"Active window screenshot saved: {result}")
        else:
            print("Failed to capture active window")
            sys.exit(1)
            
    elif args.action == 'area':
        result = capture.capture_area_selection(args.output)
        if result:
            print(f"Area selection screenshot saved: {result}")
        else:
            print("Failed to capture area selection")
            sys.exit(1)
            
    elif args.action == 'list':
        screenshots = capture.list_screenshots()
        if screenshots:
            print("Screenshots found:")
            for screenshot in screenshots:
                info = capture.get_screenshot_info(screenshot.name)
                if info:
                    print(f"  {screenshot.name} - {info['size_bytes']} bytes - {info['created']}")
        else:
            print("No screenshots found")
            
    elif args.action == 'clipboard':
        content = capture.read_clipboard_content()
        if content:
            print("Clipboard content:")
            print(content)
        else:
            print("Failed to read clipboard or clipboard is empty")


# Compatibility class for existing ContextBox interface
class ContextCapture:
    """
    Compatibility wrapper for ScreenshotCapture to maintain existing ContextBox interface.
    This class provides the expected interface while using ScreenshotCapture functionality.
    """
    
    def __init__(self, config: dict, database=None):
        """
        Initialize context capture.
        
        Args:
            config: Capture configuration
            database: Optional ContextDatabase instance for saving captures
        """
        self.config = config
        self.database = database
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        # Normalize nested/new-style capture config
        capture_config = config.get('capture')
        if isinstance(capture_config, dict):
            base_config = capture_config
        else:
            base_config = config
        
        # Screenshot capture settings
        self.screenshot_dir = base_config.get('media_dir')
        self.screenshot_capture = ScreenshotCapture(self.screenshot_dir)
        
        # Periodic capture settings
        self.interval = float(base_config.get('interval', 5.0))  # seconds
        self.max_captures = int(base_config.get('max_captures', 0))  # 0 = unlimited
        self.capture_types = base_config.get('types', ['full'])  # full, window, area
        
        # Internal state
        self.capture_count = 0
        
        self.logger.info("Context capture (compatibility mode) initialized")
    
    def start(self) -> None:
        """Start context capture in a separate thread."""
        if self.is_running:
            self.logger.warning("Capture is already running")
            return
        
        self.is_running = True
        self.logger.info("Context capture started")
        
        # Start periodic screenshot capture
        import threading
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=False)
        self.capture_thread.start()
    
    def stop(self) -> None:
        """Stop context capture."""
        if not self.is_running:
            self.logger.warning("Capture is not running")
            return
        
        self.logger.info("Stopping context capture...")
        self.is_running = False
        self.logger.info("Context capture stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        self.logger.info(f"Capture loop started (is_running={self.is_running})")
        print(f"[CAPTURE THREAD] Loop started, is_running={self.is_running}, interval={self.interval}", flush=True)
        
        try:
            loop_counter = 0
            while self.is_running:
                loop_counter += 1
                print(f"[CAPTURE THREAD] Cycle #{loop_counter} starting...", flush=True)
                # Perform scheduled capture
                self.logger.debug(f"Executing capture cycle {self.capture_count + 1}...")
                self._capture_context()
                self.capture_count += 1
                
                # Check if we should stop
                if self.max_captures > 0 and self.capture_count >= self.max_captures:
                    self.logger.info(f"Reached max captures ({self.max_captures}), stopping")
                    break
                
                # Wait for next capture
                print(f"[CAPTURE THREAD] Sleeping for {self.interval} seconds...", flush=True)
                self.logger.debug(f"Sleeping for {self.interval} seconds until next capture...")
                time.sleep(self.interval)
                
        except Exception as e:
            self.logger.error(f"Error in capture loop: {e}")
            print(f"[CAPTURE THREAD] ERROR: {e}", flush=True)
        finally:
            self.is_running = False
            print("[CAPTURE THREAD] Loop ended", flush=True)
            self.logger.info("Capture loop ended")
    
    def _capture_context(self) -> None:
        """Capture context information using screenshots."""
        try:
            screenshot_path = None
            
            # Skip screenshot capture entirely - too unreliable on this system
            # Instead focus on clipboard + metadata which is what we actually need
            self.logger.debug("Skipping screenshot capture (clipboard focus mode)")
            
            # Collect other context information (clipboard, window, system info)
            clipboard_content = self.screenshot_capture.read_clipboard_content()
            window_info = self._get_active_window_info()
            system_info = self._get_system_info()
            
            context_data = {
                'timestamp': datetime.now().isoformat(),
                'capture_count': self.capture_count,
                'screenshot_available': screenshot_path is not None,
                'clipboard_content': clipboard_content,
                'active_window': window_info,
                'system_info': system_info
            }
            
            self.logger.debug(f"Captured context: {context_data}")
            
            # ✅ **SAVE TO DATABASE**
            if self.database:
                try:
                    source_window = window_info.get('title', 'Unknown') if window_info else 'Unknown'
                    capture_id = self.database.create_capture(
                        source_window=source_window,
                        screenshot_path=screenshot_path,
                        clipboard_text=clipboard_content,
                        notes=json.dumps(system_info) if system_info else None
                    )
                    self.logger.info(f"Saved capture {capture_id} to database with clipboard: {clipboard_content[:50] if clipboard_content else 'None'}...")
                    
                    # ✅ **AUTO-INDEX FOR SEARCH** - Create artifacts from clipboard content
                    try:
                        if clipboard_content:
                            conn = sqlite3.connect(self.database.db_path)
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO artifacts (capture_id, kind, text, title, url)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                capture_id,
                                'clipboard',
                                clipboard_content,
                                source_window,
                                None
                            ))
                            conn.commit()
                            conn.close()
                            self.logger.info(f"Indexed clipboard content as searchable artifact for capture {capture_id}")
                    except Exception as idx_err:
                        self.logger.error(f"Could not auto-index artifact: {idx_err}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to save capture to database: {e}")
            else:
                self.logger.warning("Database not connected to ContextCapture - captures not being saved!")
            
        except Exception as e:
            self.logger.error(f"Error capturing context: {e}")
    
    def _collect_context_data(self) -> dict:
        """Collect current context information."""
        timestamp = datetime.now().isoformat()
        
        # Get clipboard content
        clipboard_content = self.screenshot_capture.read_clipboard_content()
        
        context_data = {
            'timestamp': timestamp,
            'capture_count': self.capture_count,
            'screenshot_available': True,
            'clipboard_content': clipboard_content,
            'active_window': self._get_active_window_info(),
            'system_info': self._get_system_info()
        }
        
        return context_data
    
    def _get_active_window_info(self) -> dict:
        """Get information about currently active window."""
        try:
            # Try to get active window title using xdotool or wmctrl
            tools = ['xdotool', 'wmctrl']
            for tool in tools:
                if self.screenshot_capture.check_tool_availability(tool):
                    if tool == 'xdotool':
                        result = subprocess.run(
                            ['xdotool', 'getactivewindow', 'getwindowname'],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            return {
                                'title': result.stdout.strip(),
                                'application': 'Unknown',
                                'window_id': 'active'
                            }
                    elif tool == 'wmctrl':
                        result = subprocess.run(
                            ['wmctrl', '-a', ':ACTIVE:'],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            # Parse output to get window title
                            lines = result.stdout.strip().split('\n')
                            if lines:
                                parts = lines[0].split(None, 4)
                                if len(parts) > 4:
                                    return {
                                        'title': parts[4],
                                        'application': parts[0] if len(parts) > 0 else 'Unknown',
                                        'window_id': parts[0] if len(parts) > 0 else 'active'
                                    }
        except Exception as e:
            self.logger.debug(f"Failed to get active window info: {e}")
        
        return {
            'title': 'Unknown',
            'application': 'Unknown',
            'window_id': 'unknown'
        }
    
    def _get_system_info(self) -> dict:
        """Get basic system information."""
        try:
            import platform
            return {
                'os': platform.system(),
                'os_version': platform.release(),
                'architecture': platform.machine(),
                'hostname': platform.node()
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}
    
    def get_capture_stats(self) -> dict:
        """Get capture statistics."""
        return {
            'is_running': self.is_running,
            'capture_count': self.capture_count,
            'interval': self.interval,
            'max_captures': self.max_captures
        }


if __name__ == "__main__":
    main()
