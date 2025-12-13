#!/usr/bin/env python3

"""
ContextBox Post-Installation Setup Wizard
Interactive wizard to configure ContextBox preferences after installation
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    rich_available = True
    console = Console()
except ImportError:
    rich_available = False


class ContextBoxSetupWizard:
    """Interactive setup wizard for ContextBox configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.expanduser("~/.config/contextbox/config.json")
        self.config_dir = os.path.dirname(self.config_path)
        self.config = {}
        self.defaults = self._load_default_config()
        self.terminal_mode = not GUI_AVAILABLE and not rich_available
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration template"""
        return {
            "version": "2.0",
            "log_level": "INFO",
            "backup_on_startup": True,
            "capture": {
                "interval": 1.0,
                "max_captures": 0,
                "enabled_sources": ["clipboard", "active_window", "recent_files"],
                "screenshot": {
                    "format": "png",
                    "quality": 90,
                    "delay": 0.1,
                    "include_cursor": False
                }
            },
            "database": {
                "path": "contextbox.db",
                "backup_enabled": True,
                "backup_interval": 3600,
                "auto_vacuum": True,
                "max_db_size": "500MB"
            },
            "extractors": {
                "enabled_extractors": ["text", "system", "network"],
                "confidence_threshold": 0.5,
                "parallel_processing": True,
                "max_workers": 4,
                "cache_enabled": True,
                "cache_size": 100
            },
            "llm": {
                "provider": "local",
                "model": "llama3.1",
                "api_key": None,
                "max_tokens": 1000,
                "temperature": 0.7,
                "enabled": True
            },
            "security": {
                "encrypt_stored_data": False,
                "retention_days": 30,
                "auto_cleanup": True,
                "secure_delete": True,
                "privacy_mode": False
            },
            "ui": {
                "notifications": True,
                "theme": "dark",
                "show_in_tray": True,
                "hotkey": "Ctrl+Alt+C",
                "window_size": "800x600",
                "always_on_top": False
            },
            "performance": {
                "cache_size": 100,
                "max_memory_usage": "1GB",
                "optimize_for_speed": False,
                "background_processing": True,
                "batch_operations": True
            },
            "integrations": {
                "browser_extension": False,
                "desktop_shortcuts": True,
                "system_tray": True,
                "auto_start": False,
                "cloud_sync": False,
                "cloud_provider": "none"
            }
        }
    
    def display_welcome(self):
        """Display welcome message"""
        if self.terminal_mode:
            print("\n" + "="*60)
            print("  ContextBox Setup Wizard")
            print("="*60)
            print("\nWelcome to ContextBox! This wizard will help you configure")
            print("your preferred settings for the best experience.")
            print("\nPress Enter to continue or Ctrl+C to exit...")
            input()
        elif rich_available:
            panel = Panel.fit(
                "[bold blue]ContextBox Setup Wizard[/bold blue]\n\n"
                "Welcome! This wizard will help you configure ContextBox for your needs.\n"
                "We'll guide you through setting up:\n"
                "• Capture preferences\n"
                "• Privacy and security settings\n"
                "• User interface options\n"
                "• LLM integration\n"
                "• Performance optimization",
                border_style="blue"
            )
            console.print(panel)
            console.print("\n[yellow]Press Enter to continue...[/yellow]")
            input()
        else:
            print("ContextBox Setup Wizard")
            print("=" * 30)
    
    def load_existing_config(self) -> bool:
        """Load existing configuration if available"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                
                if self.terminal_mode:
                    response = input(f"\nExisting configuration found at {self.config_path}. Load it? (y/N): ")
                    if response.lower() != 'y':
                        self.config = self.defaults.copy()
                        return False
                    return True
                elif rich_available:
                    if Confirm.ask(f"\nExisting configuration found at {self.config_path}. Load it?"):
                        return True
                    else:
                        self.config = self.defaults.copy()
                        return False
                else:
                    return True
            except Exception as e:
                print(f"Error loading existing config: {e}")
                self.config = self.defaults.copy()
                return False
        else:
            self.config = self.defaults.copy()
            return False
    
    def setup_capture_preferences(self):
        """Configure capture-related settings"""
        self.print_section("Capture Preferences")
        
        # Capture interval
        interval = self._get_user_input(
            "Capture interval (seconds, 0 for manual only):",
            self.config["capture"]["interval"],
            float,
            "Higher values reduce CPU usage but capture less frequently"
        )
        self.config["capture"]["interval"] = interval
        
        # Max captures (0 for unlimited)
        max_captures = self._get_user_input(
            "Maximum captures (0 for unlimited):",
            self.config["capture"]["max_captures"],
            int,
            "Set to 0 to capture without limits, or a specific number to prevent database growth"
        )
        self.config["capture"]["max_captures"] = max_captures
        
        # Enabled sources
        print("\nCapture sources (space-separated, comma for none):")
        available_sources = ["screenshot", "clipboard", "active_window", "recent_files", "system_info"]
        current_sources = self.config["capture"]["enabled_sources"]
        
        if self.terminal_mode:
            print(f"Available: {', '.join(available_sources)}")
            print(f"Current: {', '.join(current_sources)}")
            selection = input("Enter sources to enable: ").strip()
            if selection:
                self.config["capture"]["enabled_sources"] = [
                    s.strip() for s in selection.split() if s.strip() in available_sources
                ]
        elif rich_available:
            selected = []
            for source in available_sources:
                if Confirm.ask(f"Enable {source}?", default=source in current_sources):
                    selected.append(source)
            self.config["capture"]["enabled_sources"] = selected
        else:
            # Simple GUI or default
            self.config["capture"]["enabled_sources"] = current_sources
        
        # Screenshot settings
        if "screenshot" in self.config["capture"]:
            screenshot_config = self.config["capture"]["screenshot"]
            
            screenshot_format = self._get_user_input(
                "Screenshot format (png/jpg):",
                screenshot_config.get("format", "png"),
                str,
                "PNG for lossless quality, JPG for smaller file size"
            )
            screenshot_config["format"] = screenshot_format.lower()
            
            screenshot_quality = self._get_user_input(
                "Screenshot quality (1-100):",
                screenshot_config.get("quality", 90),
                int,
                "Higher values = better quality but larger files"
            )
            screenshot_config["quality"] = max(1, min(100, screenshot_quality))
    
    def setup_privacy_security(self):
        """Configure privacy and security settings"""
        self.print_section("Privacy & Security")
        
        # Data encryption
        encrypt = self._get_user_input_bool(
            "Enable data encryption?",
            self.config["security"]["encrypt_stored_data"],
            "Encrypts stored data for additional privacy"
        )
        self.config["security"]["encrypt_stored_data"] = encrypt
        
        # Data retention
        retention = self._get_user_input(
            "Data retention period (days):",
            self.config["security"]["retention_days"],
            int,
            "Older data will be automatically deleted after this period"
        )
        self.config["security"]["retention_days"] = max(1, retention)
        
        # Auto cleanup
        auto_cleanup = self._get_user_input_bool(
            "Enable automatic cleanup?",
            self.config["security"]["auto_cleanup"],
            "Automatically removes expired and temporary files"
        )
        self.config["security"]["auto_cleanup"] = auto_cleanup
        
        # Privacy mode
        privacy_mode = self._get_user_input_bool(
            "Enable privacy mode?",
            self.config["security"]["privacy_mode"],
            "Prevents sending data to external services"
        )
        self.config["security"]["privacy_mode"] = privacy_mode
    
    def setup_user_interface(self):
        """Configure UI settings"""
        self.print_section("User Interface")
        
        # Theme selection
        if self.terminal_mode:
            theme = input("Theme (dark/light) [dark]: ").strip().lower()
            if theme not in ['dark', 'light']:
                theme = 'dark'
        elif rich_available:
            theme = Prompt.ask(
                "Theme",
                choices=["dark", "light"],
                default=self.config["ui"]["theme"]
            )
        else:
            theme = self.config["ui"]["theme"]
        
        self.config["ui"]["theme"] = theme
        
        # Show in system tray
        show_tray = self._get_user_input_bool(
            "Show in system tray?",
            self.config["ui"]["show_in_tray"],
            "Allows quick access from system tray"
        )
        self.config["ui"]["show_in_tray"] = show_tray
        
        # Enable notifications
        notifications = self._get_user_input_bool(
            "Enable desktop notifications?",
            self.config["ui"]["notifications"],
            "Shows system notifications for capture events"
        )
        self.config["ui"]["notifications"] = notifications
        
        # Hotkey configuration
        print("\nConfigure global hotkey:")
        if self.terminal_mode:
            hotkey = input(f"Hotkey [{self.config['ui']['hotkey']}]: ").strip()
            if hotkey:
                self.config["ui"]["hotkey"] = hotkey
        else:
            self.config["ui"]["hotkey"] = self.config["ui"]["hotkey"]  # Keep default for now
    
    def setup_llm_integration(self):
        """Configure LLM integration settings"""
        self.print_section("LLM Integration")
        
        # Enable LLM
        enable_llm = self._get_user_input_bool(
            "Enable LLM integration?",
            self.config["llm"]["enabled"],
            "Allows AI-powered context analysis and summarization"
        )
        self.config["llm"]["enabled"] = enable_llm
        
        if not enable_llm:
            return
        
        # Provider selection
        if self.terminal_mode:
            print("\nAvailable providers: local, openai, anthropic, ollama")
            provider = input(f"Provider [{self.config['llm']['provider']}]: ").strip()
            if provider:
                self.config["llm"]["provider"] = provider
        elif rich_available:
            self.config["llm"]["provider"] = Prompt.ask(
                "LLM Provider",
                choices=["local", "openai", "anthropic", "ollama"],
                default=self.config["llm"]["provider"]
            )
        
        # Model configuration
        if self.terminal_mode:
            model = input(f"Model [{self.config['llm']['model']}]: ").strip()
            if model:
                self.config["llm"]["model"] = model
        
        # API key for hosted providers
        if self.config["llm"]["provider"] in ["openai", "anthropic"]:
            if self.terminal_mode:
                api_key = input("API Key (leave empty to skip): ").strip()
                if api_key:
                    self.config["llm"]["api_key"] = api_key
            elif rich_available:
                api_key = Prompt.ask("API Key", password=True, default="")
                if api_key:
                    self.config["llm"]["api_key"] = api_key
    
    def setup_performance_optimization(self):
        """Configure performance settings"""
        self.print_section("Performance Optimization")
        
        # Optimize for speed vs. resource usage
        optimize_speed = self._get_user_input_bool(
            "Optimize for speed?",
            self.config["performance"]["optimize_for_speed"],
            "Uses more CPU/memory for faster processing"
        )
        self.config["performance"]["optimize_for_speed"] = optimize_speed
        
        # Cache size
        cache_size = self._get_user_input(
            "Cache size (items):",
            self.config["performance"]["cache_size"],
            int,
            "Higher values use more memory but improve performance"
        )
        self.config["performance"]["cache_size"] = max(10, cache_size)
        
        # Max memory usage
        max_memory = self._get_user_input(
            "Max memory usage (MB):",
            1024,
            int,
            "ContextBox will limit memory usage to this amount"
        )
        self.config["performance"]["max_memory_usage"] = f"{max_memory}MB"
        
        # Background processing
        bg_processing = self._get_user_input_bool(
            "Enable background processing?",
            self.config["performance"]["background_processing"],
            "Process data in background for better responsiveness"
        )
        self.config["performance"]["background_processing"] = bg_processing
    
    def setup_integrations(self):
        """Configure system integrations"""
        self.print_section("System Integrations")
        
        # Desktop shortcuts
        shortcuts = self._get_user_input_bool(
            "Create desktop shortcuts?",
            self.config["integrations"]["desktop_shortcuts"],
            "Creates shortcuts for easy access"
        )
        self.config["integrations"]["desktop_shortcuts"] = shortcuts
        
        # System tray
        system_tray = self._get_user_input_bool(
            "Enable system tray integration?",
            self.config["integrations"]["system_tray"],
            "Adds ContextBox to system tray for quick access"
        )
        self.config["integrations"]["system_tray"] = system_tray
        
        # Auto-start
        auto_start = self._get_user_input_bool(
            "Start ContextBox automatically?",
            self.config["integrations"]["auto_start"],
            "ContextBox will start when you log in"
        )
        self.config["integrations"]["auto_start"] = auto_start
        
        # Browser extension (if available)
        if self._check_browser_support():
            browser_ext = self._get_user_input_bool(
                "Enable browser extension?",
                self.config["integrations"]["browser_extension"],
                "Capture context directly from web pages"
            )
            self.config["integrations"]["browser_extension"] = browser_ext
    
    def _check_browser_support(self) -> bool:
        """Check if browser extension support is available"""
        # This would check for browser extension support
        return True  # Simplified for now
    
    def review_and_confirm(self) -> bool:
        """Review configuration and get user confirmation"""
        self.print_section("Configuration Review")
        
        # Display configuration summary
        self._display_config_summary()
        
        # Confirmation
        if self.terminal_mode:
            response = input("\nApply these settings? (Y/n): ").strip().lower()
            return response != 'n'
        elif rich_available:
            return Confirm.ask("\nApply these settings?")
        else:
            return True
    
    def _display_config_summary(self):
        """Display a summary of the current configuration"""
        summary_items = [
            ("Capture interval", f"{self.config['capture']['interval']}s"),
            ("Data retention", f"{self.config['security']['retention_days']} days"),
            ("Theme", self.config['ui']['theme']),
            ("LLM enabled", "Yes" if self.config['llm']['enabled'] else "No"),
            ("Hotkey", self.config['ui']['hotkey']),
            ("Encryption", "Yes" if self.config['security']['encrypt_stored_data'] else "No"),
        ]
        
        if rich_available:
            table = Table(title="Configuration Summary")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            
            for setting, value in summary_items:
                table.add_row(setting, str(value))
            
            console.print(table)
        else:
            print("\nConfiguration Summary:")
            print("-" * 40)
            for setting, value in summary_items:
                print(f"{setting}: {value}")
            print("-" * 40)
    
    def save_configuration(self) -> bool:
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Save as JSON
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Also save as YAML for human readability
            yaml_path = self.config_path.replace('.json', '.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            print(f"Configuration saved to: {self.config_path}")
            print(f"YAML backup saved to: {yaml_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def create_shortcuts_and_integrations(self):
        """Create system shortcuts and configure integrations"""
        print("\nCreating shortcuts and integrations...")
        
        try:
            # Create desktop shortcuts if requested
            if self.config["integrations"]["desktop_shortcuts"]:
                self._create_desktop_shortcuts()
            
            # Configure auto-start if requested
            if self.config["integrations"]["auto_start"]:
                self._configure_auto_start()
            
            # Configure system tray if requested
            if self.config["integrations"]["system_tray"]:
                self._configure_system_tray()
            
            print("✓ Shortcuts and integrations created successfully")
        except Exception as e:
            print(f"Warning: Could not create shortcuts: {e}")
    
    def _create_desktop_shortcuts(self):
        """Create desktop shortcuts"""
        desktop_dir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(desktop_dir, exist_ok=True)
        
        shortcut_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=ContextBox
Comment=Context capture and organization system
Exec=contextbox %f
Icon=applications-accessories
Terminal=true
Categories=Utility;Accessories;
Keywords=screenshot;capture;context;notes;
"""
        
        shortcut_path = os.path.join(desktop_dir, "contextbox.desktop")
        with open(shortcut_path, 'w') as f:
            f.write(shortcut_content)
        
        os.chmod(shortcut_path, 0o755)
    
    def _configure_auto_start(self):
        """Configure auto-start with desktop environment"""
        autostart_dir = os.path.expanduser("~/.config/autostart")
        os.makedirs(autostart_dir, exist_ok=True)
        
        # Create autostart file
        autostart_content = f"""[Desktop Entry]
Type=Application
Name=ContextBox
Exec=contextbox --start-background
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
        
        autostart_path = os.path.join(autostart_dir, "contextbox.desktop")
        with open(autostart_path, 'w') as f:
            f.write(autostart_content)
    
    def _configure_system_tray(self):
        """Configure system tray integration"""
        # This would typically involve setting up tray icon and menu
        # Implementation depends on desktop environment
        pass
    
    def _get_user_input(self, prompt: str, default: Any, data_type: type, help_text: str = "") -> Any:
        """Get user input with type conversion and validation"""
        if self.terminal_mode:
            if help_text:
                print(f"  {help_text}")
            full_prompt = f"{prompt} [{default}]: "
            value = input(full_prompt).strip()
            if not value:
                return default
            
            try:
                return data_type(value)
            except ValueError:
                print(f"Invalid input. Using default: {default}")
                return default
        elif rich_available:
            if help_text:
                console.print(f"[dim]{help_text}[/dim]")
            value = Prompt.ask(prompt, default=str(default))
            try:
                return data_type(value)
            except ValueError:
                console.print(f"[yellow]Invalid input. Using default: {default}[/yellow]")
                return default
        else:
            return default
    
    def _get_user_input_bool(self, prompt: str, default: bool, help_text: str = "") -> bool:
        """Get boolean user input"""
        if self.terminal_mode:
            if help_text:
                print(f"  {help_text}")
            response = input(f"{prompt} [{'Y' if default else 'N'}]: ").strip().lower()
            if not response:
                return default
            return response in ['y', 'yes', 'true', '1']
        elif rich_available:
            if help_text:
                console.print(f"[dim]{help_text}[/dim]")
            return Confirm.ask(prompt, default=default)
        else:
            return default
    
    def print_section(self, title: str):
        """Print section header"""
        if rich_available:
            console.print(f"\n[bold blue]═══ {title} ═══[/bold blue]")
        else:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
    
    def run(self):
        """Run the setup wizard"""
        self.display_welcome()
        self.load_existing_config()
        
        # Run configuration steps
        steps = [
            self.setup_capture_preferences,
            self.setup_privacy_security,
            self.setup_user_interface,
            self.setup_llm_integration,
            self.setup_performance_optimization,
            self.setup_integrations
        ]
        
        for step in steps:
            try:
                step()
            except KeyboardInterrupt:
                print("\nSetup cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print(f"Error in {step.__name__}: {e}")
                continue
        
        # Review and confirm
        if not self.review_and_confirm():
            print("Configuration cancelled.")
            sys.exit(0)
        
        # Save configuration
        if self.save_configuration():
            self.create_shortcuts_and_integrations()
            
            # Final message
            if self.terminal_mode:
                print("\n" + "=" * 60)
                print("  Setup Complete!")
                print("=" * 60)
                print("\nContextBox has been configured successfully!")
                print("\nQuick Start:")
                print("1. Run: contextbox --help")
                print("2. Start capturing: contextbox start")
                print("3. Open GUI: contextbox gui")
                print(f"\nConfiguration file: {self.config_path}")
            elif rich_available:
                console.print(Panel.fit(
                    "[bold green]Setup Complete![/bold green]\n\n"
                    "ContextBox has been configured successfully!\n\n"
                    f"Configuration: {self.config_path}\n"
                    "Quick start: contextbox --help",
                    border_style="green"
                ))
            else:
                print("Setup Complete! Configuration saved.")
        else:
            print("Failed to save configuration. Please check permissions.")
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContextBox Setup Wizard")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--gui", action="store_true", help="Force GUI mode")
    parser.add_argument("--terminal", action="store_true", help="Force terminal mode")
    args = parser.parse_args()
    
    try:
        wizard = ContextBoxSetupWizard(config_path=args.config)
        wizard.run()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()