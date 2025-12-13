# ContextBox Installation Scripts - Enhanced Features

This document describes the enhanced installation system for ContextBox with comprehensive dependency checking, robust installation, and user-friendly setup.

## 🚀 What's New

### 1. Enhanced Requirements.txt

**Location:** `requirements.txt`

**Added Dependencies:**
- **notify2** - Desktop notifications (Linux)
- **plyer** - Cross-platform notifications  
- **pynput** - Global hotkey support
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **cryptography** - Data encryption
- **keyring** - Secure credential storage
- **httpx** - Modern HTTP client
- **aiofiles** - Async file operations

**Benefits:**
- Complete desktop integration
- Better security and encryption
- Enhanced data processing capabilities
- Modern async support
- Cross-platform notification system

### 2. Robust Uninstall Script

**Location:** `uninstall.sh`

**Features:**
- **Complete removal** of all ContextBox files and directories
- **Stops running processes** gracefully
- **Removes configuration** files (`~/.config/contextbox`, `~/.contextbox`)
- **Cleans desktop entries** and shortcuts
- **Removes virtual environments** and Python packages
- **Cleans database files** and cache
- **Removes system integration** (autostart, systemd)
- **Generates uninstall report** for transparency
- **Backup support** with restore options

**Usage:**
```bash
# Interactive uninstall (recommended)
./uninstall.sh

# Force uninstall (skip confirmation)
./uninstall.sh --force

# Show help
./uninstall.sh --help
```

### 3. Enhanced Installation Script

**Location:** `install_enhanced.sh`

**Key Improvements:**

#### Dependency Checking
- **Automatic OS detection** (Ubuntu, Debian, CentOS, Fedora, Arch, macOS)
- **Package verification** with automatic installation
- **Missing dependency detection** with installation suggestions
- **System resource checking** (memory, disk space, CPU)

#### Error Handling & Recovery
- **Comprehensive error handling** with detailed logging
- **Backup creation** before installation
- **Recovery mode** for fixing broken installations
- **Rollback capability** on failure
- **Installation state tracking**

#### Enhanced Configuration
- **Advanced configuration template** with version 2.0 schema
- **Performance optimization options**
- **Security settings** (encryption, privacy mode)
- **UI customization** (theme, hotkeys, notifications)
- **LLM integration** settings

#### Testing & Validation
- **Pre-installation tests**
- **Post-installation validation**
- **Module import testing**
- **CLI command testing**
- **Configuration file verification**

**Usage:**
```bash
# Standard installation
./install_enhanced.sh

# Recovery mode
./install_enhanced.sh --recovery

# Skip dependency check
./install_enhanced.sh --no-deps

# Dry run (show what would be done)
./install_enhanced.sh --dry-run

# Show help
./install_enhanced.sh --help
```

### 4. Post-Installation Setup Wizard

**Location:** `setup_wizard.py`

**Features:**

#### Interactive Configuration
- **Terminal, Rich, or GUI mode** based on available libraries
- **Step-by-step wizard** with clear navigation
- **Default value suggestions** with helpful descriptions
- **Input validation** and type checking

#### Configuration Sections

**Capture Preferences:**
- Capture interval and limits
- Enabled sources (screenshot, clipboard, window, files)
- Screenshot format and quality settings

**Privacy & Security:**
- Data encryption toggle
- Retention period configuration
- Auto-cleanup settings
- Privacy mode for external service blocking

**User Interface:**
- Theme selection (dark/light)
- System tray integration
- Notification preferences
- Global hotkey configuration

**LLM Integration:**
- Provider selection (local, OpenAI, Anthropic, Ollama)
- Model configuration
- API key setup
- Token limits and temperature

**Performance Optimization:**
- Speed vs. resource optimization
- Cache size configuration
- Memory usage limits
- Background processing options

**System Integrations:**
- Desktop shortcuts creation
- System tray integration
- Auto-start configuration
- Browser extension support

#### Configuration Review
- **Visual summary** of all settings
- **Confirmation before applying**
- **Dual format saving** (JSON + YAML)

**Usage:**
```bash
# Interactive setup wizard
python3 setup_wizard.py

# Force GUI mode
python3 setup_wizard.py --gui

# Force terminal mode
python3 setup_wizard.py --terminal

# Custom config path
python3 setup_wizard.py --config /path/to/config.json

# Show help
python3 setup_wizard.py --help
```

### 5. Comprehensive Dependency Checker

**Location:** `check_dependencies.sh`

**Features:**

#### System Detection
- **Automatic OS and distribution detection**
- **Package manager identification** (apt, yum, pacman, brew)
- **Version detection** for all components

#### Dependency Categories

**Python Environment:**
- Python 3.7+ version check
- pip availability and version
- Virtual environment support
- Development headers verification

**Build Tools:**
- GCC, G++, Make, CMake
- pkg-config
- Development libraries (libpq, libsqlite3, libffi)

**GUI Dependencies:**
- Display environment detection (X11/Wayland)
- Screenshot tools (gnome-screenshot, scrot, imagemagick)
- Clipboard tools (xclip, xsel, wl-copy)
- Window management (wmctrl, xdotool)
- Notification systems (dunst, notify-osd, gnome-shell)

**OCR Dependencies:**
- Tesseract OCR installation
- Language pack verification
- Dependencies for image processing

**Networking:**
- curl, wget, jq tools
- Internet connectivity testing
- HTTP client libraries

**Optional Dependencies:**
- Multimedia tools (ffmpeg, imagemagick)
- Development tools (git, tree, htop)
- Documentation tools

**System Resources:**
- Memory availability checking
- Disk space verification
- CPU core detection
- Permission checking

#### Auto-Fix Capability
- **Automatic installation** of missing packages
- **Distribution-specific commands**
- **Progress tracking** and error reporting

**Usage:**
```bash
# Standard dependency check
./check_dependencies.sh

# Auto-fix missing dependencies
./check_dependencies.sh --auto-fix

# Verbose output
./check_dependencies.sh --verbose

# JSON output (future feature)
./check_dependencies.sh --json

# Show help
./check_dependencies.sh --help
```

### 6. Master Installer Script

**Location:** `install_master.sh`

**Features:**

#### Complete Installation Workflow
1. **Dependency checking** with comprehensive validation
2. **Enhanced installation** with error handling
3. **Setup wizard** for user configuration
4. **Final integration** and testing

#### Installation Modes
- **Interactive mode** (default) - guides user through process
- **Auto mode** - non-interactive for scripts
- **Recovery mode** - fixes broken installations
- **Selective installation** - skip dependency check or setup

#### Backup & Recovery
- **Automatic backup** of existing installations
- **Rollback on failure** with restoration
- **Comprehensive logging** for troubleshooting

#### Integration Features
- **Desktop database updates**
- **System tray configuration**
- **Autostart setup**
- **PATH configuration**

**Usage:**
```bash
# Master installer (recommended)
./install_master.sh

# Non-interactive installation
./install_master.sh --auto

# Skip dependency check
./install_master.sh --skip-deps

# Skip setup wizard
./install_master.sh --skip-setup

# Recovery mode
./install_master.sh --recovery

# Run uninstaller
./install_master.sh --uninstall

# Show help
./install_master.sh --help
```

## 📋 File Overview

```
contextbox/
├── requirements.txt           # Enhanced dependencies
├── install.sh                 # Original basic installer
├── install_enhanced.sh        # Enhanced installer with recovery
├── install_master.sh          # Complete installer workflow
├── uninstall.sh              # Complete uninstaller
├── check_dependencies.sh      # Comprehensive dependency checker
├── setup_wizard.py           # Interactive setup wizard
└── install_ubuntu.sh         # Ubuntu-specific installer
```

## 🎯 Usage Recommendations

### For New Users
```bash
# Use the master installer for complete setup
./install_master.sh
```

### For Developers
```bash
# Use enhanced installer with recovery mode
./install_enhanced.sh --recovery
```

### For System Administrators
```bash
# Check dependencies first
./check_dependencies.sh --auto-fix

# Then install
./install_master.sh --auto
```

### For Custom Installations
```bash
# Skip setup wizard for custom configuration
./install_master.sh --skip-setup
```

### For Troubleshooting
```bash
# Check system dependencies
./check_dependencies.sh --verbose

# Run recovery mode
./install_master.sh --recovery
```

### For Complete Removal
```bash
# Complete uninstallation
./uninstall.sh

# Force uninstall (skip confirmation)
./uninstall.sh --force
```

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Set custom installation directory
export CONTEXTBOX_HOME=/opt/contextbox

# Enable debug logging
export CONTEXTBOX_DEBUG=1

# Skip interactive prompts
export CONTEXTBOX_AUTO=1
```

### Configuration Files
- **JSON format:** `~/.config/contextbox/config.json`
- **YAML format:** `~/.config/contextbox/config.yaml`
- **Backup location:** `/tmp/contextbox_backup_TIMESTAMP/`

### Logs and Debugging
- **Installation log:** `/tmp/contextbox_install.log`
- **Dependency check log:** `/tmp/contextbox_deps_check.log`
- **Master install log:** `/tmp/contextbox_master_install.log`

## 🛠️ Troubleshooting

### Common Issues

**1. Permission Errors**
```bash
# Fix permissions
sudo chown -R $USER:$USER ~/.config/contextbox
chmod +x *.sh *.py
```

**2. Python Version Issues**
```bash
# Check Python version
python3 --version

# Install specific version
sudo apt install python3.8 python3.8-dev
```

**3. Missing System Dependencies**
```bash
# Run dependency checker
./check_dependencies.sh --auto-fix

# Manual installation
sudo apt update && sudo apt install python3-dev build-essential
```

**4. Virtual Environment Issues**
```bash
# Clean virtual environment
rm -rf venv_contextbox

# Recreate
python3 -m venv venv_contextbox
source venv_contextbox/bin/activate
pip install -r requirements.txt
```

**5. Configuration Issues**
```bash
# Reset configuration
rm ~/.config/contextbox/config.json

# Run setup wizard
python3 setup_wizard.py
```

### Recovery Procedures

**1. Complete Reset**
```bash
# Uninstall and reinstall
./uninstall.sh --force
./install_master.sh
```

**2. Fix Dependencies**
```bash
# Auto-fix dependencies
./check_dependencies.sh --auto-fix

# Manual dependency installation
sudo apt update
sudo apt install python3-dev python3-pip build-essential
```

**3. Recovery Mode Installation**
```bash
# Use recovery mode
./install_master.sh --recovery
```

## 📊 Testing

All installation scripts include comprehensive testing:

- **Pre-installation tests** - System validation
- **Installation tests** - Package verification
- **Post-installation tests** - Functionality validation
- **Configuration tests** - Settings verification

Test commands:
```bash
# Test installation
contextbox --help

# Test configuration
contextbox config show

# Test capture (if enabled)
contextbox capture --test
```

## 🔐 Security Features

- **Encrypted storage** option for sensitive data
- **Privacy mode** to prevent external data transmission
- **Secure credential storage** using keyring
- **Automatic cleanup** of temporary files
- **Permission validation** before installation

## 🌟 Best Practices

1. **Always backup** before major changes
2. **Run dependency checker** first on new systems
3. **Use master installer** for first-time setup
4. **Review configuration** before applying
5. **Test installation** after completion
6. **Keep logs** for troubleshooting

## 📚 Additional Resources

- **Documentation:** README.md
- **Configuration Guide:** `~/.config/contextbox/config.yaml`
- **Troubleshooting:** Check log files in `/tmp/`
- **Support:** https://github.com/contextbox/contextbox/issues

---

**Installation System Version:** 2.0  
**Last Updated:** November 2025  
**Compatibility:** Linux, macOS, Windows (via WSL)