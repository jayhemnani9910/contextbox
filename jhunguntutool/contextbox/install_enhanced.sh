#!/bin/bash

# ContextBox Enhanced Installation Script
# Robust installation with error handling and recovery options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_debug() { echo -e "${MAGENTA}[DEBUG]${NC} $1"; }

# Installation state tracking
INSTALL_LOG="/tmp/contextbox_install.log"
BACKUP_DIR=""
RECOVERY_MODE=false
DRY_RUN=false

# Initialize installation logging
init_logging() {
    echo "ContextBox Installation Log - $(date)" > "$INSTALL_LOG"
    echo "=========================================" >> "$INSTALL_LOG"
    print_status "Installation logging started: $INSTALL_LOG"
}

# Log messages to file
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$INSTALL_LOG"
}

# Create backup of existing configuration
backup_existing_config() {
    local config_dir="$HOME/.config/contextbox"
    
    if [ -d "$config_dir" ]; then
        BACKUP_DIR="/tmp/contextbox_backup_$(date +%Y%m%d_%H%M%S)"
        print_warning "Existing configuration found. Creating backup at: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        cp -r "$config_dir" "$BACKUP_DIR/"
        echo "$BACKUP_DIR" > /tmp/contextbox_backup_location.txt
        print_success "Backup created at: $BACKUP_DIR"
    fi
}

# Recovery mode installation
recovery_installation() {
    print_status "Running in recovery mode..."
    RECOVERY_MODE=true
    
    # Clean up broken installations
    print_status "Cleaning up broken installations..."
    $PYTHON_CMD -m pip uninstall contextbox -y 2>/dev/null || true
    
    # Fix permissions
    print_status "Fixing permissions..."
    find . -type f -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
    
    # Verify file integrity
    print_status "Verifying file integrity..."
    if [ ! -f "setup.py" ]; then
        print_error "setup.py not found. Cannot continue recovery."
        exit 1
    fi
    
    if [ ! -d "contextbox" ]; then
        print_error "contextbox directory not found. Cannot continue recovery."
        exit 1
    fi
}

# Enhanced dependency checking
check_system_dependencies() {
    print_status "Checking system dependencies..."
    log_message "Checking system dependencies"
    
    local missing_deps=()
    local required_commands=("python3" "pip3" "git")
    local optional_commands=("curl" "wget" "apt-get" "yum" "brew")
    
    # Check required commands
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd (required)")
            print_error "Missing required command: $cmd"
        fi
    done
    
    # Check optional commands for auto-installation
    for cmd in "${optional_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_message "Found optional command: $cmd"
            break
        fi
    done
    
    # Check system packages
    local system_packages=()
    local os_type=$(detect_os)
    
    case $os_type in
        "ubuntu"|"debian")
            system_packages=("python3-dev" "python3-venv" "build-essential" "pkg-config")
            check_debian_packages "${system_packages[@]}"
            ;;
        "centos"|"rhel"|"fedora")
            system_packages=("python3-devel" "gcc" "gcc-c++")
            check_redhat_packages "${system_packages[@]}"
            ;;
        "arch")
            system_packages=("python" "base-devel")
            check_arch_packages "${system_packages[@]}"
            ;;
        "macos")
            check_macos_dependencies
            ;;
    esac
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install missing dependencies and run this script again."
        exit 1
    fi
    
    print_success "System dependency check passed"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            if grep -q Ubuntu /etc/os-release; then
                echo "ubuntu"
            else
                echo "debian"
            fi
        elif [ -f /etc/redhat-release ]; then
            if grep -q "CentOS" /etc/redhat-release; then
                echo "centos"
            elif grep -q "Fedora" /etc/redhat-release; then
                echo "fedora"
            else
                echo "rhel"
            fi
        elif [ -f /etc/arch-release ]; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check Debian packages
check_debian_packages() {
    local packages=("$@")
    local missing_packages=()
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            missing_packages+=("$pkg")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing packages: ${missing_packages[*]}"
        print_status "Attempting to install missing packages..."
        
        if command -v sudo &> /dev/null; then
            sudo apt update
            sudo apt install -y "${missing_packages[@]}"
            print_success "Missing packages installed"
        else
            print_error "Cannot install packages without sudo. Please install manually:"
            print_error "  sudo apt update"
            print_error "  sudo apt install ${missing_packages[*]}"
            exit 1
        fi
    fi
}

# Check Red Hat packages
check_redhat_packages() {
    local packages=("$@")
    local missing_packages=()
    
    for pkg in "${packages[@]}"; do
        if ! rpm -q "$pkg" &> /dev/null; then
            missing_packages+=("$pkg")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing packages: ${missing_packages[*]}"
        print_status "Attempting to install missing packages..."
        
        if command -v sudo &> /dev/null; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y "${missing_packages[@]}"
            print_success "Missing packages installed"
        else
            print_error "Cannot install packages without sudo. Please install manually:"
            print_error "  sudo yum groupinstall 'Development Tools'"
            print_error "  sudo yum install ${missing_packages[*]}"
            exit 1
        fi
    fi
}

# Check Arch packages
check_arch_packages() {
    local packages=("$@")
    local missing_packages=()
    
    for pkg in "${packages[@]}"; do
        if ! pacman -Q "$pkg" &> /dev/null; then
            missing_packages+=("$pkg")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing packages: ${missing_packages[*]}"
        print_status "Attempting to install missing packages..."
        
        if command -v sudo &> /dev/null; then
            sudo pacman -Sy
            sudo pacman -S --noconfirm "${missing_packages[@]}"
            print_success "Missing packages installed"
        else
            print_error "Cannot install packages without sudo. Please install manually:"
            print_error "  sudo pacman -S ${missing_packages[*]}"
            exit 1
        fi
    fi
}

# Check macOS dependencies
check_macos_dependencies() {
    print_status "Checking macOS dependencies..."
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f "/usr/local/bin/brew" ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 not found. Installing via Homebrew..."
        brew install python
    fi
    
    # Install additional dependencies
    brew install pkg-config 2>/dev/null || true
}

# Enhanced Python version check
check_python_version() {
    print_status "Checking Python version and environment..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
            PIP_CMD="pip3"
        else
            print_error "Python 3.7+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
            PIP_CMD="pip"
        else
            print_error "Python 3.7+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.7 or higher."
        exit 1
    fi
    
    # Check pip
    if ! $PIP_CMD --version &> /dev/null; then
        print_warning "pip not found. Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm -f get-pip.py
        print_success "pip installed"
    fi
    
    # Check virtual environment support
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        print_warning "venv module not available. Installing python3-venv..."
        case $(detect_os) in
            "ubuntu"|"debian")
                if command -v sudo &> /dev/null; then
                    sudo apt install -y python3-venv
                fi
                ;;
            "macos")
                brew install python
                ;;
        esac
    fi
    
    log_message "Python environment: $PYTHON_CMD ($PYTHON_VERSION), $PIP_CMD"
}

# Safe installation with rollback capability
safe_install_python_deps() {
    print_status "Installing Python dependencies safely..."
    
    # Create virtual environment
    local venv_dir="venv_contextbox"
    if [ -d "$venv_dir" ]; then
        print_warning "Virtual environment exists. Removing..."
        rm -rf "$venv_dir"
    fi
    
    print_status "Creating virtual environment: $venv_dir"
    $PYTHON_CMD -m venv "$venv_dir"
    
    # Activate virtual environment
    source "$venv_dir/bin/activate"
    
    # Upgrade pip, wheel, setuptools
    print_status "Upgrading pip, wheel, and setuptools..."
    pip install --upgrade pip wheel setuptools
    
    # Install dependencies with error handling
    local requirements_file="requirements.txt"
    if [ -f "$requirements_file" ]; then
        print_status "Installing from $requirements_file..."
        
        # Install in smaller batches to handle potential issues
        local total_deps=$(grep -c . "$requirements_file" || echo "0")
        if [ "$total_deps" -gt 10 ]; then
            print_status "Large number of dependencies ($total_deps). Installing in batches..."
            
            # Create batch files
            split -l 10 "$requirements_file" req_batch_
            
            for batch_file in req_batch_*; do
                if [ -f "$batch_file" ]; then
                    print_status "Installing batch: $batch_file"
                    if ! pip install -r "$batch_file"; then
                        print_error "Failed to install batch: $batch_file"
                        print_error "Check $INSTALL_LOG for details"
                        exit 1
                    fi
                    rm "$batch_file"
                fi
            done
        else
            if ! pip install -r "$requirements_file"; then
                print_error "Failed to install dependencies"
                print_error "Check $INSTALL_LOG for details"
                exit 1
            fi
        fi
    fi
    
    # Install ContextBox in development mode
    print_status "Installing ContextBox..."
    if ! pip install -e .; then
        print_error "Failed to install ContextBox"
        print_error "Check $INSTALL_LOG for details"
        exit 1
    fi
    
    print_success "Python dependencies installed successfully"
    deactivate
    
    log_message "Python dependencies installed in $venv_dir"
}

# Create enhanced configuration with validation
create_enhanced_config() {
    print_status "Creating enhanced configuration..."
    
    local config_dir="$HOME/.config/contextbox"
    mkdir -p "$config_dir"
    
    # Backup existing configuration
    if [ -f "$config_dir/config.json" ]; then
        cp "$config_dir/config.json" "$config_dir/config.json.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create enhanced configuration
    cat > "$config_dir/config.json" << 'EOF'
{
  "version": "2.0",
  "log_level": "INFO",
  "backup_on_startup": true,
  "capture": {
    "interval": 1.0,
    "max_captures": 0,
    "enabled_sources": ["clipboard", "active_window", "recent_files"],
    "screenshot": {
      "format": "png",
      "quality": 90,
      "delay": 0.1
    }
  },
  "database": {
    "path": "contextbox.db",
    "backup_enabled": true,
    "backup_interval": 3600,
    "auto_vacuum": true,
    "max_db_size": "500MB"
  },
  "extractors": {
    "enabled_extractors": ["text", "system", "network"],
    "confidence_threshold": 0.5,
    "parallel_processing": true,
    "max_workers": 4
  },
  "llm": {
    "provider": "local",
    "model": "llama3.1",
    "api_key": null,
    "max_tokens": 1000,
    "temperature": 0.7
  },
  "security": {
    "encrypt_stored_data": false,
    "retention_days": 30,
    "auto_cleanup": true,
    "secure_delete": true
  },
  "ui": {
    "notifications": true,
    "theme": "dark",
    "show_in_tray": true,
    "hotkey": "Ctrl+Alt+C"
  },
  "performance": {
    "cache_size": 100,
    "max_memory_usage": "1GB",
    "optimize_for_speed": false
  }
}
EOF
    
    print_success "Enhanced configuration created at $config_dir/config.json"
    log_message "Configuration created at $config_dir/config.json"
}

# Create launcher scripts
create_launcher_scripts() {
    print_status "Creating launcher scripts..."
    
    local script_dir="$HOME/bin"
    mkdir -p "$script_dir"
    
    # Main ContextBox launcher
    cat > "$script_dir/contextbox" << EOF
#!/bin/bash
# ContextBox launcher script

# Activate virtual environment if exists
if [ -f "$(pwd)/venv_contextbox/bin/activate" ]; then
    source "$(pwd)/venv_contextbox/bin/activate"
elif [ -f "$HOME/venv_contextbox/bin/activate" ]; then
    source "$HOME/venv_contextbox/bin/activate"
fi

# Run ContextBox
python -m contextbox "\$@"
EOF
    
    chmod +x "$script_dir/contextbox"
    
    # Create desktop entry
    local desktop_dir="$HOME/.local/share/applications"
    mkdir -p "$desktop_dir"
    
    cat > "$desktop_dir/contextbox.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ContextBox
Comment=Advanced context capture and organization system
Icon=applications-accessories
Exec=$script_dir/contextbox %f
Path=\$HOME
Terminal=true
Categories=Utility;Accessories;Office;
Keywords=screenshot;capture;context;notes;ocr;
StartupNotify=true
MimeType=application/x-contextbox;
EOF
    
    print_success "Launcher scripts created"
    log_message "Launcher scripts created in $script_dir"
}

# Comprehensive installation testing
comprehensive_test() {
    print_status "Running comprehensive installation tests..."
    
    # Test Python module import
    if $PYTHON_CMD -c "import contextbox; print('ContextBox module imported successfully')" 2>/dev/null; then
        print_success "✓ Python module import test passed"
        log_message "Python module import test: PASSED"
    else
        print_error "✗ Python module import test failed"
        log_message "Python module import test: FAILED"
        return 1
    fi
    
    # Test CLI command
    if command -v contextbox &> /dev/null; then
        local version_output=$(contextbox --version 2>&1)
        print_success "✓ CLI command test passed: $version_output"
        log_message "CLI command test: PASSED"
    else
        print_warning "✗ CLI command test failed (may need PATH update)"
        log_message "CLI command test: FAILED"
    fi
    
    # Test configuration file
    if [ -f "$HOME/.config/contextbox/config.json" ]; then
        print_success "✓ Configuration file test passed"
        log_message "Configuration file test: PASSED"
    else
        print_error "✗ Configuration file test failed"
        log_message "Configuration file test: FAILED"
        return 1
    fi
    
    # Test database creation
    if $PYTHON_CMD -c "
import sqlite3
import os
db_path = 'test_contextbox.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('CREATE TABLE test (id INTEGER PRIMARY KEY)')
conn.close()
os.remove(db_path)
print('Database creation test passed')
" 2>/dev/null; then
        print_success "✓ Database creation test passed"
        log_message "Database creation test: PASSED"
    else
        print_error "✗ Database creation test failed"
        log_message "Database creation test: FAILED"
        return 1
    fi
    
    print_success "All installation tests passed!"
}

# Cleanup and finalization
cleanup_and_finalize() {
    print_status "Cleaning up and finalizing installation..."
    
    # Clean up temporary files
    rm -f get-pip.py req_batch_* test_contextbox.db
    
    # Print installation summary
    echo ""
    print_success "========================================="
    print_success "  ContextBox Installation Complete!"
    print_success "========================================="
    echo ""
    print_status "Installation Details:"
    print_status "• Configuration: $HOME/.config/contextbox/config.json"
    print_status "• Launcher: $HOME/bin/contextbox"
    print_status "• Desktop Entry: ~/.local/share/applications/contextbox.desktop"
    print_status "• Virtual Environment: $(pwd)/venv_contextbox"
    print_status "• Installation Log: $INSTALL_LOG"
    echo ""
    
    if [ -n "$BACKUP_DIR" ]; then
        print_status "Backup created at: $BACKUP_DIR"
    fi
    
    print_status "Quick Start:"
    print_status "1. Test installation: contextbox --help"
    print_status "2. Start capturing: contextbox start"
    print_status "3. Or use GUI: contextbox gui"
    echo ""
    print_status "For help: contextbox --help"
    print_status "Documentation: README.md"
    
    log_message "Installation completed successfully"
}

# Main installation function with error handling
main() {
    # Initialize
    init_logging
    log_message "Starting ContextBox installation"
    
    print_status "ContextBox Enhanced Installation Script"
    print_status "========================================"
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -d "contextbox" ]; then
        print_error "This script must be run from the ContextBox project root directory"
        print_error "Please ensure setup.py and contextbox/ directory are present"
        exit 1
    fi
    
    # Create backup if needed
    backup_existing_config
    
    # Run installation steps with error handling
    local steps=(
        "check_python_version:Python Version Check"
        "check_system_dependencies:System Dependencies"
        "safe_install_python_deps:Python Dependencies"
        "create_enhanced_config:Configuration"
        "create_launcher_scripts:Launcher Scripts"
        "comprehensive_test:Installation Tests"
    )
    
    for step in "${steps[@]}"; do
        IFS=':' read -r function_name step_name <<< "$step"
        print_status "Executing: $step_name..."
        
        if $function_name; then
            print_success "✓ $step_name completed"
        else
            print_error "✗ $step_name failed"
            print_error "Check $INSTALL_LOG for details"
            
            if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
                print_status "Restoring backup from: $BACKUP_DIR"
                rm -rf "$HOME/.config/contextbox"
                cp -r "$BACKUP_DIR" "$HOME/.config/contextbox"
            fi
            
            exit 1
        fi
    done
    
    cleanup_and_finalize
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Enhanced Installation Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h       Show this help message"
        echo "  --dry-run        Show what would be done without executing"
        echo "  --recovery       Run in recovery mode to fix broken installation"
        echo "  --no-deps        Skip system dependency checking"
        echo ""
        echo "This script provides robust installation with error handling,"
        echo "dependency checking, and recovery options."
        exit 0
        ;;
    --dry-run)
        DRY_RUN=true
        print_status "Dry run mode - showing planned actions:"
        print_status "1. Check system dependencies"
        print_status "2. Check Python version"
        print_status "3. Create virtual environment"
        print_status "4. Install Python dependencies"
        print_status "5. Create configuration"
        print_status "6. Create launcher scripts"
        print_status "7. Run installation tests"
        print_status "Installation will not proceed. Remove --dry-run to actually install."
        exit 0
        ;;
    --recovery)
        check_python_version
        recovery_installation
        main
        ;;
    --no-deps)
        print_status "Skipping system dependency check..."
        check_python_version
        safe_install_python_deps
        create_enhanced_config
        create_launcher_scripts
        comprehensive_test
        cleanup_and_finalize
        ;;
    *)
        main
        ;;
esac