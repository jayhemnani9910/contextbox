#!/bin/bash

# ContextBox Installation Script
# This script sets up ContextBox on Unix-like systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.7+ is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 7) else 1)'; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
            PIP_CMD="pip3"
        else
            print_error "Python 3.7 or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python -c 'import sys; exit(0 if sys.version_info >= (3, 7) else 1)'; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
            PIP_CMD="pip"
        else
            print_error "Python 3.7 or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.7 or higher."
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    print_status "Checking pip..."
    
    if command -v $PIP_CMD &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip not found. Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm get-pip.py
        print_success "pip installed"
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Checking system dependencies..."
    
    OS=$(uname -s)
    
    case $OS in
        "Linux")
            # Check for common package managers
            if command -v apt-get &> /dev/null; then
                print_status "Detected Debian/Ubuntu system"
                print_warning "You may need to install system dependencies manually:"
                print_warning "  sudo apt-get update"
                print_warning "  sudo apt-get install python3-dev python3-pip build-essential"
            elif command -v yum &> /dev/null; then
                print_status "Detected Red Hat/CentOS system"
                print_warning "You may need to install system dependencies manually:"
                print_warning "  sudo yum groupinstall 'Development Tools'"
                print_warning "  sudo yum install python3-devel"
            elif command -v pacman &> /dev/null; then
                print_status "Detected Arch Linux system"
                print_warning "You may need to install system dependencies manually:"
                print_warning "  sudo pacman -S python python-pip base-devel"
            fi
            ;;
        "Darwin")
            print_status "Detected macOS"
            print_warning "For full functionality, consider installing dependencies via:"
            print_warning "  brew install python"
            print_warning "  brew install python-tk  # for GUI features"
            ;;
        *)
            print_warning "Unknown operating system: $OS"
            print_warning "Please ensure you have all necessary development tools installed."
            ;;
    esac
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    $PIP_CMD install --upgrade pip
    
    # Install basic dependencies
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        $PIP_CMD install -r requirements.txt
    fi
    
    # Install additional dependencies if available
    if [ -f "requirements-dev.txt" ]; then
        print_status "Installing development dependencies..."
        $PIP_CMD install -r requirements-dev.txt
    fi
    
    # Install ContextBox in development mode
    print_status "Installing ContextBox in development mode..."
    $PIP_CMD install -e .
    
    print_success "Python dependencies installed"
}

# Create default configuration
create_config() {
    print_status "Creating default configuration..."
    
    CONFIG_DIR="$HOME/.config/contextbox"
    mkdir -p "$CONFIG_DIR"
    
    if [ ! -f "$CONFIG_DIR/config.json" ]; then
        cat > "$CONFIG_DIR/config.json" << 'EOF'
{
  "log_level": "INFO",
  "capture": {
    "interval": 1.0,
    "max_captures": 0,
    "enabled_sources": ["clipboard", "active_window", "recent_files"]
  },
  "database": {
    "path": "contextbox.db",
    "backup_enabled": true,
    "backup_interval": 3600
  },
  "extractors": {
    "enabled_extractors": ["text", "system", "network"],
    "confidence_threshold": 0.5
  },
  "security": {
    "encrypt_stored_data": false,
    "retention_days": 30,
    "auto_cleanup": true
  }
}
EOF
        print_success "Default configuration created at $CONFIG_DIR/config.json"
    else
        print_warning "Configuration already exists at $CONFIG_DIR/config.json"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test CLI
    if command -v contextbox &> /dev/null; then
        print_success "ContextBox CLI installed successfully"
        contextbox --version
    else
        print_warning "ContextBox CLI not found in PATH"
        print_warning "You may need to add the Python bin directory to your PATH"
    fi
    
    # Test Python module
    if $PYTHON_CMD -c "import contextbox; print('ContextBox module imported successfully')" 2>/dev/null; then
        print_success "ContextBox Python module works"
    else
        print_error "ContextBox Python module failed to import"
        exit 1
    fi
}

# Print usage information
print_usage() {
    print_success "ContextBox installation completed!"
    echo ""
    print_status "Usage:"
    echo "  contextbox --help              Show help"
    echo "  contextbox --version           Show version"
    echo "  contextbox start               Start context capture"
    echo "  contextbox stop                Stop context capture"
    echo "  contextbox extract <file>      Extract context from file"
    echo "  contextbox query <id>          Query stored context"
    echo ""
    print_status "Configuration:"
    echo "  Configuration file: $HOME/.config/contextbox/config.json"
    echo "  Database file: contextbox.db (in current directory)"
    echo ""
    print_status "For more information, see: https://contextbox.readthedocs.io/"
}

# Main installation function
main() {
    print_status "ContextBox Installation Script"
    print_status "================================"
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -d "contextbox" ]; then
        print_error "This script must be run from the ContextBox project root directory"
        print_error "Please ensure setup.py and contextbox/ directory are present"
        exit 1
    fi
    
    # Run installation steps
    check_python
    check_pip
    install_system_deps
    install_python_deps
    create_config
    test_installation
    print_usage
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Installation Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "This script installs ContextBox and its dependencies."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac