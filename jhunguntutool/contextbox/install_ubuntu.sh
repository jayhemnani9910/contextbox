#!/bin/bash

# ContextBox Ubuntu Installation Script
# Complete installation for Ubuntu/Debian systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on Ubuntu/Debian
check_system() {
    print_status "Checking system..."
    
    if command -v lsb_release &> /dev/null; then
        DISTRO=$(lsb_release -si)
        VERSION=$(lsb_release -sr)
        print_status "Detected: $DISTRO $VERSION"
    elif [ -f /etc/debian_version ]; then
        print_status "Detected: Debian-based system"
    else
        print_warning "Not a Debian-based system. This script is optimized for Ubuntu/Debian."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Core Python development tools
    sudo apt install -y \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        python3-wheel \
        build-essential \
        libpq-dev \
        sqlite3 \
        libsqlite3-dev
    
    # Screenshot tools
    sudo apt install -y \
        gnome-screenshot \
        scrot \
        imagemagick \
        xdg-desktop-portal \
        xdg-desktop-portal-gnome
    
    # Clipboard tools
    sudo apt install -y \
        xclip \
        xsel
    
    # Window management tools
    sudo apt install -y \
        wmctrl \
        xdotool
    
    # OCR dependencies
    sudo apt install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev
    
    # Additional useful tools
    sudo apt install -y \
        curl \
        wget \
        jq \
        tree
    
    print_success "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    # Install ContextBox
    pip install -e .
    
    print_success "Python dependencies installed"
}

# Configure desktop shortcut
setup_hotkey() {
    print_status "Setting up hotkey shortcut..."
    
    # Find the path to the virtual environment Python
    VENV_PYTHON=$(pwd)/.venv/bin/python
    
    # Create desktop entry
    mkdir -p ~/.local/share/applications
    cat > ~/.local/share/applications/contextbox.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ContextBox
Comment=Capture and organize digital context
Icon=applications-accessories
Exec=$VENV_PYTHON -m contextbox %f
Path=$(pwd)
Terminal=true
Categories=Utility;Accessories;
Keywords=screenshot;capture;context;notes;
StartupNotify=true
EOF
    
    # Create bin script
    mkdir -p ~/bin
    cat > ~/bin/contextbox << EOF
#!/bin/bash
cd $(pwd)
source .venv/bin/activate
python -m contextbox "\$@"
EOF
    chmod +x ~/bin/contextbox
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        print_status "Adding ~/bin to PATH in ~/.bashrc"
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        print_warning "Please run 'source ~/.bashrc' or restart your terminal to use 'contextbox' command"
    fi
    
    print_success "Hotkey setup completed"
    print_status "To set up a hotkey:"
    print_status "1. Open Settings → Keyboard → Keyboard Shortcuts"
    print_status "2. Click 'Custom Shortcuts' → '+'"
    print_status "3. Name: ContextBox Capture"
    print_status "4. Command: $VENV_PYTHON -m contextbox capture"
    print_status "5. Press your desired key combination (e.g., Super+Shift+C)"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test basic import
    if python3 -c "import contextbox; print('✓ ContextBox module imported successfully')"; then
        print_success "ContextBox module test passed"
    else
        print_error "ContextBox module test failed"
        return 1
    fi
    
    # Test CLI
    if [ -f ~/bin/contextbox ]; then
        print_success "ContextBox CLI script created"
        print_status "Test with: contextbox --help"
    fi
    
    print_success "All tests passed!"
}

# Create configuration
create_config() {
    print_status "Creating configuration..."
    
    CONFIG_DIR="$HOME/.contextbox"
    mkdir -p "$CONFIG_DIR/media"
    
    # Create default config
    cat > "$CONFIG_DIR/config.yaml" << 'EOF'
# ContextBox Configuration
log_level: INFO

# Capture settings
capture:
  interval: 1.0  # seconds between captures (0 for manual only)
  max_captures: 0  # 0 for unlimited
  enabled_sources:
    - screenshot
    - clipboard
    - active_window

# Database settings  
database:
  path: ~/.contextbox/context.db
  backup_enabled: true
  backup_interval: 3600  # seconds

# Extractor settings
extractors:
  enabled_extractors:
    - ocr
    - url_extraction
    - clipboard_url
  confidence_threshold: 0.5

# LLM settings (optional)
llm:
  provider: local  # local, openai, anthropic
  model: llama3.1  # local model name
  api_key: null  # for hosted APIs
  max_tokens: 1000
  
# Privacy settings
security:
  encrypt_stored_data: false
  retention_days: 30
  auto_cleanup: true
  
# UI settings
ui:
  notifications: true
  theme: dark
  show_in_tray: true
EOF
    
    print_success "Configuration created at $CONFIG_DIR/config.yaml"
}

# Main installation
main() {
    print_status "ContextBox Ubuntu Installation"
    print_status "==============================="
    
    # Check if running from project root
    if [ ! -f "setup.py" ] || [ ! -d "contextbox" ]; then
        print_error "Please run this script from the ContextBox project root directory"
        print_error "Make sure setup.py and contextbox/ directory are present"
        exit 1
    fi
    
    check_system
    install_system_deps
    install_python_deps
    setup_hotkey
    create_config
    test_installation
    
    print_success "ContextBox installation completed successfully!"
    echo ""
    print_status "Next steps:"
    print_status "1. Restart terminal or run: source ~/.bashrc"
    print_status "2. Test with: contextbox --help"
    print_status "3. Set up hotkey in Settings → Keyboard → Keyboard Shortcuts"
    print_status "4. Try: contextbox capture"
    echo ""
    print_status "Documentation: README.md"
    print_status "Configuration: ~/.contextbox/config.yaml"
}

# Handle arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Ubuntu Installation Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "This script installs ContextBox on Ubuntu/Debian systems"
        echo "with all necessary system dependencies."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac