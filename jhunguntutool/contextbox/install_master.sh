#!/bin/bash

# ContextBox Master Installer
# Complete installation with dependency checking, enhanced installation, and setup wizard

set -e

# Colors and styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_LOG="/tmp/contextbox_master_install.log"
BACKUP_DIR=""
AUTO_MODE=false
SKIP_DEPS=false
SKIP_SETUP=false
RECOVERY_MODE=false

# Print functions
print_banner() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                    ${BOLD}ContextBox Master Installer${NC}${CYAN}                    ║${NC}"
    echo -e "${CYAN}║${NC}             Advanced Context Capture System Installer${CYAN}           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${MAGENTA}[INFO]${NC} $1"; }

# Logging
init_logging() {
    echo "ContextBox Master Installer - $(date)" > "$INSTALL_LOG"
    echo "=============================================" >> "$INSTALL_LOG"
    echo "Arguments: $@" >> "$INSTALL_LOG"
    echo "User: $(whoami)" >> "$INSTALL_LOG"
    echo "System: $(uname -a)" >> "$INSTALL_LOG"
    echo "" >> "$INSTALL_LOG"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$INSTALL_LOG"
}

# Check if running in correct directory
check_installation_directory() {
    if [ ! -f "$SCRIPT_DIR/setup.py" ] || [ ! -d "$SCRIPT_DIR/contextbox" ]; then
        print_error "This script must be run from the ContextBox project root directory"
        print_error "Please ensure setup.py and contextbox/ directory are present"
        echo ""
        print_info "Current directory: $SCRIPT_DIR"
        print_info "Looking for: setup.py and contextbox/ directory"
        exit 1
    fi
    print_success "Installation directory verified"
}

# Display welcome message and options
show_welcome() {
    print_banner
    
    if [ "$AUTO_MODE" = false ]; then
        echo -e "${BOLD}Welcome to ContextBox installation!${NC}"
        echo ""
        echo "This installer will:"
        echo "  1. Check system dependencies"
        echo "  2. Install ContextBox with enhanced features"
        echo "  3. Configure your preferences"
        echo "  4. Set up system integration"
        echo ""
        
        echo -e "${BOLD}Installation options:${NC}"
        echo "  • Standard installation (recommended)"
        echo "  • Skip dependency checking (if already done)"
        echo "  • Skip setup wizard (use defaults)"
        echo "  • Recovery mode (fix broken installation)"
        echo ""
        
        read -p "Proceed with installation? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "Installation cancelled."
            exit 0
        fi
        echo ""
    fi
}

# Create backup of existing installation
create_backup() {
    local contextbox_dirs=(
        "$HOME/.config/contextbox"
        "$HOME/.contextbox"
        "$(pwd)/venv_contextbox"
    )
    
    local backup_needed=false
    for dir in "${contextbox_dirs[@]}"; do
        if [ -d "$dir" ] || [ -f "$HOME/bin/contextbox" ]; then
            backup_needed=true
            break
        fi
    done
    
    if [ "$backup_needed" = true ]; then
        BACKUP_DIR="/tmp/contextbox_backup_$(date +%Y%m%d_%H%M%S)"
        print_warning "Existing ContextBox installation detected."
        print_info "Creating backup at: $BACKUP_DIR"
        
        mkdir -p "$BACKUP_DIR"
        
        # Backup configuration
        if [ -d "$HOME/.config/contextbox" ]; then
            cp -r "$HOME/.config/contextbox" "$BACKUP_DIR/" 2>/dev/null || true
        fi
        
        # Backup virtual environment
        if [ -d "$(pwd)/venv_contextbox" ]; then
            cp -r "$(pwd)/venv_contextbox" "$BACKUP_DIR/" 2>/dev/null || true
        fi
        
        # Backup launcher
        if [ -f "$HOME/bin/contextbox" ]; then
            cp "$HOME/bin/contextbox" "$BACKUP_DIR/" 2>/dev/null || true
        fi
        
        echo "$BACKUP_DIR" > /tmp/contextbox_backup_location.txt
        print_success "Backup created successfully"
    fi
}

# Step 1: Check system dependencies
step_check_dependencies() {
    if [ "$SKIP_DEPS" = true ]; then
        print_info "Skipping dependency check (--skip-deps specified)"
        return
    fi
    
    print_step "1/4 - Checking System Dependencies"
    log_message "Starting dependency check"
    
    # Make dependency checker executable
    chmod +x "$SCRIPT_DIR/check_dependencies.sh"
    
    # Run dependency checker
    if bash "$SCRIPT_DIR/check_dependencies.sh"; then
        print_success "All dependencies satisfied"
        log_message "Dependency check: PASSED"
    else
        print_warning "Some dependencies may be missing"
        log_message "Dependency check: WARNINGS"
        
        if [ "$AUTO_MODE" = false ]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_error "Installation cancelled due to missing dependencies"
                exit 1
            fi
        fi
    fi
    echo ""
}

# Step 2: Install ContextBox with enhanced features
step_install_contextbox() {
    print_step "2/4 - Installing ContextBox"
    log_message "Starting ContextBox installation"
    
    # Make enhanced installer executable
    chmod +x "$SCRIPT_DIR/install_enhanced.sh"
    
    # Run enhanced installer
    local installer_args=""
    
    if [ "$RECOVERY_MODE" = true ]; then
        installer_args="--recovery"
    elif [ "$SKIP_DEPS" = true ]; then
        installer_args="--no-deps"
    fi
    
    print_info "Running enhanced installer..."
    if bash "$SCRIPT_DIR/install_enhanced.sh" $installer_args; then
        print_success "ContextBox installed successfully"
        log_message "ContextBox installation: SUCCESS"
    else
        print_error "ContextBox installation failed"
        log_message "ContextBox installation: FAILED"
        
        # Restore backup if installation failed
        if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
            print_info "Restoring backup..."
            restore_from_backup
        fi
        
        exit 1
    fi
    echo ""
}

# Step 3: Run setup wizard
step_setup_wizard() {
    if [ "$SKIP_SETUP" = true ]; then
        print_info "Skipping setup wizard (--skip-setup specified)"
        return
    fi
    
    print_step "3/4 - Configuring Preferences"
    log_message "Starting setup wizard"
    
    # Make setup wizard executable
    chmod +x "$SCRIPT_DIR/setup_wizard.py"
    
    # Check if Python dependencies for wizard are available
    if ! python3 -c "import rich" 2>/dev/null; then
        print_info "Installing setup wizard dependencies..."
        pip3 install rich 2>/dev/null || true
    fi
    
    print_info "Launching setup wizard..."
    if python3 "$SCRIPT_DIR/setup_wizard.py"; then
        print_success "Configuration completed"
        log_message "Setup wizard: SUCCESS"
    else
        print_warning "Setup wizard failed or was cancelled"
        print_info "Using default configuration"
        log_message "Setup wizard: SKIPPED_OR_FAILED"
    fi
    echo ""
}

# Step 4: Final integration and testing
step_final_integration() {
    print_step "4/4 - Final Integration"
    log_message "Starting final integration"
    
    # Make uninstall script executable for future use
    chmod +x "$SCRIPT_DIR/uninstall.sh"
    
    # Update desktop database if available
    if command -v update-desktop-database &> /dev/null; then
        print_info "Updating desktop database..."
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    fi
    
    # Test final installation
    print_info "Running final installation test..."
    
    # Test Python module
    if python3 -c "import contextbox; print('ContextBox module: OK')" 2>/dev/null; then
        print_success "Python module test passed"
    else
        print_error "Python module test failed"
        log_message "Final test: PYTHON_MODULE FAILED"
    fi
    
    # Test CLI
    if command -v contextbox &> /dev/null; then
        local version=$(contextbox --version 2>&1 | head -1)
        print_success "CLI test passed: $version"
    else
        print_warning "CLI test failed (may need PATH update)"
        log_message "Final test: CLI FAILED"
    fi
    
    # Test configuration
    if [ -f "$HOME/.config/contextbox/config.json" ]; then
        print_success "Configuration file exists"
    else
        print_warning "Configuration file not found"
        log_message "Final test: CONFIG MISSING"
    fi
    
    echo ""
}

# Restore from backup
restore_from_backup() {
    if [ -n "$BACKUP_DIR" ] && [ -f /tmp/contextbox_backup_location.txt ]; then
        local backup_location=$(cat /tmp/contextbox_backup_location.txt)
        
        print_info "Restoring from backup: $backup_location"
        
        # Restore configuration
        if [ -d "$backup_location/contextbox" ]; then
            mkdir -p "$HOME/.config"
            cp -r "$backup_location/contextbox" "$HOME/.config/" 2>/dev/null || true
        fi
        
        # Restore virtual environment
        if [ -d "$backup_location/venv_contextbox" ]; then
            cp -r "$backup_location/venv_contextbox" "$(pwd)/" 2>/dev/null || true
        fi
        
        # Restore launcher
        if [ -f "$backup_location/contextbox" ]; then
            mkdir -p "$HOME/bin"
            cp "$backup_location/contextbox" "$HOME/bin/" 2>/dev/null || true
            chmod +x "$HOME/bin/contextbox" 2>/dev/null || true
        fi
        
        print_success "Backup restored"
    fi
}

# Show installation summary
show_installation_summary() {
    print_banner
    
    echo -e "${BOLD}Installation Complete!${NC}"
    echo ""
    
    echo -e "${GREEN}✓${NC} ContextBox has been successfully installed and configured"
    echo ""
    
    echo -e "${BOLD}Installation Details:${NC}"
    echo "  • Configuration: $HOME/.config/contextbox/config.json"
    echo "  • Command: contextbox"
    echo "  • Desktop Entry: ~/.local/share/applications/contextbox.desktop"
    
    if [ -d "$(pwd)/venv_contextbox" ]; then
        echo "  • Virtual Environment: $(pwd)/venv_contextbox"
    fi
    
    echo "  • Installation Log: $INSTALL_LOG"
    
    if [ -n "$BACKUP_DIR" ]; then
        echo "  • Backup: $BACKUP_DIR"
    fi
    
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  1. Test installation: ${GREEN}contextbox --help${NC}"
    echo "  2. Start capturing: ${GREEN}contextbox start${NC}"
    echo "  3. Open GUI: ${GREEN}contextbox gui${NC}"
    echo "  4. Configuration: ${GREEN}contextbox config${NC}"
    
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo "  • Set up global hotkey in your desktop environment"
    echo "  • Configure auto-start if desired"
    echo "  • Explore ContextBox features with ${GREEN}contextbox --help${NC}"
    
    echo ""
    echo -e "${BOLD}Support:${NC}"
    echo "  • Documentation: README.md"
    echo "  • Issues: https://github.com/contextbox/contextbox/issues"
    echo "  • Uninstall: $SCRIPT_DIR/uninstall.sh"
    
    echo ""
    print_success "Enjoy using ContextBox!"
    
    log_message "Installation completed successfully"
}

# Handle errors gracefully
handle_error() {
    local exit_code=$?
    print_error "Installation failed with exit code: $exit_code"
    print_info "Check $INSTALL_LOG for details"
    
    # Offer to restore backup
    if [ -n "$BACKUP_DIR" ] && [ -f /tmp/contextbox_backup_location.txt ]; then
        echo ""
        read -p "Restore from backup? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            restore_from_backup
            print_success "Backup restored. ContextBox should be functional again."
        fi
    fi
    
    exit $exit_code
}

# Main installation function
main() {
    # Set up error handling
    trap handle_error ERR
    
    # Initialize
    init_logging "$@"
    
    print_banner
    print_info "Starting ContextBox installation process..."
    echo ""
    
    # Pre-installation checks
    check_installation_directory
    show_welcome
    create_backup
    
    # Installation steps
    step_check_dependencies
    step_install_contextbox
    step_setup_wizard
    step_final_integration
    
    # Show summary
    show_installation_summary
    
    log_message "Installation process completed successfully"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Master Installer"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --auto              Run in non-interactive mode"
        echo "  --skip-deps         Skip dependency checking"
        echo "  --skip-setup        Skip setup wizard"
        echo "  --recovery          Run in recovery mode"
        echo "  --uninstall         Run uninstaller instead"
        echo ""
        echo "This is the master installer that handles the complete"
        echo "ContextBox installation process including dependency"
        echo "checking, enhanced installation, and configuration."
        exit 0
        ;;
    --auto)
        AUTO_MODE=true
        print_info "Running in auto mode (non-interactive)"
        ;;
    --skip-deps)
        SKIP_DEPS=true
        print_info "Will skip dependency checking"
        ;;
    --skip-setup)
        SKIP_SETUP=true
        print_info "Will skip setup wizard"
        ;;
    --recovery)
        RECOVERY_MODE=true
        print_info "Running in recovery mode"
        ;;
    --uninstall)
        print_info "Running uninstaller..."
        chmod +x "$SCRIPT_DIR/uninstall.sh"
        exec "$SCRIPT_DIR/uninstall.sh" "${@:2}"
        ;;
    *)
        main "$@"
        ;;
esac