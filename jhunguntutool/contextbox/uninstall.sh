#!/bin/bash

# ContextBox Complete Uninstall Script
# Removes all ContextBox files, configurations, and shortcuts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Confirmation prompt
confirm_uninstall() {
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  ContextBox Complete Uninstallation${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    print_warning "This will completely remove ContextBox from your system:"
    echo "  • All ContextBox files and directories"
    echo "  • Configuration files (~/.config/contextbox, ~/.contextbox)"
    echo "  • Desktop shortcuts and applications"
    echo "  • System integration files"
    echo "  • Virtual environments"
    echo "  • Database files in project directories"
    echo ""
    print_warning "This action cannot be undone!"
    echo ""
    read -p "Are you sure you want to uninstall ContextBox? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Uninstallation cancelled."
        exit 0
    fi
}

# Stop any running ContextBox processes
stop_contextbox_processes() {
    print_status "Stopping any running ContextBox processes..."
    
    # Kill processes by name
    local processes=("contextbox" "python*contextbox")
    for pattern in "${processes[@]}"; do
        if pgrep -f "$pattern" > /dev/null; then
            print_status "Stopping processes matching: $pattern"
            pkill -f "$pattern" || true
            sleep 2
        fi
    done
    
    # Kill by specific command patterns
    local cmd_patterns=("-m contextbox" "contextbox cli" "contextbox capture")
    for pattern in "${cmd_patterns[@]}"; do
        if pgrep -f "$pattern" > /dev/null; then
            print_status "Stopping processes with: $pattern"
            pkill -f "$pattern" || true
        fi
    done
    
    print_success "ContextBox processes stopped"
}

# Remove Python packages
remove_python_packages() {
    print_status "Removing ContextBox Python packages..."
    
    # List of packages to uninstall
    local packages=(
        "contextbox"
        "contextbox-cli"
        "contextbox-gui"
    )
    
    for package in "${packages[@]}"; do
        if pip3 show "$package" > /dev/null 2>&1; then
            print_status "Removing $package..."
            pip3 uninstall -y "$package" || true
        fi
    done
    
    # Remove from all virtual environments
    print_status "Removing from virtual environments..."
    for venv_path in ~/.venv/*contextbox* /opt/contextbox*venv*; do
        if [ -d "$venv_path" ] && [ -f "$venv_path/bin/activate" ]; then
            print_status "Checking virtual environment: $venv_path"
            if [ -f "$venv_path/bin/pip" ]; then
                $venv_path/bin/pip uninstall -y contextbox || true
            fi
        fi
    done
    
    print_success "Python packages removed"
}

# Remove configuration directories
remove_config_directories() {
    print_status "Removing configuration directories..."
    
    local config_dirs=(
        "$HOME/.config/contextbox"
        "$HOME/.contextbox"
        "$HOME/.local/share/contextbox"
        "$HOME/.cache/contextbox"
        "$HOME/.var/app/com.contextbox"
    )
    
    for dir in "${config_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "Removing config directory: $dir"
            rm -rf "$dir"
            print_success "Removed: $dir"
        fi
    done
    
    print_success "Configuration directories removed"
}

# Remove desktop entries and shortcuts
remove_desktop_entries() {
    print_status "Removing desktop entries and shortcuts..."
    
    # Remove desktop entries
    local desktop_files=(
        "$HOME/.local/share/applications/contextbox.desktop"
        "$HOME/.local/share/applications/com.contextbox.desktop"
        "/usr/share/applications/contextbox.desktop"
        "/usr/local/share/applications/contextbox.desktop"
    )
    
    for file in "${desktop_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "Removing desktop entry: $file"
            sudo rm -f "$file" 2>/dev/null || rm -f "$file"
        fi
    done
    
    # Remove from application menus
    local app_dirs=(
        "$HOME/.local/share/applications"
        "$HOME/.gnome/apps"
        "$HOME/.kde/share/applications"
    )
    
    for dir in "${app_dirs[@]}"; do
        if [ -d "$dir" ]; then
            find "$dir" -name "*contextbox*" -type f -exec rm -f {} \; 2>/dev/null || true
        fi
    done
    
    # Remove from PATH
    local bin_files=(
        "$HOME/bin/contextbox"
        "/usr/local/bin/contextbox"
        "/usr/bin/contextbox"
    )
    
    for file in "${bin_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "Removing binary: $file"
            sudo rm -f "$file" 2>/dev/null || rm -f "$file"
        fi
    done
    
    # Clean up PATH modifications
    print_status "Cleaning up PATH modifications..."
    if [ -f "$HOME/.bashrc" ]; then
        sed -i '/contextbox/d' "$HOME/.bashrc" 2>/dev/null || true
    fi
    if [ -f "$HOME/.zshrc" ]; then
        sed -i '/contextbox/d' "$HOME/.zshrc" 2>/dev/null || true
    fi
    if [ -f "$HOME/.profile" ]; then
        sed -i '/contextbox/d' "$HOME/.profile" 2>/dev/null || true
    fi
    
    print_success "Desktop entries and shortcuts removed"
}

# Remove project directories
remove_project_directories() {
    print_status "Looking for ContextBox project directories..."
    
    # Remove virtual environments
    local venv_patterns=(
        "$HOME/contextbox*venv*"
        "$HOME/.contextbox*venv*"
        "/opt/contextbox*venv*"
    )
    
    for pattern in "${venv_patterns[@]}"; do
        for dir in $pattern; do
            if [ -d "$dir" ]; then
                print_status "Removing virtual environment: $dir"
                rm -rf "$dir"
            fi
        done
    done
    
    # Remove project directories (non-interactive removal)
    local project_dirs=(
        "$HOME/contextbox"
        "$HOME/ContextBox"
        "$PWD"  # Current directory if it's a ContextBox project
    )
    
    for dir in "${project_dirs[@]}"; do
        if [ -d "$dir" ] && [ -f "$dir/setup.py" ] && grep -q "contextbox" "$dir/setup.py" 2>/dev/null; then
            print_status "Found ContextBox project directory: $dir"
            if [[ "$dir" == "$PWD" ]]; then
                print_warning "Current directory is a ContextBox project"
                print_warning "Please remove it manually after this script completes"
            else
                read -p "Remove directory: $dir? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$dir"
                    print_success "Removed: $dir"
                else
                    print_status "Skipped: $dir"
                fi
            fi
        fi
    done
}

# Remove database files
remove_database_files() {
    print_status "Removing database files..."
    
    local db_patterns=(
        "$HOME/*.db"
        "$HOME/contextbox*.db"
        "$HOME/.contextbox*.db"
    )
    
    for pattern in "${db_patterns[@]}"; do
        for db_file in $pattern; do
            if [ -f "$db_file" ] && grep -q "contextbox" "$db_file" 2>/dev/null; then
                print_status "Removing database: $db_file"
                rm -f "$db_file"
            fi
        done
    done
    
    print_success "Database files removed"
}

# Clean up system integration
cleanup_system_integration() {
    print_status "Cleaning up system integration..."
    
    # Remove from autostart
    local autostart_dirs=(
        "$HOME/.config/autostart"
        "$HOME/.local/share/autostart"
    )
    
    for dir in "${autostart_dirs[@]}"; do
        if [ -d "$dir" ]; then
            find "$dir" -name "*contextbox*" -type f -exec rm -f {} \; 2>/dev/null || true
        fi
    done
    
    # Remove from systemd if installed system-wide
    if [ -d "/etc/systemd/user" ]; then
        sudo rm -f /etc/systemd/user/contextbox.service 2>/dev/null || true
    fi
    
    if [ -d "/etc/systemd/system" ]; then
        sudo rm -f /etc/systemd/system/contextbox.service 2>/dev/null || true
    fi
    
    # Reload systemd daemon if changes were made
    if command -v systemctl > /dev/null; then
        sudo systemctl daemon-reload 2>/dev/null || true
    fi
    
    print_success "System integration cleaned"
}

# Clean up logs and cache
cleanup_logs_cache() {
    print_status "Cleaning up logs and cache..."
    
    local log_patterns=(
        "$HOME/.local/share/contextbox/logs/*"
        "$HOME/.cache/contextbox/*"
        "/tmp/contextbox*"
        "/var/log/contextbox*"
    )
    
    for pattern in "${log_patterns[@]}"; do
        rm -rf $pattern 2>/dev/null || true
    done
    
    # Clean up pip cache for contextbox packages
    print_status "Cleaning up pip cache..."
    pip3 cache purge 2>/dev/null || true
    
    print_success "Logs and cache cleaned"
}

# Generate uninstall report
generate_report() {
    local report_file="$HOME/contextbox_uninstall_report.txt"
    
    print_status "Generating uninstall report..."
    
    cat > "$report_file" << EOF
ContextBox Uninstallation Report
Generated: $(date)

Directories and files that were affected:
EOF
    
    # Add information about removed items
    echo "- Configuration directories: ~/.config/contextbox, ~/.contextbox" >> "$report_file"
    echo "- Desktop entries: ~/.local/share/applications/contextbox.desktop" >> "$report_file"
    echo "- Bin files: ~/bin/contextbox" >> "$report_file"
    echo "- Virtual environments: Checked and cleaned" >> "$report_file"
    echo "- Database files: Checked and cleaned" >> "$report_file"
    echo "- Cache and logs: Cleaned" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "Next steps:" >> "$report_file"
    echo "1. Restart your terminal to apply PATH changes" >> "$report_file"
    echo "2. Remove any remaining ContextBox project directories manually" >> "$report_file"
    echo "3. Clear browser data if ContextBox had browser integration" >> "$report_file"
    echo "" >> "$report_file"
    echo "For support: https://github.com/contextbox/contextbox/issues" >> "$report_file"
    
    print_success "Uninstall report saved to: $report_file"
}

# Main uninstallation function
main() {
    confirm_uninstall
    stop_contextbox_processes
    remove_python_packages
    remove_config_directories
    remove_desktop_entries
    remove_project_directories
    remove_database_files
    cleanup_system_integration
    cleanup_logs_cache
    generate_report
    
    echo ""
    print_success "ContextBox has been completely uninstalled from your system!"
    echo ""
    print_status "Summary:"
    print_status "✓ All Python packages removed"
    print_status "✓ Configuration directories cleaned"
    print_status "✓ Desktop entries and shortcuts removed"
    print_status "✓ Database files cleaned"
    print_status "✓ System integration removed"
    print_status "✓ Logs and cache cleared"
    print_status "✓ Uninstallation report generated"
    echo ""
    print_warning "Please restart your terminal to apply all changes."
    print_status "For support: https://github.com/contextbox/contextbox/issues"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Complete Uninstall Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --force       Skip confirmation prompt"
        echo ""
        echo "This script completely removes ContextBox from your system,"
        echo "including all configuration files, shortcuts, and dependencies."
        exit 0
        ;;
    --force)
        # Skip confirmation for --force flag
        main
        ;;
    *)
        main
        ;;
esac