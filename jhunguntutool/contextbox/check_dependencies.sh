#!/bin/bash

# ContextBox Dependency Checker
# Comprehensive system dependency validation before installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_debug() { echo -e "${MAGENTA}[DEBUG]${NC} $1"; }
print_header() { echo -e "${CYAN}[HEADER]${NC} $1"; }

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/contextbox_deps_check.log"
OS_TYPE=""
DISTRO=""
DISTRO_VERSION=""
MISSING_DEPS=()
INSTALL_SUGGESTIONS=()
EXIT_CODE=0

# Initialize logging
init_logging() {
    echo "ContextBox Dependency Check - $(date)" > "$LOG_FILE"
    echo "=====================================" >> "$LOG_FILE"
    echo "OS: $(uname -a)" >> "$LOG_FILE"
    echo "User: $(whoami)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Detect operating system and distribution
detect_system() {
    print_header "System Detection"
    print_status "Detecting operating system and distribution..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        
        # Check for various distribution indicators
        if [ -f /etc/os-release ]; then
            source /etc/os-release
            DISTRO=$ID
            DISTRO_VERSION=$VERSION_ID
        elif [ -f /etc/debian_version ]; then
            DISTRO="debian"
            DISTRO_VERSION=$(cat /etc/debian_version)
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
            DISTRO_VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+\.?[0-9]*')
        elif [ -f /etc/arch-release ]; then
            DISTRO="arch"
            DISTRO_VERSION=$(cat /etc/version 2>/dev/null || echo "rolling")
        elif [ -f /etc/opensuse-release ]; then
            DISTRO="opensuse"
            DISTRO_VERSION=$(cat /etc/opensuse-release | grep -oE '[0-9]+\.?[0-9]*')
        else
            DISTRO="unknown"
            DISTRO_VERSION="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        DISTRO="macos"
        DISTRO_VERSION=$(sw_vers -productVersion)
    else
        OS_TYPE="unknown"
        DISTRO="unknown"
        DISTRO_VERSION="unknown"
    fi
    
    print_success "Detected: $OS_TYPE - $DISTRO $DISTRO_VERSION"
    log_message "System: $OS_TYPE $DISTRO $DISTRO_VERSION"
}

# Check Python installation and version
check_python() {
    print_header "Python Environment Check"
    
    local python_cmd=""
    local python_version=""
    
    # Check for Python 3.7+
    if command -v python3 &> /dev/null; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local major=$(python3 -c 'import sys; print(sys.version_info.major)')
        local minor=$(python3 -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 7 ]; then
            print_success "Python 3 found: $python_version"
            python_cmd="python3"
        else
            print_error "Python 3.7+ required, found: $python_version"
            MISSING_DEPS+=("python3")
            INSTALL_SUGGESTIONS+=("Install Python 3.7+")
            EXIT_CODE=1
            return
        fi
    elif command -v python &> /dev/null; then
        python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local major=$(python -c 'import sys; print(sys.version_info.major)')
        local minor=$(python -c 'import sys; print(sys.version_info.minor)')
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 7 ]; then
            print_success "Python 3 found: $python_version"
            python_cmd="python"
        else
            print_error "Python 3.7+ required, found: $python_version"
            MISSING_DEPS+=("python3")
            INSTALL_SUGGESTIONS+=("Install Python 3.7+")
            EXIT_CODE=1
            return
        fi
    else
        print_error "Python 3 not found"
        MISSING_DEPS+=("python3")
        INSTALL_SUGGESTIONS+=("Install Python 3.7+")
        EXIT_CODE=1
        return
    fi
    
    # Check Python development headers
    if ! $python_cmd -c "import sysconfig; print(sysconfig.get_path('include'))" &> /dev/null; then
        print_warning "Python development headers not found"
        case $DISTRO in
            "ubuntu"|"debian")
                INSTALL_SUGGESTIONS+=("sudo apt install python3-dev")
                ;;
            "centos"|"rhel"|"fedora")
                INSTALL_SUGGESTIONS+=("sudo yum install python3-devel")
                ;;
            "arch")
                INSTALL_SUGGESTIONS+=("sudo pacman -S python")
                ;;
            "macos")
                if command -v brew &> /dev/null; then
                    INSTALL_SUGGESTIONS+=("brew install python")
                fi
                ;;
        esac
    else
        print_success "Python development headers found"
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        local pip_version=$(pip3 --version | grep -oE '[0-9]+\.[0-9]+\.?[0-9]*' | head -1)
        print_success "pip found: $pip_version"
    elif $python_cmd -m pip --version &> /dev/null; then
        print_success "pip found (via module)"
    else
        print_error "pip not found"
        MISSING_DEPS+=("pip3")
        INSTALL_SUGGESTIONS+=("Install pip for Python 3")
        EXIT_CODE=1
    fi
    
    # Check virtual environment support
    if ! $python_cmd -m venv --help &> /dev/null; then
        print_warning "venv module not available"
        case $DISTRO in
            "ubuntu"|"debian")
                INSTALL_SUGGESTIONS+=("sudo apt install python3-venv")
                ;;
            "arch")
                INSTALL_SUGGESTIONS+=("sudo pacman -S python")
                ;;
            "macos")
                INSTALL_SUGGESTIONS+=("brew install python")
                ;;
        esac
    else
        print_success "venv module available"
    fi
    
    log_message "Python check: $python_cmd $python_version"
}

# Check build tools and compilation dependencies
check_build_tools() {
    print_header "Build Tools Check"
    
    local build_tools=("gcc" "g++" "make" "cmake")
    
    for tool in "${build_tools[@]}"; do
        if command -v $tool &> /dev/null; then
            local version=$($tool --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.?[0-9]*' | head -1)
            print_success "$tool found: $version"
        else
            print_warning "$tool not found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install build-essential")
                    ;;
                "centos"|"rhel"|"fedora")
                    INSTALL_SUGGESTIONS+=("sudo yum groupinstall 'Development Tools'")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S base-devel")
                    ;;
                "macos")
                    if command -v xcode-select &> /dev/null; then
                        INSTALL_SUGGESTIONS+=("xcode-select --install")
                    fi
                    ;;
            esac
        fi
    done
    
    # Check pkg-config
    if command -v pkg-config &> /dev/null; then
        print_success "pkg-config found"
    else
        print_warning "pkg-config not found"
        case $DISTRO in
            "ubuntu"|"debian")
                INSTALL_SUGGESTIONS+=("sudo apt install pkg-config")
                ;;
            "centos"|"rhel"|"fedora")
                INSTALL_SUGGESTIONS+=("sudo yum install pkgconfig")
                ;;
            "arch")
                INSTALL_SUGGESTIONS+=("sudo pacman -S pkg-config")
                ;;
            "macos")
                INSTALL_SUGGESTIONS+=("brew install pkg-config")
                ;;
        esac
    fi
}

# Check system libraries
check_system_libraries() {
    print_header "System Libraries Check"
    
    local libraries=("libpq" "libsqlite3" "libffi")
    local pkg_names=("libpq-dev" "libsqlite3-dev" "libffi-dev")
    
    for i in "${!libraries[@]}"; do
        local lib=${libraries[i]}
        local pkg_name=${pkg_names[i]}
        
        if pkg-config --exists $lib 2>/dev/null; then
            local version=$(pkg-config --modversion $lib)
            print_success "$lib found: $version"
        else
            print_warning "$lib not found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install $pkg_name")
                    ;;
                "centos"|"rhel"|"fedora")
                    INSTALL_SUGGESTIONS+=("sudo yum install postgresql-devel sqlite-devel libffi-devel")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S postgresql-libs sqlite libffi")
                    ;;
                "macos")
                    if command -v brew &> /dev/null; then
                        INSTALL_SUGGESTIONS+=("brew install postgresql sqlite libffi")
                    fi
                    ;;
            esac
        fi
    done
}

# Check GUI and display dependencies
check_gui_dependencies() {
    print_header "GUI Dependencies Check"
    
    # Check for X11/Wayland
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
        print_success "Display environment detected"
        
        # Check screenshot tools
        local screenshot_tools=("gnome-screenshot" "scrot" "import" "screenshot" "xwd")
        local screenshot_found=false
        
        for tool in "${screenshot_tools[@]}"; do
            if command -v $tool &> /dev/null; then
                print_success "Screenshot tool found: $tool"
                screenshot_found=true
                break
            fi
        done
        
        if [ "$screenshot_found" = false ]; then
            print_warning "No screenshot tool found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install gnome-screenshot scrot imagemagick")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S gnome-screenshot scrot imagemagick")
                    ;;
                "macos")
                    INSTALL_SUGGESTIONS+=("screencapture is built-in")
                    ;;
            esac
        fi
        
        # Check clipboard tools
        local clipboard_tools=("xclip" "xsel" "wl-copy" "pbpaste")
        local clipboard_found=false
        
        for tool in "${clipboard_tools[@]}"; do
            if command -v $tool &> /dev/null; then
                print_success "Clipboard tool found: $tool"
                clipboard_found=true
                break
            fi
        done
        
        if [ "$clipboard_found" = false ]; then
            print_warning "No clipboard tool found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install xclip xsel")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S xclip xsel")
                    ;;
            esac
        fi
        
        # Check window management tools
        local wm_tools=("wmctrl" "xdotool")
        local wm_found=false
        
        for tool in "${wm_tools[@]}"; do
            if command -v $tool &> /dev/null; then
                print_success "Window management tool found: $tool"
                wm_found=true
            fi
        done
        
        if [ "$wm_found" = false ]; then
            print_warning "No window management tools found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install wmctrl xdotool")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S wmctrl xdotool")
                    ;;
            esac
        fi
        
    else
        print_warning "No display environment detected (headless/server environment)"
        print_status "GUI features will not be available"
    fi
    
    # Check notification systems
    local notification_daemons=("dunst" "notify-osd" "gnome-shell" "plasmashell")
    local notification_found=false
    
    for daemon in "${notification_daemons[@]}"; do
        if systemctl --user is-active --quiet $daemon 2>/dev/null || \
           pgrep -x $daemon > /dev/null; then
            print_success "Notification system found: $daemon"
            notification_found=true
            break
        fi
    done
    
    if [ "$notification_found" = false ]; then
        print_warning "No notification system detected"
        case $DISTRO in
            "ubuntu"|"debian")
                INSTALL_SUGGESTIONS+=("sudo apt install dunst libnotify-bin")
                ;;
            "arch")
                INSTALL_SUGGESTIONS+=("sudo pacman -S dunst libnotify")
                ;;
        esac
    fi
}

# Check OCR dependencies
check_ocr_dependencies() {
    print_header "OCR Dependencies Check"
    
    if command -v tesseract &> /dev/null; then
        local version=$(tesseract --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.?[0-9]*' | head -1)
        print_success "Tesseract OCR found: $version"
        
        # Check for language packs
        if tesseract --list-langs 2>/dev/null | grep -q "eng"; then
            print_success "English OCR language pack found"
        else
            print_warning "English OCR language pack not found"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install tesseract-ocr-eng")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S tesseract-data-eng")
                    ;;
            esac
        fi
    else
        print_warning "Tesseract OCR not found"
        case $DISTRO in
            "ubuntu"|"debian")
                INSTALL_SUGGESTIONS+=("sudo apt install tesseract-ocr tesseract-ocr-eng")
                ;;
            "arch")
                INSTALL_SUGGESTIONS+=("sudo pacman -S tesseract tesseract-data-eng")
                ;;
            "macos")
                if command -v brew &> /dev/null; then
                    INSTALL_SUGGESTIONS+=("brew install tesseract")
                fi
                ;;
        esac
    fi
}

# Check networking and web dependencies
check_networking_dependencies() {
    print_header "Networking Dependencies Check"
    
    # Check for web-related tools
    local net_tools=("curl" "wget" "jq")
    
    for tool in "${net_tools[@]}"; do
        if command -v $tool &> /dev/null; then
            print_success "$tool found"
        else
            print_warning "$tool not found"
            case $DISTRO in
                "ubuntu"|"debian")
                    case $tool in
                        "curl") INSTALL_SUGGESTIONS+=("sudo apt install curl") ;;
                        "wget") INSTALL_SUGGESTIONS+=("sudo apt install wget") ;;
                        "jq") INSTALL_SUGGESTIONS+=("sudo apt install jq") ;;
                    esac
                    ;;
                "arch")
                    case $tool in
                        "curl") INSTALL_SUGGESTIONS+=("sudo pacman -S curl") ;;
                        "wget") INSTALL_SUGGESTIONS+=("sudo pacman -S wget") ;;
                        "jq") INSTALL_SUGGESTIONS+=("sudo pacman -S jq") ;;
                    esac
                    ;;
                "macos")
                    if command -v brew &> /dev/null; then
                        case $tool in
                            "curl") INSTALL_SUGGESTIONS+=("brew install curl") ;;
                            "wget") INSTALL_SUGGESTIONS+=("brew install wget") ;;
                            "jq") INSTALL_SUGGESTIONS+=("brew install jq") ;;
                        esac
                    fi
                    ;;
            esac
        fi
    done
    
    # Check network connectivity
    if ping -c 1 google.com &> /dev/null; then
        print_success "Internet connectivity confirmed"
    else
        print_warning "No internet connectivity detected"
        INSTALL_SUGGESTIONS+=("Check network connection")
    fi
}

# Check optional dependencies
check_optional_dependencies() {
    print_header "Optional Dependencies Check"
    
    # Check for multimedia tools
    local media_tools=("ffmpeg" "vlc" "convert")
    
    for tool in "${media_tools[@]}"; do
        if command -v $tool &> /dev/null; then
            print_success "$tool found (optional)"
        else
            print_status "$tool not found (optional)"
            case $DISTRO in
                "ubuntu"|"debian")
                    case $tool in
                        "ffmpeg") INSTALL_SUGGESTIONS+=("sudo apt install ffmpeg (optional)") ;;
                        "convert") INSTALL_SUGGESTIONS+=("sudo apt install imagemagick (optional)") ;;
                    esac
                    ;;
                "arch")
                    case $tool in
                        "ffmpeg") INSTALL_SUGGESTIONS+=("sudo pacman -S ffmpeg (optional)") ;;
                        "convert") INSTALL_SUGGESTIONS+=("sudo pacman -S imagemagick (optional)") ;;
                    esac
                    ;;
                "macos")
                    if command -v brew &> /dev/null; then
                        case $tool in
                            "ffmpeg") INSTALL_SUGGESTIONS+=("brew install ffmpeg (optional)") ;;
                            "convert") INSTALL_SUGGESTIONS+=("brew install imagemagick (optional)") ;;
                        esac
                    fi
                    ;;
            esac
        fi
    done
    
    # Check for development tools
    local dev_tools=("git" "tree" "htop")
    
    for tool in "${dev_tools[@]}"; do
        if command -v $tool &> /dev/null; then
            print_success "$tool found"
        else
            print_status "$tool not found (optional)"
            case $DISTRO in
                "ubuntu"|"debian")
                    INSTALL_SUGGESTIONS+=("sudo apt install git tree htop (optional)")
                    ;;
                "arch")
                    INSTALL_SUGGESTIONS+=("sudo pacman -S git tree htop (optional)")
                    ;;
                "macos")
                    if command -v brew &> /dev/null; then
                        INSTALL_SUGGESTIONS+=("brew install git tree htop (optional)")
                    fi
                    ;;
            esac
        fi
    done
}

# Check system resources
check_system_resources() {
    print_header "System Resources Check"
    
    # Check available memory
    if command -v free &> /dev/null; then
        local total_mem=$(free -m | awk 'NR==2{print $2}')
        local available_mem=$(free -m | awk 'NR==2{print $7}')
        
        print_status "Total memory: ${total_mem}MB"
        print_status "Available memory: ${available_mem}MB"
        
        if [ "$available_mem" -lt 512 ]; then
            print_warning "Low memory available (<512MB)"
            INSTALL_SUGGESTIONS+=("Consider freeing up memory or using lightweight mode")
        elif [ "$available_mem" -lt 1024 ]; then
            print_status "Moderate memory available (512MB-1GB)"
        else
            print_success "Good memory available (>1GB)"
        fi
    elif command -v vm_stat &> /dev/null; then
        # macOS
        local page_size=$(vm_stat | grep "page size" | grep -oE '[0-9]+' | head -1)
        local free_pages=$(vm_stat | grep "Pages free" | grep -oE '[0-9]+' | head -1)
        local available_mem=$((free_pages * page_size / 1024 / 1024))
        
        print_status "Available memory: ~${available_mem}MB"
    else
        print_warning "Could not determine memory information"
    fi
    
    # Check disk space
    local root_space=$(df / | awk 'NR==2 {print $4}')
    local root_space_mb=$((root_space / 1024))
    
    print_status "Available disk space: ~${root_space_mb}MB"
    
    if [ "$root_space_mb" -lt 1000 ]; then
        print_warning "Low disk space (<1GB)"
        INSTALL_SUGGESTIONS+=("Free up disk space for installation")
    elif [ "$root_space_mb" -lt 5000 ]; then
        print_status "Moderate disk space (1-5GB)"
    else
        print_success "Good disk space (>5GB)"
    fi
    
    # Check CPU
    if [ -f /proc/cpuinfo ]; then
        local cpu_count=$(grep -c ^processor /proc/cpuinfo)
        print_status "CPU cores: $cpu_count"
        
        if [ "$cpu_count" -lt 2 ]; then
            print_warning "Single core CPU detected - performance may be limited"
        else
            print_success "Multi-core CPU detected"
        fi
    elif command -v sysctl &> /dev/null; then
        # macOS
        local cpu_count=$(sysctl -n hw.ncpu)
        print_status "CPU cores: $cpu_count"
    fi
}

# Check permissions and file system
check_permissions() {
    print_header "Permissions Check"
    
    # Check if we can write to common directories
    local test_dirs=("$HOME" "/tmp" "/usr/local/bin")
    
    for dir in "${test_dirs[@]}"; do
        if [ -w "$dir" ] || [ -w "$(dirname "$dir")" ]; then
            print_success "Write access to $dir"
        else
            print_warning "No write access to $dir"
            INSTALL_SUGGESTIONS+=("May need sudo for $dir")
        fi
    done
    
    # Check home directory
    if [ -w "$HOME" ]; then
        print_success "Write access to home directory"
    else
        print_error "No write access to home directory"
        INSTALL_SUGGESTIONS+=("Fix home directory permissions")
        EXIT_CODE=1
    fi
}

# Generate installation report
generate_report() {
    print_header "Installation Report"
    
    echo ""
    print_status "Dependency Check Summary:"
    echo "================================"
    
    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        print_success "All critical dependencies satisfied!"
    else
        print_error "Missing dependencies: ${MISSING_DEPS[*]}"
        EXIT_CODE=1
    fi
    
    echo ""
    
    if [ ${#INSTALL_SUGGESTIONS[@]} -gt 0 ]; then
        print_status "Installation suggestions:"
        echo "----------------------------------------"
        # Remove duplicates and print
        printf '%s\n' "${INSTALL_SUGGESTIONS[@]}" | sort -u | while IFS= read -r suggestion; do
            echo "  • $suggestion"
        done
        echo "----------------------------------------"
    fi
    
    echo ""
    print_status "System Information:"
    echo "  OS: $OS_TYPE"
    echo "  Distribution: $DISTRO $DISTRO_VERSION"
    echo "  Architecture: $(uname -m)"
    echo "  Kernel: $(uname -r)"
    
    echo ""
    print_status "Log file: $LOG_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "Ready to install ContextBox!"
    else
        print_error "Some dependencies are missing. Install them and run this script again."
    fi
    
    log_message "Check completed with exit code: $EXIT_CODE"
}

# Offer to fix missing dependencies
offer_auto_fix() {
    if [ ${#MISSING_DEPS[@]} -gt 0 ] && command -v sudo &> /dev/null; then
        echo ""
        read -p "Attempt to install missing dependencies automatically? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            auto_install_deps
        fi
    fi
}

# Auto-install missing dependencies
auto_install_deps() {
    print_status "Attempting to install missing dependencies..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            if command -v apt-get &> /dev/null; then
                print_status "Updating package list..."
                sudo apt update || print_warning "Failed to update package list"
                
                # Install critical packages
                local critical_packages=("python3" "python3-dev" "python3-pip" "python3-venv" "build-essential")
                for pkg in "${critical_packages[@]}"; do
                    print_status "Installing $pkg..."
                    sudo apt install -y $pkg || print_warning "Failed to install $pkg"
                done
                
                print_success "Installation attempt completed"
            fi
            ;;
        "centos"|"rhel"|"fedora")
            if command -v yum &> /dev/null; then
                print_status "Installing development tools..."
                sudo yum groupinstall -y "Development Tools" || print_warning "Failed to install development tools"
                sudo yum install -y python3 python3-devel python3-pip || print_warning "Failed to install python packages"
                print_success "Installation attempt completed"
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                print_status "Installing packages via Homebrew..."
                brew install python || print_warning "Failed to install Python"
                print_success "Installation attempt completed"
            else
                print_warning "Homebrew not found. Please install Homebrew first."
            fi
            ;;
        *)
            print_warning "Auto-installation not supported for this distribution"
            ;;
    esac
}

# Main execution
main() {
    # Initialize
    init_logging
    
    print_header "ContextBox Dependency Checker"
    print_status "Checking system for ContextBox installation requirements..."
    echo ""
    
    # Run checks
    detect_system
    check_python
    check_build_tools
    check_system_libraries
    check_gui_dependencies
    check_ocr_dependencies
    check_networking_dependencies
    check_optional_dependencies
    check_system_resources
    check_permissions
    
    # Generate report
    generate_report
    
    # Offer auto-fix
    offer_auto_fix
    
    echo ""
    exit $EXIT_CODE
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ContextBox Dependency Checker"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h      Show this help message"
        echo "  --auto-fix      Attempt to automatically install missing dependencies"
        echo "  --json          Output results in JSON format"
        echo "  --verbose       Show detailed output"
        echo ""
        echo "This script checks for all required dependencies before"
        echo "installing ContextBox, including system packages and Python environment."
        exit 0
        ;;
    --auto-fix)
        detect_system
        auto_install_deps
        exit 0
        ;;
    --json)
        # JSON output mode would be implemented here
        print_status "JSON output mode not yet implemented"
        exit 1
        ;;
    --verbose)
        # Verbose mode - could add more debug output
        main
        ;;
    *)
        main
        ;;
esac