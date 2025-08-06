#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling function
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Function to detect container environment
detect_container() {
    local container_type="none"
    
    # Check for Docker
    if [[ -f /.dockerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        container_type="docker"
    # Check for Apptainer/Singularity
    elif [[ -n "${APPTAINER_CONTAINER:-}" ]] || [[ -n "${SINGULARITY_CONTAINER:-}" ]] || [[ -n "${APPTAINER_NAME:-}" ]] || [[ -n "${SINGULARITY_NAME:-}" ]]; then
        container_type="apptainer"
    # Additional check for Apptainer bind mounts
    elif grep -q "/proc/.*/root" /proc/mounts 2>/dev/null; then
        container_type="apptainer"
    fi
    
    echo "$container_type"
}

# Function to check if file exists and is executable
check_executable() {
    local file="$1"
    local description="$2"
    
    if [[ ! -f "$file" ]]; then
        error_exit "$description does not exist: $file"
    fi
    
    if [[ ! -x "$file" ]]; then
        error_exit "$description is not executable: $file"
    fi
    
    log "$description found and executable: $file"
}

# Function to check Python script exists
check_python_script() {
    local script="$1"
    local description="$2"
    
    if [[ ! -f "$script" ]]; then
        error_exit "$description does not exist: $script"
    fi
    
    log "$description found: $script"
}

# Function to setup system Python environment with detailed logging
setup_system_python() {
    log "=== System Python Environment Setup ==="
    
    # Detect container environment
    local container_env=$(detect_container)
    log "Container environment detected: $container_env"
    
    # Find Python binary
    local python_cmd=""
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            python_cmd="$cmd"
            break
        fi
    done
    
    if [[ -z "$python_cmd" ]]; then
        error_exit "No Python interpreter found (tried python3, python)"
    fi
    
    log "Python command: $python_cmd"
    log "Python binary location: $(which $python_cmd)"
    log "Python version: $($python_cmd --version)"
    
    # Check pip availability
    local pip_cmd=""
    for cmd in pip3 pip; do
        if command -v "$cmd" >/dev/null 2>&1; then
            pip_cmd="$cmd"
            break
        fi
    done
    
    if [[ -z "$pip_cmd" ]]; then
        log "WARNING: No pip found, trying python -m pip"
        if ! $python_cmd -m pip --version >/dev/null 2>&1; then
            error_exit "Neither pip command nor 'python -m pip' is available"
        fi
        pip_cmd="$python_cmd -m pip"
    fi
    
    log "Pip command: $pip_cmd"
    log "Pip version: $($pip_cmd --version)"
    
    # Set environment variables for consistent Python usage
    export PYTHON_CMD="$python_cmd"
    export PIP_CMD="$pip_cmd"
    
    # Check Python site-packages directory
    local site_packages=$($python_cmd -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "unknown")
    log "Python site-packages directory: $site_packages"
    
    # Check user site-packages directory
    local user_site=$($python_cmd -c "import site; print(site.getusersitepackages())" 2>/dev/null || echo "unknown")
    log "Python user site-packages directory: $user_site"
    
    # Container-specific adjustments
    case "$container_env" in
        "docker")
            log "=== Docker Container Configuration ==="
            log "Running in Docker container"
            # In Docker, we typically have root access and can install system-wide
            export PIP_INSTALL_ARGS="--no-cache-dir"
            ;;
        "apptainer")
            log "=== Apptainer Container Configuration ==="
            log "Running in Apptainer/Singularity container"
            # In Apptainer, filesystem is typically read-only except for bind mounts
            # Use user site-packages if possible
            export PIP_INSTALL_ARGS="--user --no-cache-dir"
            export PYTHONUSERBASE="${HOME}/.local"
            # Ensure user site-packages is in Python path
            export PYTHONPATH="${PYTHONUSERBASE}/lib/python$($python_cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')/site-packages:${PYTHONPATH:-}"
            ;;
        "none")
            log "=== Native System Configuration ==="
            log "Running on native system (not in container)"
            # Use user install to avoid system conflicts
            export PIP_INSTALL_ARGS="--user --no-cache-dir"
            ;;
    esac
    
    log "Pip install arguments: ${PIP_INSTALL_ARGS:-none}"
    
    # List currently installed packages
    log "=== Currently Installed Python Packages ==="
    $python_cmd -m pip list 2>/dev/null | head -20 || log "Could not list packages"
    local package_count=$($python_cmd -m pip list 2>/dev/null | wc -l || echo "0")
    if [[ "$package_count" -gt 0 ]]; then
        log "Total packages found: $((package_count - 2))"  # Subtract header lines
    fi
    
    # Check for specific required packages
    log "=== Checking Required Packages ==="
    local required_packages=("xarray" "numpy" "pandas" "matplotlib" "scipy")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if $python_cmd -c "import $package" 2>/dev/null; then
            local version=$($python_cmd -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            log "✓ $package ($version) - OK"
        else
            log "✗ $package - NOT FOUND"
            missing_packages+=("$package")
        fi
    done
    
    # Install missing packages if any
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log "=== Installing Missing Packages ==="
        log "Missing packages: ${missing_packages[*]}"
        
        # Check if we can install packages
        if [[ "$container_env" == "apptainer" ]]; then
            log "WARNING: In Apptainer container, package installation may fail if filesystem is read-only"
            log "Consider pre-installing packages in the container image or using bind mounts"
        fi
        
        for package in "${missing_packages[@]}"; do
            log "Installing $package..."
            if eval "$PIP_CMD install ${PIP_INSTALL_ARGS:-} $package"; then
                log "✓ Successfully installed $package"
            else
                log "✗ Failed to install $package"
                error_exit "Could not install required package: $package"
            fi
        done
    fi
    
    # Final verification
    log "=== Final Environment Verification ==="
    log "Python interpreter: $(which $python_cmd)"
    log "Python version: $($python_cmd --version)"
    log "Python executable path: $($python_cmd -c 'import sys; print(sys.executable)')"
    log "Python path: $($python_cmd -c 'import sys; print(sys.path[:3])')"
    
    # Test import of critical packages
    for package in "${required_packages[@]}"; do
        if ! $python_cmd -c "import $package" 2>/dev/null; then
            error_exit "Final verification failed: cannot import $package"
        fi
    done
    
    log "System Python environment setup completed successfully"
}

# Function to run Python script with error handling
run_python_script() {
    local script="$1"
    local description="$2"
    
    log "=== Running $description ==="
    log "Script: $script"
    log "Using Python: ${PYTHON_CMD} ($(which ${PYTHON_CMD}))"
    
    # Set additional environment variables for the script
    export PYTHONUNBUFFERED=1  # Ensure output is not buffered
    export PYTHONDONTWRITEBYTECODE=1  # Don't write .pyc files
    
    if ! ${PYTHON_CMD} "$script"; then
        error_exit "Failed to run $description: $script"
    fi
    
    log "Successfully completed $description"
}

# Function to handle setup script with container awareness
run_setup_script() {
    local setup_script="$1"
    local container_env=$(detect_container)
    
    log "=== Setup Script Execution ==="
    log "Setup script: $setup_script"
    log "Container environment: $container_env"
    
    # Check if setup script exists and is executable
    check_executable "$setup_script" "Setup script"
    
    # Run setup script with appropriate flags
    case "$container_env" in
        "docker"|"apptainer")
            log "Running setup script in container mode..."
            # Skip virtual environment creation in containers
            if [[ -x "$setup_script" ]]; then
                if ! "$setup_script" -f --no-venv 2>/dev/null; then
                    log "Setup script doesn't support --no-venv flag, trying with -f only..."
                    if ! "$setup_script" -f; then
                        error_exit "Setup script failed"
                    fi
                fi
            fi
            ;;
        *)
            log "Running setup script in native mode..."
            if ! "$setup_script" -f; then
                error_exit "Setup script failed"
            fi
            ;;
    esac
    
    log "Setup script completed successfully"
}

# Main execution
main() {
    log "=== Starting Container-Aware Experiment Pipeline ==="
    log "Script directory: $SCRIPT_DIR"
    log "Working directory: $(pwd)"
    log "User: $(whoami)"
    log "UID: $(id -u)"
    log "GID: $(id -g)"
    
    # Detect and log environment
    local container_env=$(detect_container)
    log "Detected environment: $container_env"
    
    # Setup system Python environment
    setup_system_python
    
    # Download data
    log "=== Download Phase ==="
    download_script="$SCRIPT_DIR/download_blend_mean_temperature.sh"
    
    if [[ -f "$download_script" ]]; then
        check_executable "$download_script" "Download script"
        
        log "Running download script..."
        if ! "$download_script"; then
            error_exit "Download script failed"
        fi
        log "Download script completed successfully"
    else
        log "WARNING: Download script not found: $download_script"
        log "Continuing without download script..."
    fi
    
    # Check all Python scripts exist before running
    log "=== Pre-flight Script Check ==="
    local python_scripts=(
        "$SCRIPT_DIR/prepare_ecad_observations.py:ECAD observations preparation script"
        "$SCRIPT_DIR/kriging/run_ok.py:Kriging script"
        "$SCRIPT_DIR/idw/run_idw.py:IDW script"
        "$SCRIPT_DIR/inr/sinet/run_sinet.py:SINET script"
    )
    
    local scripts_to_run=()
    for script_info in "${python_scripts[@]}"; do
        IFS=':' read -r script_path script_desc <<< "$script_info"
        if [[ -f "$script_path" ]]; then
            check_python_script "$script_path" "$script_desc"
            scripts_to_run+=("$script_path:$script_desc")
        else
            log "WARNING: $script_desc not found: $script_path"
        fi
    done
    
    if [[ ${#scripts_to_run[@]} -eq 0 ]]; then
        error_exit "No Python scripts found to execute"
    fi
    
    # Run Python scripts
    log "=== Execution Phase ==="
    for script_info in "${scripts_to_run[@]}"; do
        IFS=':' read -r script_path script_desc <<< "$script_info"
        run_python_script "$script_path" "$script_desc"
    done
    
    log "=== Pipeline Completed Successfully ==="
    log "Container environment: $container_env"
    log "Python used: ${PYTHON_CMD} ($(${PYTHON_CMD} --version))"
}

# Handle script termination gracefully
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "Pipeline terminated with error (exit code: $exit_code)"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Run main function
main "$@"