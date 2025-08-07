#!/bin/bash

set -euo pipefail

if [ -z "$CLIMATRIX_EXP_DIR" ]; then
	echo "Error: The CLIMATRIX_EXP_DIR environment variable is not set." >&2
	echo "Please set it before running the script, for example:" >&2
	echo "export CLIMATRIX_EXP_DIR=\"/path/to/your/directory\"" >&2
	echo "Refer to the experiment README.md file" >&2
	exit 1
 fi

# Script directory
SCRIPT_DIR="$CLIMATRIX_EXP_DIR/scripts"
# Virtual environment path
VENV_PATH="$CLIMATRIX_EXP_DIR/conf/exp1"

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

# Function to setup and activate virtual environment
setup_virtual_environment() {
    log "=== Virtual Environment Setup ==="
    log "Target venv path: $VENV_PATH"
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        error_exit "Virtual environment directory does not exist: $VENV_PATH"
    fi
    
    # Check for activation script
    local activate_script="$VENV_PATH/bin/activate"
    if [[ ! -f "$activate_script" ]]; then
        error_exit "Virtual environment activation script not found: $activate_script"
    fi
    
    log "Found virtual environment at: $VENV_PATH"
    
    # Source the virtual environment
    log "Activating virtual environment..."
    set +u  # Temporarily disable undefined variable checking for venv activation
    source "$activate_script"
    set -u  # Re-enable undefined variable checking
    
    # Verify activation
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        error_exit "Failed to activate virtual environment"
    fi
    
    log "Virtual environment activated: $VIRTUAL_ENV"
    
    return 0
}

# Function to setup Python environment with virtual environment support
setup_python_environment() {
    log "=== Python Environment Setup ==="
    
    # Detect container environment
    local container_env=$(detect_container)
    log "Container environment detected: $container_env"
    
    # Setup and activate virtual environment
    setup_virtual_environment
    
    # Find Python binary (should now be from venv)
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
    
    # Verify we're using the venv Python
    local python_executable=$($python_cmd -c "import sys; print(sys.executable)")
    if [[ "$python_executable" != "$VENV_PATH"* ]]; then
        log "WARNING: Python executable is not from expected venv path"
        log "Expected: $VENV_PATH/bin/python*"
        log "Actual: $python_executable"
    else
        log "✓ Confirmed using venv Python: $python_executable"
    fi
    
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
    
    # Verify pip is using the venv
    local pip_location=$($pip_cmd --version | grep -o '/[^[:space:]]*' | head -1 || echo "unknown")
    if [[ "$pip_location" != "unknown" && "$pip_location" == "$VENV_PATH"* ]]; then
        log "✓ Confirmed using venv pip: $pip_location"
    elif [[ "$pip_location" != "unknown" ]]; then
        log "WARNING: Pip location may not be from venv: $pip_location"
    fi
    
    # Set environment variables for consistent Python usage
    export PYTHON_CMD="$python_cmd"
    export PIP_CMD="$pip_cmd"
    
    # Check Python site-packages directory
    local site_packages=$($python_cmd -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "unknown")
    log "Python site-packages directory: $site_packages"
    
    # For venv, we typically install directly (no --user needed)
    export PIP_INSTALL_ARGS="--no-cache-dir"
    log "Pip install arguments: ${PIP_INSTALL_ARGS}"
    
    # List currently installed packages in venv
    log "=== Currently Installed Python Packages in venv ==="
    $python_cmd -m pip list 2>/dev/null | head -20 || log "Could not list packages"
    local package_count=$($python_cmd -m pip list 2>/dev/null | wc -l || echo "0")
    if [[ "$package_count" -gt 0 ]]; then
        log "Total packages found: $((package_count - 2))"  # Subtract header lines
    fi
    
    # Check for specific required packages
    log "=== Checking Required Packages ==="
    local required_packages=("climatrix")
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
        log "=== Installing Missing Packages to venv ==="
        log "Missing packages: ${missing_packages[*]}"
        
        for package in "${missing_packages[@]}"; do
            log "Installing $package to virtual environment..."
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
    log "Virtual environment: $VIRTUAL_ENV"
    log "Python interpreter: $(which $python_cmd)"
    log "Python version: $($python_cmd --version)"
    log "Python executable path: $($python_cmd -c 'import sys; print(sys.executable)')"
    log "Python path (first 3 entries): $($python_cmd -c 'import sys; print(sys.path[:3])')"
    
    # Test import of critical packages
    for package in "${required_packages[@]}"; do
        if ! $python_cmd -c "import $package" 2>/dev/null; then
            error_exit "Final verification failed: cannot import $package"
        fi
    done
    
    log "Python environment setup completed successfully"
}

# Function to run Python script with error handling
run_python_script() {
    local script="$1"
    local description="$2"
    
    log "=== Running $description ==="
    log "Script: $script"
    log "Using Python: ${PYTHON_CMD} ($(which ${PYTHON_CMD}))"
    log "Virtual environment: ${VIRTUAL_ENV:-none}"
    
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
    
    # Since we're using our own venv, we can run setup script normally
    # but still try to skip additional venv creation if possible
    log "Running setup script..."
    case "$container_env" in
        "docker"|"apptainer")
            # Try to skip additional venv creation since we already have one
            if ! "$setup_script" -f --no-venv 2>/dev/null; then
                log "Setup script doesn't support --no-venv flag, trying with -f only..."
                if ! "$setup_script" -f; then
                    error_exit "Setup script failed"
                fi
            fi
            ;;
        *)
            if ! "$setup_script" -f; then
                error_exit "Setup script failed"
            fi
            ;;
    esac
    
    log "Setup script completed successfully"
}

# Main execution
main() {
    log "=== Starting Container-Aware Experiment Pipeline with venv ==="
    log "Script directory: $SCRIPT_DIR"
    log "Working directory: $(pwd)"
    log "User: $(whoami)"
    log "UID: $(id -u)"
    log "GID: $(id -g)"
    log "Target virtual environment: $VENV_PATH"
    
    # Detect and log environment
    local container_env=$(detect_container)
    log "Detected environment: $container_env"
    
    # Setup Python environment with venv
    setup_python_environment
    
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
    log "Virtual environment used: ${VIRTUAL_ENV:-none}"
    log "Python used: ${PYTHON_CMD} ($(${PYTHON_CMD} --version))"
}

# Handle script termination gracefully
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "Pipeline terminated with error (exit code: $exit_code)"
    fi
    
    # Deactivate virtual environment if it was activated
    if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v deactivate >/dev/null 2>&1; then
        log "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Run main function
main "$@"