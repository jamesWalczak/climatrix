#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../conf/exp1"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling function
error_exit() {
    log "ERROR: $1"
    exit 1
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

# Function to activate virtual environment with detailed logging
activate_venv() {
    log "=== Virtual Environment Setup ==="
    log "Virtual environment path: $VENV_PATH"
    
    # Check if virtual environment directory exists
    if [[ ! -d "$VENV_PATH" ]]; then
        error_exit "Virtual environment directory does not exist: $VENV_PATH"
    fi
    
    # Check if activate script exists
    local activate_script="$VENV_PATH/bin/activate"
    if [[ ! -f "$activate_script" ]]; then
        error_exit "Virtual environment activate script not found: $activate_script"
    fi
    
    log "Activating virtual environment..."
    source "$activate_script"
    
    # Verify activation was successful
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        error_exit "Failed to activate virtual environment"
    fi
    
    log "Virtual environment activated successfully"
    log "VIRTUAL_ENV: $VIRTUAL_ENV"
    
    # Check Python binary locations and versions
    log "=== Python Environment Check ==="
    local python_bin="$VIRTUAL_ENV/bin/python"
    local pip_bin="$VIRTUAL_ENV/bin/pip"
    
    if [[ ! -x "$python_bin" ]]; then
        error_exit "Python binary not found or not executable: $python_bin"
    fi
    
    if [[ ! -x "$pip_bin" ]]; then
        error_exit "Pip binary not found or not executable: $pip_bin"
    fi
    
    log "Python binary: $python_bin"
    log "Python path from which: $(which python)"
    log "Python version: $($python_bin --version)"
    log "Pip binary: $pip_bin"
    log "Pip version: $($pip_bin --version)"
    
    # Check if we're using the correct Python (from venv)
    local current_python=$(which python)
    if [[ "$current_python" != "$python_bin" ]]; then
        log "WARNING: 'which python' points to $current_python, expected $python_bin"
    else
        log "SUCCESS: Using correct Python from virtual environment"
    fi
    
    # List installed packages for debugging
    log "=== Installed Python Packages ==="
    $python_bin -m pip list | head -20  # Show first 20 packages to avoid too much output
    local package_count=$($python_bin -m pip list | wc -l)
    log "Total packages installed: $((package_count - 2))"  # Subtract header lines
    
    # Check for specific required packages
    log "=== Checking Required Packages ==="
    local required_packages=("xarray" "numpy" "pandas" "matplotlib" "scipy")
    for package in "${required_packages[@]}"; do
        if $python_bin -c "import $package" 2>/dev/null; then
            local version=$($python_bin -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            log "✓ $package ($version) - OK"
        else
            log "✗ $package - NOT FOUND"
        fi
    done
}

# Function to run Python script with error handling
run_python_script() {
    local script="$1"
    local description="$2"
    
    log "=== Running $description ==="
    log "Script: $script"
    log "Using Python: $(which python)"
    
    if ! python "$script"; then
        error_exit "Failed to run $description: $script"
    fi
    
    log "Successfully completed $description"
}

# Main execution
main() {
    log "=== Starting Experiment Pipeline ==="
    log "Script directory: $SCRIPT_DIR"
    log "Working directory: $(pwd)"
    
    # Setup virtual environment and dependencies
    log "=== Setup Phase ==="
    setup_script="$SCRIPT_DIR/../conf/setup.sh"
    check_executable "$setup_script" "Setup script"
    
    log "Running setup script with -f flag..."
    if ! "$setup_script" -f; then
        error_exit "Setup script failed"
    fi
    log "Setup script completed successfully"
    
    # Download data
    log "=== Download Phase ==="
    download_script="$SCRIPT_DIR/download_blend_mean_temperature.sh"
    check_executable "$download_script" "Download script"
    
    log "Running download script..."
    if ! "$download_script"; then
        error_exit "Download script failed"
    fi
    log "Download script completed successfully"
    
    # Activate virtual environment and verify setup
    activate_venv
    
    # Check all Python scripts exist before running
    log "=== Pre-flight Script Check ==="
    check_python_script "$SCRIPT_DIR/prepare_ecad_observations.py" "ECAD observations preparation script"
    check_python_script "$SCRIPT_DIR/kriging/run_ok.py" "Kriging script"
    check_python_script "$SCRIPT_DIR/idw/run_idw.py" "IDW script"
    check_python_script "$SCRIPT_DIR/inr/sinet/run_sinet.py" "SINET script"
    
    # Run Python scripts
    log "=== Execution Phase ==="
    run_python_script "$SCRIPT_DIR/prepare_ecad_observations.py" "ECAD observations preparation"
    run_python_script "$SCRIPT_DIR/kriging/run_ok.py" "Kriging analysis"
    run_python_script "$SCRIPT_DIR/idw/run_idw.py" "IDW interpolation"
    run_python_script "$SCRIPT_DIR/inr/sinet/run_sinet.py" "SINET analysis"
    
    log "=== Pipeline Completed Successfully ==="
}

# Run main function
main "$@"