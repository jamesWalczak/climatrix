#!/bin/bash

set -euo pipefail

# Require CLIMATRIX_EXP_DIR to be explicitly set by user
if [ -z "${CLIMATRIX_EXP_DIR:-}" ]; then
    echo "Error: The CLIMATRIX_EXP_DIR environment variable is not set." >&2
    echo "Please set it before running the script, for example:" >&2
    echo "export CLIMATRIX_EXP_DIR=\"/path/to/your/directory\"" >&2
    echo "Refer to the experiment README.md file" >&2
    exit 1
fi

# Script directory
SCRIPT_DIR="$CLIMATRIX_EXP_DIR/scripts"
# Virtual environment path - should already exist from container build
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
    
    # Check if virtual environment exists (should be built into container)
    if [[ ! -d "$VENV_PATH" ]]; then
        error_exit "Virtual environment directory does not exist: $VENV_PATH (should be built into container)"
    fi
    
    # Check for activation script
    local activate_script="$VENV_PATH/bin/activate"
    if [[ ! -f "$activate_script" ]]; then
        error_exit "Virtual environment activation script not found: $activate_script"
    fi
    
    log "Found pre-built virtual environment at: $VENV_PATH"
    
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
    
    # Verify Python version
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "Using Python version: $python_version (from virtual environment)"
    
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
    
    # Set environment variables for consistent Python usage
    export PYTHON_CMD="$python_cmd"
    export PIP_CMD="$pip_cmd"
    
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

# Function to parse command line arguments
parse_arguments() {
    local models=()
    local dataset_id=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                if [[ -n "${2:-}" ]]; then
                    models+=("$2")
                    shift 2
                else
                    error_exit "Error: --model requires a value"
                fi
                ;;
            --dataset_id)
                if [[ -n "${2:-}" ]]; then
                    # Check if the value is an integer
                    if [[ "$2" =~ ^[0-9]+$ ]]; then
                        dataset_id="$2"
                        shift 2
                    else
                        error_exit "Error: --dataset_id must be an integer, got: $2"
                    fi
                else
                    error_exit "Error: --dataset_id requires a value"
                fi
                ;;
            *)
                error_exit "Unknown argument: $1"
                ;;
        esac
    done
    
    # Export the models array for use in other functions
    export SELECTED_MODELS="${models[*]}"
    
    # If no models specified, default to all available models
    if [[ ${#models[@]} -eq 0 ]]; then
        log "No models specified, using default: idw"
        export SELECTED_MODELS="idw"
    fi
    
    # Export dataset_id if provided, otherwise set to None
    if [[ -n "$dataset_id" ]]; then
        export DATASET_ID="$dataset_id"
        log "Dataset ID: ${DATASET_ID}"
    else
        log "No dataset ID specified, using None"
        export DATASET_ID="None"
    fi
    
    log "Selected models: ${SELECTED_MODELS}"
}

# Function to validate model names
validate_models() {
    local valid_models=("idw" "ok" "sinet" "mmgn")
    local selected_models=($SELECTED_MODELS)
    
    for model in "${selected_models[@]}"; do
        local is_valid=false
        for valid_model in "${valid_models[@]}"; do
            if [[ "$model" == "$valid_model" ]]; then
                is_valid=true
                break
            fi
        done
        
        if [[ "$is_valid" != true ]]; then
            error_exit "Invalid model: $model. Valid models are: ${valid_models[*]}"
        fi
    done
    
    log "All specified models are valid"
}

# Function to get script info for a model
get_model_script() {
    local model="$1"
    
    case "$model" in
        "idw")
            echo "$SCRIPT_DIR/idw/run_idw.py:IDW script"
            ;;
        "ok")
            echo "$SCRIPT_DIR/kriging/run_ok.py:Kriging script"
            ;;
        "sinet")
            echo "$SCRIPT_DIR/inr/sinet/run_sinet.py:SINET script"
            ;;
        "mmgn")
            echo "$SCRIPT_DIR/inr/mmgn/run_mmgn.py:MMGN script"
            ;;
        *)
            error_exit "Unknown model: $model"
            ;;
    esac
}

# Function to run Python script with error handling
run_python_script() {
    local script="$1"
    local description="$2"
    local dataset_id="$3"    
    
    log "=== Running $description ==="
    log "Script: $script"
    log "Using Python: ${PYTHON_CMD} ($(which ${PYTHON_CMD}))"
    log "Virtual environment: ${VIRTUAL_ENV:-none}"
    log "Dataset ID: $dataset_id"
    
    # Set additional environment variables for the script
    export PYTHONUNBUFFERED=1  # Ensure output is not buffered
    export PYTHONDONTWRITEBYTECODE=1  # Don't write .pyc files
    
    if [[ "$dataset_id" != "None" ]]; then
        ${PYTHON_CMD} "$script_path" --dataset_id "$dataset_id" || error_exit "Failed to run $script_desc"
    else
        ${PYTHON_CMD} "$script_path" || error_exit "Failed to run $script_desc"
    fi

    log "Successfully completed $description"
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
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate specified models
    validate_models
    
    # Detect and log environment
    local container_env=$(detect_container)
    log "Detected environment: $container_env"
    
    # Setup Python environment with venv
    setup_python_environment
    
    # Always run the preparation script first
    log "=== Pre-flight Script Check ==="
    local prep_script="$SCRIPT_DIR/prepare_ecad_observations.py"
    check_python_script "$prep_script" "ECAD observations preparation script"
    
    # Build list of model scripts to run based on selected models
    local selected_models=($SELECTED_MODELS)
    local scripts_to_run=("$prep_script:ECAD observations preparation script")
    
    for model in "${selected_models[@]}"; do
        local script_info=$(get_model_script "$model")
        IFS=':' read -r script_path script_desc <<< "$script_info"
        
        if [[ -f "$script_path" ]]; then
            check_python_script "$script_path" "$script_desc"
            scripts_to_run+=("$script_path:$script_desc")
        else
            log "WARNING: $script_desc not found: $script_path"
        fi
    done
    
    if [[ ${#scripts_to_run[@]} -le 1 ]]; then
        error_exit "No model scripts found to execute (only preparation script available)"
    fi
    
    # Run Python scripts
    log "=== Execution Phase ==="
    for script_info in "${scripts_to_run[@]}"; do
        IFS=':' read -r script_path script_desc <<< "$script_info"
        run_python_script "$script_path" "$script_desc" "$DATASET_ID"
    done
    
    log "=== Pipeline Completed Successfully ==="
    log "Container environment: $container_env"
    log "Virtual environment used: ${VIRTUAL_ENV:-none}"
    log "Python used: ${PYTHON_CMD} ($(${PYTHON_CMD} --version))"
    log "Models executed: ${SELECTED_MODELS}"
    log "Dataset ID: ${DATASET_ID}"
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
