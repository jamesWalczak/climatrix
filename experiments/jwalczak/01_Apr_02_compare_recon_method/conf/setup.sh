#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default CLIMATRIX_EXP_DIR if not provided (for container builds)
if [ -z "$CLIMATRIX_EXP_DIR" ]; then
    if [ -d "/app" ]; then
        export CLIMATRIX_EXP_DIR="/app"
        echo "Using default CLIMATRIX_EXP_DIR=/app for container build"
    else
        echo "Error: The CLIMATRIX_EXP_DIR environment variable is not set." >&2
        echo "Please set it before running the script, for example:" >&2
        echo "export CLIMATRIX_EXP_DIR=\"/path/to/your/directory\"" >&2
        echo "Refer to the experiment README.md file" >&2
        exit 1
    fi
fi

VENV_NAME="${CLIMATRIX_EXP_DIR}/conf/exp1"

function create_venv() {
    echo "Creating virtual environment: $VENV_NAME"
    # Use --copies to avoid symlinks that break in containers
    python3 -m venv --copies "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error creating virtual environment."
        exit 1
    fi
    
    # Fix shebangs for container compatibility - this is critical!
    echo "Fixing virtual environment shebangs for container use..."
    find "$VENV_NAME/bin" -type f -executable | while read script; do
        if head -1 "$script" | grep -q python; then
            echo "Fixing shebang in: $script"
            sed -i '1c#!/usr/bin/env python3' "$script"
        fi
    done
    
    # Also fix the pyvenv.cfg file to use container paths
    echo "Fixing pyvenv.cfg for container use..."
    cat > "$VENV_NAME/pyvenv.cfg" << EOF
home = /usr/bin
include-system-site-packages = false
version = $(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
executable = /usr/bin/python3
command = /usr/bin/python3 -m venv --copies $VENV_NAME
EOF
    
    echo "Virtual environment '$VENV_NAME' created successfully."
}

function activate_venv() {
    echo "Activating virtual environment: $VENV_NAME"
    source "$VENV_NAME/bin/activate"
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error activating virtual environment."
        exit 1
    fi
    echo "Virtual environment '$VENV_NAME' activated."
}

function install_dependencies() {
    echo "Installing dependencies from requirements.txt"
    # Use python -m pip to avoid shebang issues
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error installing dependencies."
        exit 1
    fi
    
    # Fix any newly installed script shebangs
    echo "Fixing shebangs after package installation..."
    find "$VENV_NAME/bin" -type f -executable | while read script; do
        if head -1 "$script" | grep -q python; then
            sed -i '1c#!/usr/bin/env python3' "$script"
        fi
    done
    
    echo "Dependencies installed successfully."
}

function fix_broken_venv() {
    echo "Attempting to fix broken virtual environment..."
    
    # Remove the broken pyvenv.cfg that contains host paths
    if [ -f "$VENV_NAME/pyvenv.cfg" ]; then
        rm "$VENV_NAME/pyvenv.cfg"
    fi
    
    # Recreate pyvenv.cfg with container paths
    cat > "$VENV_NAME/pyvenv.cfg" << EOF
home = /usr/bin
include-system-site-packages = false
version = $(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
executable = /usr/bin/python3
command = /usr/bin/python3 -m venv --copies $VENV_NAME
EOF

    # Fix all shebangs
    find "$VENV_NAME/bin" -type f -executable | while read script; do
        if head -1 "$script" | grep -q python; then
            echo "Fixing shebang in: $script"
            sed -i '1c#!/usr/bin/env python3' "$script"
        fi
    done
    
    echo "Virtual environment fixed."
}

# Build completion marker to prevent rebuild issues
BUILD_MARKER="$VENV_NAME/.container_built"

# Always remove existing venv if we're in a container build to ensure clean state
if [ -d "/app" ] && [ "$PWD" = "/app/conf" ]; then
    echo "Container build detected - ensuring clean virtual environment"
    if [ -d "$VENV_NAME" ]; then
        echo "Removing existing virtual environment for clean container build"
        rm -rf "$VENV_NAME"
    fi
fi

if [ -d "$VENV_NAME" ]; then
    if [ -f "$BUILD_MARKER" ]; then
        echo "Virtual environment '$VENV_NAME' already built and ready."
        # Don't activate during build, just verify
        if [ -f "$VENV_NAME/bin/activate" ]; then
            echo "Virtual environment is properly configured."
        else
            echo "Virtual environment appears broken, fixing..."
            fix_broken_venv
            touch "$BUILD_MARKER"
        fi
    elif [ "$1" == "-f" ]; then
        echo "Force reinstall requested. Removing existing virtual environment '$VENV_NAME'."
        rm -rf "$VENV_NAME"
        create_venv
        activate_venv
        install_dependencies
        touch "$BUILD_MARKER"
    else
        echo "Virtual environment exists but not marked as complete. Fixing..."
        fix_broken_venv
        activate_venv
        install_dependencies
        touch "$BUILD_MARKER"
    fi
else
    create_venv
    activate_venv
    install_dependencies
    touch "$BUILD_MARKER"
fi

echo "Setup complete. Virtual environment '$VENV_NAME' is configured with dependencies installed."