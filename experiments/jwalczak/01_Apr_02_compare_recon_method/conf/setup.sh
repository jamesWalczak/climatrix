#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$CLIMATRIX_EXP_DIR" ]; then
	echo "Error: The CLIMATRIX_EXP_DIR environment variable is not set." >&2
	echo "Please set it before running the script, for example:" >&2
	echo "export CLIMATRIX_EXP_DIR=\"/path/to/your/directory\"" >&2
	echo "Refer to the experiment README.md file" >&2
	exit 1
fi
VENV_NAME="${CLIMATRIX_EXP_DIR}/exp1"

function create_venv() {
  echo "Creating virtual environment: $VENV_NAME"
  python3 -m venv "$VENV_NAME"
  if [ $? -ne 0 ]; then
    echo "Error creating virtual environment."
    exit 1
  fi
  
  # Fix shebangs for container compatibility
  echo "Fixing virtual environment shebangs for container use..."
  find "$VENV_NAME/bin" -type f -executable | while read script; do
    if head -1 "$script" | grep -q python; then
      sed -i '1c#!/usr/bin/env python3' "$script"
    fi
  done
  
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
  ${VENV_NAME}/bin/pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    echo "Error installing dependencies."
    exit 1
  fi
  echo "Dependencies installed successfully."
}

# Build completion marker to prevent rebuild issues
BUILD_MARKER="$VENV_NAME/.container_built"

if [ -d "$VENV_NAME" ]; then
  if [ -f "$BUILD_MARKER" ]; then
    echo "Virtual environment '$VENV_NAME' already built and ready."
    activate_venv
  elif [ "$1" == "-f" ]; then
    echo "Force reinstall requested. Removing existing virtual environment '$VENV_NAME'."
    rm -rf "$VENV_NAME"
    create_venv
    activate_venv
    install_dependencies
    touch "$BUILD_MARKER"
  else
    echo "Virtual environment '$VENV_NAME' exists but incomplete. Completing setup."
    # Fix shebangs in case they weren't fixed before
    find "$VENV_NAME/bin" -type f -executable | while read script; do
      if head -1 "$script" | grep -q python; then
        sed -i '1c#!/usr/bin/env python3' "$script"
      fi
    done
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

echo "Setup complete. Virtual environment '$VENV_NAME' is active with dependencies installed."