#!/bin/bash

VENV_NAME="exp1"

function create_venv() {
  echo "Creating virtual environment: $VENV_NAME"
  python3 -m venv "$VENV_NAME"
  if [ $? -ne 0 ]; then
    echo "Error creating virtual environment."
    exit 1
  fi
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
  echo "Installing pip dependencies..."
  pip install climatrix[plot,ml,optim,ok] jupyterlab
  if [ $? -ne 0 ]; then
    echo "Error installing dependencies."
    deactivate
    exit 1
  fi
  echo "Dependencies installed successfully."
}

if [ -d "$VENV_NAME" ]; then
  if [ "$1" == "-f" ]; then
    echo "Force reinstall requested. Removing existing virtual environment '$VENV_NAME'."
    rm -rf "$VENV_NAME"
    create_venv
    activate_venv
    install_dependencies
  else
    echo "Virtual environment '$VENV_NAME' already exists."
    exit 1
  fi
else
  create_venv
  activate_venv
  install_dependencies
fi

echo "Setup complete. Virtual environment '$VENV_NAME' is active with dependencies installed."