#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function activate_venv() {
  echo "Activating virtual environment: $VENV_NAME"
  source "$VENV_NAME/bin/activate"
  if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error activating virtual environment."
    exit 1
  fi
  echo "Virtual environment '$VENV_NAME' activated."
}

echo "Setting up virtual environment and installing dependencies..."
setup_script="$SCRIPT_DIR/../conf/setup.sh"
if [[ -x "$setup_script" ]]; then
  echo "Running setup script: $setup_script"
  "$setup_script" -f
else
  echo "Error: Setup script $setup_script is not executable or does not exist."
  exit 1
fi

download_script="$SCRIPT_DIR/download_blend_mean_temperature.sh"
if [[ -x "$download_script" ]]; then
  echo "Running download script: $download_script"
  "$download_script"
else
  echo "Error: Download script $download_script is not executable or does not exist."
  exit 1
fi

activate_venv
python "$SCRIPT_DIR/prepare_ecad_observations.py"
python "$SCRIPT_DIR/kriging/run_ok.py"
python "$SCRIPT_DIR/idw/run_idw.py"
python "$SCRIPT_DIR/inr/sinet/run_sinet.py"