#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define variables
URL="https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_tg.zip"
ZIP_FILE="/tmp/ecad_blend.zip"
TARGET_DIR="$SCRIPT_DIR/../data/ecad_blend"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# --- PRIMARY CHECK ---
# If the target directory already has files, we assume the work is done and exit.
if [ -n "$(ls -A "$TARGET_DIR")" ]; then
	  echo "Data is already present in $TARGET_DIR. Nothing to do. ✅"
	    exit 0
    fi

    # If we reach here, the target directory is empty and we need to fetch the data.
    echo "Target directory is empty. Proceeding with data retrieval..."

    # --- Download Logic ---
    if [ ! -f "$ZIP_FILE" ]; then
	      echo "Downloading zip file from $URL..."
	        curl -L "$URL" -o "$ZIP_FILE"

		  if [[ $? -ne 0 ]]; then
			      echo "Error: Failed to download file from $URL"
			          exit 1
				    fi
			    else
				      echo "Zip file already exists. Skipping download. 👍"
			      fi

			      # --- Unzip Logic ---
			      echo "Unzipping file to $TARGET_DIR..."
			      unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

			      if [[ $? -ne 0 ]]; then
				        echo "Error: Failed to unzip $ZIP_FILE"
					  exit 1
				  fi

				  echo "Done! Files are in: $TARGET_DIR"
