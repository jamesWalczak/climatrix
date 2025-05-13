# Observations are downloaded from https://www.ecad.eu/dailydata/predefinedseries.php
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Non-blended ECA dataset
URL="https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_tg.zip"
ZIP_FILE="/tmp/ecad_blend.zip"
TARGET_DIR="$SCRIPT_DIR/../../../data/ecad_blend"

mkdir -p "$TARGET_DIR"

echo "Downloading zip file from $URL..."
curl -L "$URL" -o "$ZIP_FILE"

if [[ $? -ne 0 ]]; then
  echo "Error: Failed to download file from $URL"
  exit 1
fi

echo "Unzipping file to $TARGET_DIR..."
unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

if [[ $? -ne 0 ]]; then
  echo "Error: Failed to unzip $ZIP_FILE"
  exit 1
fi

echo "Done! Files are in: $TARGET_DIR"
