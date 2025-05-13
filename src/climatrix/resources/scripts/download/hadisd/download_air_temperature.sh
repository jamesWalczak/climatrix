# Observations are downloaded from https://catalogue.ceda.ac.uk/uuid/2a01faf75de64308b2bf4c7b43d393ef/
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadISD/subdaily/HadISDTable/r1/v3-4-1-2024f/v20241231/tas/tas_HadISD_HadOBS_19310101-20250101_v3-4-1-2024f.nc?download=1"
NETCDF_FILE="/tmp/air_temperature.nc"
TARGET_DIR="$SCRIPT_DIR/../../../data/hadisd"

mkdir -p "$TARGET_DIR"

echo "Downloading NetCDF file from $URL..."
curl -L "$URL" -o "$NETCDF_FILE"

if [[ $? -ne 0 ]]; then
  echo "Error: Failed to download file from $URL"
  exit 1
fi

echo "Moving file to $TARGET_DIR..."
mv "$NETCDF_FILE" "$TARGET_DIR/"

echo "Done! File is in: $TARGET_DIR"