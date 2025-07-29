## ⬇️ How to Prepare the Data

Since we don’t have the rights to redistribute the ECA&D dataset, you’ll need to download it manually  
from the official webpage: [ECA&D Dataset](https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_tg.zip)  
📥 *Alternatively, you can use our custom Bash script located at* `../scripts/download_blend_mean_temperature.sh`:

> [!CAUTION]\
> Make sure the `TARGET_DIR` variable is set correctly before running the script!


```bash
# Observations are downloaded from https://www.ecad.eu/dailydata/predefinedseries.php
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Non-blended ECA dataset
URL="https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_tg.zip"
ZIP_FILE="/tmp/ecad_blend.zip"
TARGET_DIR="$SCRIPT_DIR/data/ecad_blend"

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
```

▶️ Then, run the Python script located at ../scripts/prepare_ecad_observations.py:

```python
python prepare_ecad_observations.py
```

📁 This will generate the train/validation/test samples in the ../data directory, ready for use in the notebooks!