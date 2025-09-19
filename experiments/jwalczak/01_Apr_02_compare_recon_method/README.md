# Experiment Framework with Docker
This repository provides a reproducible Docker-based environment for running experiments.  
It ensures that datasets are properly downloaded/prepared and that results are stored locally.

---

## 📂 Project Structure

```
01_Apr_02_compare_recon_method/
├── conf/                  # Configurations 
│   └── setup.sh           # Script to create virtual environment & install deps
│   └── requirements.txt   # Dependencies to install
├── data/                  # Datasets (mounted from host)
├── src/                   # Source code (Python modules)
├── results/               # Experiment outputs (mounted from host)
├── notebook/              # Jupyter notebooks
├── scripts/               # Utility scripts
|   ├── idw/               # Scripts related to IDW method
|   ├── kriging/           # Scripts related to OK method
|   ├── inr/               # Scripts related to INR method(s)
|   |   ├── sinet/         # Scripts for SiNET method (the proposed one)
|   |   └── mmgn/          # Scripts for MMGN method (the referenced one)
│   ├── prepare_ecad_observations.py
├── entrypoint.sh
├── Dockerfile
└── experiment.def
```

---

## Prerequisites
Before running the experiment, you need to set `CLIMATRIX_EXP_DIR` environmental 
variable with the absolute path to the directory where experiments' files are stored 
and where data and results will be saved (`01_Apr_02_compare_recon_method` directory).
If you are using containerized application (Docker or Apptainer), set it to `/app`:

```bash
export CLIMATRIX_EXP_DIR="/app"
```

## 💻 Local run

### 1.Setup virtual environment
First, let us create virtual environment and install all necessery dependencies.
To do so, navigate to `conf` directory and run `setup.sh` script.
It will create under `${CLIMATRIX_EXP_DIR}/conf` a directory with Python virual environemnt called `exp1`.

```bash
bash conf/setup.sh
```

After that, activate your virtual environment by calling:

```bash
source $CLIMATRIX_EXP_DIR/conf/exp1/bin/activate
```

### 2. Download & prepare data
To download data, navigate to `scripts` under your `$CLIMATRIX_EXP_DIR` directory and run `download_blend_mean_temperature.sh`:

```bash
python scripts/prepare_ecad_observations.py
```

Remember to have virtual enviroment activated!

### 4. Run experiments
Finally, we can run experiments:

```bash
python scripts/idw/run_idw.py
python scripts/kriging/run_ok.py
python scripts/inr/sinet/run_sinet.py
python scripts/inr/mmgn/run_mmgn.py
```


## 🐳 Docker Setup

### 1. Build the Docker Image

From inside the `images/` directory:

```bash
cd 01_Apr_02_compare_recon_method
docker build -t my-experiments -f images/Dockerfile .
```

This creates an image named `my-experiments`.

---

### 2. Run the Container

To run experiments while keeping **data** and **results** synced with your host:

```bash
docker run --rm \
  -v $(pwd)/../data:/app/data \
  -v $(pwd)/../results:/app/results \
  my-experiments
```

- `-v $(pwd)/../data:/app/data` → Mounts host `01_Apr_02_compare_recon_method/data` into container `/app/data`  
- `-v $(pwd)/../results:/app/results` → Mounts host `01_Apr_02_compare_recon_method/results` into container `/app/results`  
- `--rm` → Automatically removes the container after execution  

---

# 🛰️ Using Apptainer (Singularity)

### 1. Build the Apptainer Image

From inside the `images/` directory:

```bash
cd 01_Apr_02_compare_recon_method/images
apptainer build experiment.sif experiment.def
```

If you don’t have root privileges, you may need:
```bash
apptainer build --fakeroot experiment.sif experiment.def
```

---

### 2. Run the Container

```bash
apptainer run \
  --bind ../data:/app/data \
  --bind ../results:/app/results \
  experiment.sif
```

- `--bind ../data:/app/data` → Mounts host `01_Apr_02_compare_recon_method/data`  
- `--bind ../results:/app/results` → Mounts host `01_Apr_02_compare_recon_method/results`  
- Results will appear in your host `01_Apr_02_compare_recon_method/results`.

---

## ⚙️ What Happens Inside the Container

When the container starts (`CMD` in Dockerfile):

1. **Download data** (if not already present)  
   ```bash
   scripts/download_blend_mean_temperature.sh
   ```
   - Downloads data into `/app/data` (mapped to host `01_Apr_02_compare_recon_method/data`).

2. **Prepare dataset**  
   ```bash
   python scripts/prepare_ecad_observations.py
   ```
   - Processes raw data and stores ready-to-use files in `/app/data`.

3. **Run experiments**  
   ```bash
   bash scripts/run_experiments.sh
   ```
   - Executes experiments and saves outputs in `/app/results` (mapped to host `01_Apr_02_compare_recon_method/results`).

---

## 📌 Virtual Environment

- A Python virtual environment is created in `/app/conf/exp1` during image build.  
- It’s automatically used for all Python commands via environment variables:
  ```dockerfile
  ENV VIRTUAL_ENV=/app/conf/exp1
  ENV PATH="$VIRTUAL_ENV/bin:$PATH"
  ```
- You do **not** need to run `source activate`; `python` and `pip` already refer to the venv.

---

## 📊 Viewing Results

After the run, check results on your host machine:

```bash
ls 01_Apr_02_compare_recon_method/results
```

Outputs generated by the experiments will appear here.

---

## 📝 License

The code and experiments are published under MIT license.