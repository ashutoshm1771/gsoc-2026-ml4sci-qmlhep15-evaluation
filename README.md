# ML4SCI QMLHEP GSoC 2026 — Evaluation Tasks

**Applicant:** Ashutosh Mishra
**Project:** QMLHEP15 — Quantum Resource Analysis and Benchmarking
**Organization:** [ML4SCI](https://ml4sci.org/) (Machine Learning for Science)

---

## Overview

This repository contains my solutions to the evaluation tasks for the QMLHEP (Quantum Machine Learning in High Energy Physics) project under Google Summer of Code 2026 with ML4SCI.

## Repository Structure

```
├── qmlhep-tasks/          # Common QMLHEP evaluation tasks (I–XII)
│   ├── assets/             # Architecture diagrams (QGNN, QViT)
│   ├── diagrams/           # KAN diagrams
│   └── data/               # Datasets (downloaded by user — see below)
├── qmlhep15/               # QMLHEP15 project-specific evaluation tasks (1–3)
├── requirements-quantum.txt
└── requirements-orchestral.txt
```

## Tasks

### Common QMLHEP Tasks — [`qmlhep-tasks/`](qmlhep-tasks/)

| # | Task | Notebook |
|---|------|----------|
| I | Quantum Computing (circuits, SWAP test) | [`Task_I_Quantum_Computing.ipynb`](qmlhep-tasks/Task_I_Quantum_Computing.ipynb) |
| II | Classical GNN for Quark/Gluon Jet Classification | [`Task_II_Classical_GNN.ipynb`](qmlhep-tasks/Task_II_Classical_GNN.ipynb) |
| III | Commentary on Quantum Machine Learning | [`Task_III_Open_Task.pdf`](qmlhep-tasks/Task_III_Open_Task.pdf) |
| IV | Quantum GAN for Signal/Background Separation | [`Task_IV_QGAN.ipynb`](qmlhep-tasks/Task_IV_QGAN.ipynb) |
| V | Quantum Graph Neural Network Design | [`Task_V_QGNN.ipynb`](qmlhep-tasks/Task_V_QGNN.ipynb) |
| VI | Quantum Representation Learning (contrastive + SWAP test) | [`Task_VI_Quantum_Representation_Learning.ipynb`](qmlhep-tasks/Task_VI_Quantum_Representation_Learning.ipynb) |
| VII | Equivariant Quantum Neural Networks (Z2 x Z2) | [`Task_VII_Equivariant_QNN.ipynb`](qmlhep-tasks/Task_VII_Equivariant_QNN.ipynb) |
| VIII | Vision Transformer + Quantum ViT Discussion | [`Task_VIII_Vision_Transformer.ipynb`](qmlhep-tasks/Task_VIII_Vision_Transformer.ipynb) |
| IX | Kolmogorov-Arnold Networks + Quantum KAN Discussion | [`Task_IX_KAN.ipynb`](qmlhep-tasks/Task_IX_KAN.ipynb) |
| X | Jets as Graphs (DeepFalcon) | [`Task_X_Jets_as_Graphs.ipynb`](qmlhep-tasks/Task_X_Jets_as_Graphs.ipynb) |
| XI | PQC Parameter Estimation with Neural Networks | [`Task_XI_PQC_Embedding.ipynb`](qmlhep-tasks/Task_XI_PQC_Embedding.ipynb) |
| XII | Reinforcement Learning for PQC Optimization | [`Task_XII_RL_PQC.ipynb`](qmlhep-tasks/Task_XII_RL_PQC.ipynb) |

### QMLHEP15 Project-Specific Tasks — [`qmlhep15/`](qmlhep15/)

| # | Task | Notebook |
|---|------|----------|
| 1 | Orchestral AI Setup & Hello World Tool | [`QMLHEP15_Orchestral_Tasks.ipynb`](qmlhep15/QMLHEP15_Orchestral_Tasks.ipynb) |
| 2 | Agent-Driven QNN Training on MNIST | *(same notebook)* |
| 3 | Agent-Based Hyperparameter Optimization | *(same notebook)* |

---

## Setup Guide

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- [Git](https://git-scm.com/)
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/ashutoshm1771/gsoc-2026-ml4sci-qmlhep15-evaluation.git
cd gsoc-2026-ml4sci-qmlhep15-evaluation
```

### Step 2: Create the Environments

Two separate conda environments are required. The common QMLHEP tasks use Python 3.11, while the QMLHEP15-specific tasks require Python 3.13+ (an Orchestral AI requirement).

#### Environment A — Common QMLHEP tasks (Tasks I–XII)

```bash
conda create -n qmlhep python=3.11 -y
conda activate qmlhep
pip install -r requirements-quantum.txt
```

`torch_geometric` requires additional companion packages. After installing the requirements, run:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

> **Note:** Replace `2.5.0` with your installed PyTorch version (`python -c "import torch; print(torch.__version__)"`). If you have a CUDA GPU, replace `cpu` with your CUDA version (e.g., `cu121`). See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

Finally, register the environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name qmlhep --display-name "QMLHEP (Python 3.11)"
```

#### Environment B — QMLHEP15-specific tasks (Tasks 1–3)

```bash
conda create -n orchestral python=3.13 -y
conda activate orchestral
pip install -r requirements-orchestral.txt
```

Register the Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name orchestral --display-name "Orchestral (Python 3.13)"
```

### Step 3: Download the Datasets

Some notebooks require datasets that are too large to include in the repository. The table below lists every dataset, which notebooks need it, and where to place the downloaded file.

| Dataset | Size | Notebooks | Auto-downloads? |
|---------|------|-----------|----------------|
| MNIST | ~60 MB | VI, VIII, IX, XI, XII, QMLHEP15 | Yes (via `torchvision`) |
| Quark/Gluon Jets (NPZ) | ~107 MB | II, IV | Yes (from Zenodo on first run) |
| DeepFalcon Jet Images (HDF5) | ~673 MB | X (DeepFalcon) | **No — manual download required** |

#### MNIST (automatic)

No action needed. The notebooks download MNIST automatically via `torchvision` on first run. The data is saved to `qmlhep-tasks/data/MNIST/` or `qmlhep15/data/MNIST/`.

#### Quark/Gluon Jets — Tasks II, IV (automatic)

No action needed. Both notebooks download `QG_jets.npz` from [Zenodo](https://zenodo.org/records/3164691) automatically if the file is not present. The data is saved to `qmlhep-tasks/data/QG_jets.npz`.

To download manually instead:

1. Go to [https://zenodo.org/records/3164691](https://zenodo.org/records/3164691)
2. Download `QG_jets.npz` (~107 MB)
3. Place it in `qmlhep-tasks/data/QG_jets.npz`

#### DeepFalcon Jet Images — Task X (manual download required)

This dataset must be downloaded manually before running the DeepFalcon notebook.

1. Go to [https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing)
2. Download the HDF5 file (~673 MB)
3. Place it in `qmlhep-tasks/data/`

The notebook looks for any `.hdf5` file in that directory, so the exact filename does not matter.

### Step 4: Set Up the API Key (QMLHEP15 Tasks Only)

The QMLHEP15 Orchestral notebook uses [Groq](https://groq.com/) as the LLM backend (free tier, no credit card required). You need a Groq API key.

#### Get a Groq API Key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up or log in (Google/GitHub sign-in supported)
3. Navigate to **API Keys** in the left sidebar
4. Click **Create API Key**, give it a name, and copy the key

#### Save the API Key

Create a file named `.env` in the repository root:

**macOS / Linux:**
```bash
echo 'GROQ_API_KEY=gsk_your_key_here' > .env
```

**Windows (PowerShell):**
```powershell
Set-Content -Path .env -Value 'GROQ_API_KEY=gsk_your_key_here'
```

**Windows (Command Prompt):**
```cmd
echo GROQ_API_KEY=gsk_your_key_here > .env
```

Replace `gsk_your_key_here` with your actual key. The `.env` file is listed in `.gitignore` and will not be committed.

> **Alternative:** Instead of a `.env` file, you can export the key as an environment variable in your terminal session before launching Jupyter:
>
> macOS/Linux: `export GROQ_API_KEY=gsk_your_key_here`
> Windows PowerShell: `$env:GROQ_API_KEY = "gsk_your_key_here"`
> Windows CMD: `set GROQ_API_KEY=gsk_your_key_here`

### Step 5: Run the Notebooks

Launch Jupyter and select the correct kernel for each notebook:

```bash
jupyter notebook
```

| Notebooks | Kernel to select |
|-----------|-----------------|
| All notebooks in `qmlhep-tasks/` | **QMLHEP (Python 3.11)** |
| `qmlhep15/QMLHEP15_Orchestral_Tasks.ipynb` | **Orchestral (Python 3.13)** |

#### Run order

The notebooks are independent — you can run them in any order. However, Task XII builds conceptually on Task XI, and the QMLHEP15 tasks build sequentially (1 → 2 → 3) within a single notebook.

---

## Troubleshooting

**`torch_geometric` import errors:**
Make sure you installed the companion packages (`torch_scatter`, `torch_sparse`, etc.) with the correct PyTorch and CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

**`No HDF5 files found` in DeepFalcon notebook:**
Download the dataset manually (see Step 3) and place the `.hdf5` file in `qmlhep-tasks/data/`.

**`GROQ_API_KEY not found` in QMLHEP15 notebook:**
Create a `.env` file in the repository root or export the key in your terminal (see Step 4).

**`ModuleNotFoundError: No module named 'orchestral'`:**
Make sure you are using the `orchestral` kernel (Python 3.13). Orchestral AI does not support Python < 3.13.

**Kernel not showing in Jupyter:**
Re-run the `python -m ipykernel install` command for the relevant environment (see Step 2).

---

## Related Work

- [Quantum Encoding Atlas](https://github.com/encoding-atlas/quantum-encoding-atlas) — my library implementing 16 quantum data encodings with benchmarking across PennyLane, Qiskit, and Cirq

## License

MIT
