# Deep Learning FDP — From Perceptrons to Advanced Applications

Welcome! This repository contains the materials for a short hands-on Faculty Development Programme (FDP) focused on practical deep learning: from perceptrons and MLPs to efficient CNNs and RNNs.

This repo uses Jupyter Notebooks to present theory, code, and exercises. Each notebook is written to be followable in a workshop setting and runnable on a modern Python environment.

**Notebooks in this folder**
- `Day_1_Intro_to_ANNs.ipynb` — Foundations and building a Multi-Layer Perceptron (MLP).
- `Day 2 - The Limits of MLPs  part-1 .ipynb` — Demonstrating MLP limitations and introductory CNN concepts.
- `Day 2 - The Limits of MLPs  part-2.ipynb` — CNN optimization and regularization techniques.
- `Day_3_Optimizing_our_CNN.ipynb` — Building and improving CNN architectures.
- `Day_3_Optimizing_our_CNN_part_2.ipynb` — Transfer learning and advanced strategies.

**Workshop Overview**
- **Goal:** Give participants practical experience building and training neural networks, understanding their limitations, and applying modern techniques (regularization, transfer learning, RNNs).
- **Format:** Short theory sections followed by hands-on notebooks. Notebooks are organized by day/topic.

**High-level Topics**
- Day 1 — Perceptrons & MLPs: linear vs. nonlinear classifiers, activation functions, loss, and a TensorFlow/Keras MLP example (MNIST).
- Day 2 — Optimization & Efficient CNNs: backpropagation recap, optimizers, dropout, batch normalization, and building efficient CNNs.
- Day 3 — Advanced CNNs & Transfer Learning: model design, parameter calculation, and using pre-trained backbones (e.g. EfficientNet) for fast, accurate models.
- Day 4 — RNNs: sequence models, LSTM/GRU, and a text classification example (IMDB).
- Day 5 — Time-series & Advanced Applications: applying RNNs/LSTMs to regression/time-series forecasting problems.

**Setup & Prerequisites**
- **Python:** 3.8 or newer (Anaconda recommended).
- **Recommended:** create a virtual environment or conda env.

Example (conda):
```
conda create -n dl-fdp python=3.9 -y; conda activate dl-fdp
pip install --upgrade pip
pip install tensorflow pandas matplotlib scikit-learn jupyterlab
```

Or using pip/venv:
```
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install tensorflow pandas matplotlib scikit-learn jupyterlab
```

**Datasets**
- **MNIST & IMDB:** loaded automatically via Keras datasets inside the notebooks; internet required on first run.
- **Fruits dataset (Fresh vs Rotten):** download manually from Kaggle (search "Fresh and Rotten Fruits Dataset").
    - Download and extract so the folder name is `dataset/` in the repository root.
    - Expected structure: `dataset/train/...` and `dataset/test/...` (check the notebook for exact paths used).
- **Time-series examples:** some notebooks load CSVs from URLs or expect a local CSV — check the specific notebook.

**How to Run**
- Open the repo folder in VS Code or start JupyterLab from the repo root:
```
jupyter lab
```
- Open the notebook you want (e.g., `Day_1_Intro_to_ANNs.ipynb`) and run cells sequentially.

**Recommended Workflow**
- Start with `Day_1_Intro_to_ANNs.ipynb` to ensure environment and basic TensorFlow code runs.
- For image-based notebooks using the Fruits dataset, confirm the `dataset/` folder is present in the repo root.

**Files & Structure (expected)**
```
<repo root>/
├─ Day_1_Intro_to_ANNs.ipynb
├─ Day 2 - The Limits of MLPs  part-1 .ipynb
├─ Day 2 - The Limits of MLPs  part-2.ipynb
├─ Day_3_Optimizing_our_CNN.ipynb
├─ Day_3_Optimizing_our_CNN_part_2.ipynb
├─ Readme.md
├─ Read me.txt
└─ dataset/   <-- Optional: place here for Day 2 & 3 image notebooks
```

**Notes & Tips**
- If you rename notebooks, update references inside the files or this README accordingly.
- GPU: If you have a compatible GPU and want faster training, install `tensorflow-gpu` (or the appropriate TF build) and verify TensorFlow sees the GPU.
- If a notebook expects a different dataset path, open its top cells — paths are documented there.

**Questions / Contact**
If you want me to:
- Rename notebooks to consistent filenames;
- Create a `requirements.txt` or `environment.yml` for reproducible installs; or
- Add a small demo runner script to execute key notebooks,
tell me which option and I will implement it.

Enjoy the workshop!
