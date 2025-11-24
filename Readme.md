
# FDP — Deep Learning: Perceptrons to RNNs (Colab-ready)

This folder contains the hands-on notebooks for a short FDP on practical deep learning. All notebooks are compatible with Google Colab — below are explicit, copy-paste-ready instructions for running them in Colab, handling datasets (Kaggle/Drive), and using GPU resources.

**What’s in this folder**
- Notebooks (examples): `Day_1_Intro_to_ANNs.ipynb`, `Day 2 - The Limits of MLPs  part-1 .ipynb`, `Day 2 - The Limits of MLPs  part-2.ipynb`, `Day_3_Optimizing_our_CNN.ipynb`, `Day_3_Optimizing_our_CNN_part_2.ipynb`, plus additional Day 3/4 RNN/transfer-learning notebooks.
- Notes: `Read me.txt`.
- Optional dataset folder: `dataset/` (used by Day 2/3 image notebooks).

**Open a notebook in Colab**
- From local: go to `https://colab.research.google.com` and upload the `.ipynb` file.
- From GitHub: open `https://colab.research.google.com/github/<username>/<repo>/blob/main/<path-to-notebook>.ipynb` (replace placeholders).

**Enable GPU**
- Menu → `Runtime` → `Change runtime type` → select `GPU` → `Save`.
- Verify with:
```python
import tensorflow as tf
print('TF', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
```

**Install packages (run once per session)**
```python
!pip install --upgrade pip
!pip install tensorflow pandas matplotlib scikit-learn kaggle
```

**Mount Google Drive (recommended for persistent datasets & checkpoints)**
```python
from google.colab import drive
drive.mount('/content/drive')
# Example persistent path: '/content/drive/MyDrive/dataset'
```

**Download Kaggle dataset (Fruits) in Colab**
Option A — upload the dataset zip to Drive and unzip into `MyDrive`.

Option B — use the Kaggle API (requires `kaggle.json`):
```bash
# upload kaggle.json to Colab (or keep in Drive and copy from Drive path)
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d <dataset-identifier> -p /content --unzip
```
After downloading, move or unzip files into Drive if you want persistence:
```bash
!cp -r /content/dataset /content/drive/MyDrive/dataset
```

**Paths & recommended pattern**
- Persistent data (recommended): `'/content/drive/MyDrive/dataset'`.
- Ephemeral (VM-only): `'/content/dataset'` (lost after VM reset).
- Use a top cell in each notebook to set `DATA_ROOT` depending on whether Drive is mounted.

**Colab helper snippet**
Put this at the top of a notebook to auto-detect Colab, mount Drive, and set `DATA_ROOT`:
```python
import os
def setup_colab(data_folder_name='dataset'):
    DATA_ROOT = './dataset'
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_ROOT = f'/content/drive/MyDrive/{data_folder_name}'
    except Exception:
        # not running in Colab or user skipped mounting
        DATA_ROOT = './dataset'
    print('DATA_ROOT =', DATA_ROOT)
    return DATA_ROOT

DATA_ROOT = setup_colab()
```

**Notebook map — quick descriptions**
- `Day_1_Intro_to_ANNs.ipynb`: Perceptrons and MLP basics; MNIST training example.
- `Day_2_The_MLP_Failure_(Part_1).ipynb` / `Day 2 - The Limits of MLPs  part-1 .ipynb`: Show MLP limitations on image tasks.
- `Day_2_Introduction_to_CNN_(part_2).ipynb` / `Day 2 - The Limits of MLPs  part-2.ipynb`: CNN intro, optimization and regularization techniques.
- `Day_3_Optimizing_our_CNN*.ipynb`: Several notebooks demonstrating CNN design, BatchNorm, Dropout, and training.
- `Day_3_Transfer_Learning_*.ipynb`: Transfer learning with pretrained models and fine-tuning.
- `Day_4_RNNs_for_Language_Translation_(Seq2Seq)*.ipynb`: Seq2Seq RNN/LSTM examples for translation and sequence tasks.

**Colab checklist (paste at top of each notebook)**
- Enable GPU runtime.
- Run package install cell.
- Mount Drive (if using persistent data).
- Set `DATA_ROOT` to Drive path when mounted.

**Common issues & fixes**
- Package/version errors: pin versions using a `requirements.txt` and install with `!pip install -r requirements.txt`.
- Dataset not found: confirm `DATA_ROOT` and whether the dataset was placed in Drive or the VM.
- GPU not visible: change runtime type to GPU and re-run the verification cell.

**Follow-ups I can implement**
- Create `requirements.txt` or `environment.yml` to pin versions.
- Add `check_data.py` that prints dataset folders and counts (useful when mounting Drive).
- Standardize notebook filenames (remove spaces/duplicates) and update README links.

Tell me which of the follow-ups you want and I'll implement it next.
