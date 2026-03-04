# PCL Detector

Detecting Patronising and Condescending Language (PCL) in online news articles about vulnerable communities — NLP Coursework, Imperial College London (2025–26).

The task is binary classification: given a paragraph, predict whether it contains PCL (label ≥ 2 on the Don't Patronise Me graded scale). The primary metric is **binary F₁ on the PCL class**.

## Repository Structure

```
main.ipynb          # All EDA, training, and evaluation
requirements.txt    # Python dependencies
assets/             # Raw dataset files (Don't Patronise Me)
data/               # Processed splits used in the notebook
BestModel/          # Saved model checkpoint (DeBERTa-v3-base + Focal Loss)
report/             # LaTeX source for the written report
outputs/            # Training output artefacts
```

## Setup

Requires **Python 3.10+**.

```bash
# 1. Clone the repository
git clone https://github.com/hardiv/pcl-detector.git
cd pcl-detector

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** the notebook detects CUDA / Apple MPS automatically and falls back to CPU. Training the proposed model on CPU is slow (~2 h); the saved `BestModel/` checkpoint is included so training can be skipped.

## Running the Notebook

```bash
jupyter notebook main.ipynb
# or open main.ipynb directly in VS Code
```

Run cells top-to-bottom. The notebook is structured as follows:

| Section | Description |
|---|---|
| Setup | Installs packages, imports, device detection |
| Data Preprocessing | Loads the Don't Patronise Me dataset and SemEval splits |
| Exploratory Data Analysis | Label distribution, lexical analysis, statistical profiling |
| Baseline Model | RoBERTa-base fine-tuned with standard cross-entropy |
| Proposed Model | DeBERTa-v3-base with focal loss, LLRD, and cosine schedule |
| Results | F₁ comparison on the official dev set |

If a saved model is found in `models/baseline/` or `models/proposed/`, training is skipped and the checkpoint is loaded directly. To retrain from scratch, delete the corresponding directory.

## Results

| Model | F₁ (PCL) |
|---|---|
| Baseline — RoBERTa-base, standard CE | 0.579 |
| Proposed — DeBERTa-v3-base, focal loss | 0.000 |

The proposed model collapsed to a majority-class predictor during training. See the report (`report/main.tex`) and the Results cell in the notebook for a full diagnosis and proposed fixes.
