# 📰 Fake News Detection Using BERT with Explainability

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Natural Language Processing system for detecting deceptive news articles using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model paired with **token-level gradient attribution** and an interactive **Streamlit** multi-page web application.

---

## 🌟 Key Features

- **Fine-Tuned Transformer Classifier:** Utilizes `bert-base-uncased` with dynamic attention masking, AdamW optimization, linear learning rate scheduling with warmup, and early stopping.
- **Data Leakage Mitigation:** Solves the notorious publisher dateline shortcut (e.g. `(Reuters) - ` in real news articles) by stripping metadata artifacts to force semantic understanding.
- **Hardware-Aware Acceleration:** Auto-detects NVIDIA CUDA GPU acceleration with automatic mixed precision (`torch.amp.autocast`) tailored for 6 GB GPUs, with graceful CPU fallback.
- **Rigorous Evaluation:** Held-out test set evaluation with Accuracy, Precision, Recall, Macro F1, ROC-AUC, annotated Confusion Matrix, and ROC/PR curve serialization.
- **Genuine Model Explainability:** Embedding-level Gradient $\times$ Input attribution with subword aggregation, outputting color-coded heatmaps and influential word rankings.
- **Multi-Page Web Interface:**
  - `app.py`: Real-time news classification with sample loaders, confidence meters, and inline attribution heatmaps.
  - `pages/1_Model_Performance.py`: Benchmark metrics dashboard with Plotly confusion matrices, ROC/PR curves, and loss progression.
  - `pages/2_Explainability.py`: Deep-dive token importance explorer and sensitivity attribution analysis.

---

## 🏗️ Project Architecture

```
fake-news-detection/
├── app.py                         # Streamlit main entry point (Detector)
├── pages/
│   ├── 1_Model_Performance.py    # Benchmark & metrics dashboard
│   └── 2_Explainability.py       # Deep-dive word attribution explorer
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset ingestion and label assignment
│   ├── preprocessing.py          # Cleaning, dateline stripping, splits, NewsDataset
│   ├── model.py                  # BERT model wrapper, device resolution, serialization
│   ├── train.py                  # Mixed-precision training loop, early stopping
│   ├── evaluate.py               # Comprehensive test-set evaluation pipeline
│   ├── predict.py                # Cached production inference wrapper
│   └── explain.py                # Embedding gradient attribution engine & HTML renderer
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis & leakage discovery
│   └── 02_model_training.ipynb   # Step-by-step training & evaluation demonstration
├── scripts/
│   └── download_data.py          # Automated kagglehub dataset downloader
├── artifacts/
│   ├── model/                    # Serialized model weights, tokenizer, config
│   └── metrics/                  # Real evaluation metrics, ROC/PR curves, loss logs
├── tests/
│   ├── test_preprocessing.py     # Preprocessing & split integrity tests
│   ├── test_prediction.py        # Output format & probability sum tests
│   └── smoke_test.py             # Pipeline readiness test
├── data/                         # Fake.csv & True.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Ensure Python 3.11+ is installed. Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Acquisition

Automatically download and stage the Kaggle dataset (`Fake.csv` and `True.csv`):

```bash
python scripts/download_data.py
```

### 3. Model Training

Train the BERT model using mixed precision:

```bash
python src/train.py
```

*Configurable parameters in `src/train.py`: `max_length` (default 256), `batch_size` (default 16), `epochs` (default 3), `learning_rate` (default 2e-5).*

### 4. Model Evaluation

Compute genuine held-out evaluation metrics and plot data:

```bash
python src/evaluate.py
```

### 5. Launch Web Application

Run the Streamlit application:

```bash
streamlit run app.py
```

### 6. Run Test Suite

```bash
pytest tests/ -v
```

---

## 🔍 Explainability & Interpretability

Rather than relying on opaque predictions, the model exposes its internal representations via **Input Embedding Gradient Attribution**:

$$\text{Attribution}(w_i) = \sum_{d} e_{i, d} \cdot \frac{\partial \mathcal{L}_{\text{target}}}{\partial e_{i, d}}$$

- **Subword Reconstruction:** BERT tokenizes rare or complex words into subwords (e.g. `["un", "##pre", "##cedented"]`). The engine aggregates attribution back into readable full words.
- **Sensational Vocabulary:** Sensational and emotionally charged vocabulary (`"shocking"`, `"suppressed"`, `"covert"`) naturally triggers positive attribution toward the **Fake** class.
- **Formal Journalism:** Neutral, descriptive phrasing (`"announced"`, `"spokesperson"`, `"unanimously"`) triggers positive attribution toward the **Real** class.

---

## ⚠️ Responsible AI & Classifier Disclaimer

This model is a statistical pattern classifier trained on a historical news corpus. **A prediction of "Fake" or "Real" indicates stylistic and linguistic similarity to labeled articles in the dataset, not empirical factual verification of real-world claims.** Always consult reputable primary sources and professional fact-checking organizations.
