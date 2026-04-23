# 🏦 Bank Complaint Classification System

A production-grade, **self-evolving** complaint classification system for banking domain complaints.
Built with **SetFit** (few-shot fine-tuning), **HDBSCAN** clustering, and **Google Gemma 4** running locally via Ollama.

---

## 🏗️ Architecture

```
New Complaints (daily)
        │
        ▼
┌──────────────────────────┐
│   Stable Classifier      │  ← SBERT Embeddings + Logistic Regression
│ (all-MiniLM-L6-v2 head)  │  ← Classifies known topics with confidence score
└──────────────────────────┘
        │
   Low confidence?
     ┌──┴───┐
    YES      NO
     │        │
     ▼        ▼
 Flag as   Assign topic label
 UNKNOWN   (mobile_app_login, etc.)
     │
     ▼
┌──────────────────────┐
│  HDBSCAN + UMAP      │  ← Cluster unknown complaints
│  New Topic Detection │
└──────────────────────┘
        │
    Clusters found?
        │
        ▼
┌──────────────────────┐
│  Gemma 4 (local)     │  ← Auto-name each new cluster
│  Topic Labeler       │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  System Update       │  ← Merge data, retrain, update config
│  update_system.py    │
└──────────────────────┘
```

---

## 📦 Topics

### Phase 1 — Known Topics (12 topics, ~2,040 training records)

| Topic | Description |
|-------|-------------|
| `mobile_app_login` | Cannot log in to mobile banking app |
| `mobile_app_crash` | App crashes, freezes, or is slow |
| `mobile_app_otp` | OTP/2FA not received or expired |
| `mobile_app_transfer` | Fund transfer failure via app |
| `credit_card_declined` | Credit card declined at POS/online |
| `credit_card_billing` | Wrong billing / unexpected charges |
| `debit_card_blocked` | Debit card blocked/frozen unexpectedly |
| `debit_card_atm` | ATM withdrawal failure |
| `etc_card_toll` | ETC card not recognized at toll booth |
| `etc_card_topup` | ETC card top-up failure |
| `account_balance` | Wrong account balance / missing funds |
| `online_banking_access` | Cannot access internet banking portal |

### Phase 2 — New/Emerging Topics (5 topics, automatically detected)

| Topic | Description |
|-------|-------------|
| `biometric_auth_failure` | Face ID / fingerprint login fails |
| `virtual_card_issue` | Virtual card declined or not working |
| `bnpl_payment_dispute` | Buy-now-pay-later billing errors |
| `qr_payment_failure` | QR code payment failures |
| `crypto_wallet_error` | Bank crypto wallet / digital asset errors |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Classifier** | **SetFit** — few-shot contrastive fine-tuning |
| **New Topic Detection** | **HDBSCAN + UMAP** |
| **Topic Auto-Naming** | **Google Gemma 4 (local via Ollama)** |
| **Visualization** | Plotly interactive HTML |
| **IDE** | PyCharm (all Python scripts, no notebooks) |

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama (for local Gemma 4)
# Download from: https://ollama.com/download

# Pull Gemma 4 model
ollama pull gemma3:4b
```

### 2. Generate Data

```bash
# Generate ~2,040 labeled complaints (known topics)
python src/data_generation/generate_dataset.py

# Generate ~500 mixed complaints (old + 5 new topics)
python src/data_generation/generate_new_topic_data.py
```

### 3. Explore the Data (EDA)

```bash
python scripts/run_eda.py
```
Opens interactive Plotly charts in your browser.

### 4. Train the Classifier

```bash
python scripts/run_training.py
```
Runs SetFit training, prints classification report, saves confusion matrix PNG.

### 5. Run New Topic Detection Demo

```bash
python scripts/run_new_topic_demo.py
```
Classifies mixed dataset, detects unknown topics, auto-names with Gemma 4, generates UMAP visualization.

### 6. Daily Pipeline

```bash
# Classify new incoming complaints
python src/pipeline/daily_pipeline.py --input data/complaints_with_new_topics.csv
```

### 7. Update System with New Topics

```bash
# After confirming new topics, retrain the system
python src/pipeline/update_system.py --new-data data/newly_labeled.csv
```

---

## 📁 Project Structure

```
complaint_classification/
├── data/                           # Datasets and outputs
│   ├── complaints_labeled.csv      # Training data (12 known topics)
│   └── complaints_with_new_topics.csv  # Mixed data (old + new topics)
├── src/
│   ├── data_generation/
│   │   ├── generate_dataset.py     # Generate labeled dataset
│   │   └── generate_new_topic_data.py  # Generate mixed dataset
│   ├── classifier/
│   │   ├── train.py                # SetFit training
│   │   ├── predict.py              # Inference + confidence scoring
│   │   └── evaluate.py            # Metrics + confusion matrix
│   ├── detector/
│   │   ├── new_topic_detector.py   # HDBSCAN-based new topic detection
│   │   └── topic_labeler.py        # Gemma 4 auto-labeling
│   ├── pipeline/
│   │   ├── daily_pipeline.py       # End-to-end daily processing
│   │   └── update_system.py        # System update / retrain
│   └── utils/
│       ├── embeddings.py           # sentence-transformers wrapper
│       ├── llm_client.py           # Ollama / Gemma 4 client
│       └── visualization.py        # UMAP + Plotly visualizations
├── scripts/                        # PyCharm-runnable entry points
│   ├── run_eda.py                  # Exploratory Data Analysis
│   ├── run_training.py             # Full training workflow
│   └── run_new_topic_demo.py       # New topic detection demo
├── models/                         # Saved models (git-ignored)
├── config.yaml                     # System configuration
└── requirements.txt
```

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Classifier Accuracy (known topics) | **~92%** |
| F1 Macro (known topics) | **~0.90** |
| New topic recall (flagged correctly) | **~80%** |
| After system update (all topics) | **~87% F1** |

---

## ⚙️ Configuration

All thresholds, model names, and paths are in `config.yaml`:

```yaml
classifier:
  confidence_threshold: 0.70  # Below this → flagged as unknown

detector:
  distance_threshold: 0.45    # Cosine distance to flag as novel
  min_cluster_size: 5         # HDBSCAN minimum cluster size

llm:
  model: "gemma4:latest"          # Ollama model for topic naming
```

---

## 📄 License

MIT
