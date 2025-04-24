# Two‑Step Political Bias Detection

*A lightweight research prototype for detecting political bias in English‑language news using a two‑stage RoBERTa pipeline plus a TF‑IDF baseline, all wrapped in a Flask + React framework.*

---

## 1 . What’s inside?

```
.
├── backend *.py              # Flask API, training / inference utilities
│   ├── app.py                # REST endpoints
│   ├── main.py               # CLI wrapper (train / evaluate / predict / explain)
│   ├── train.py              # end‑to‑end training script (incl. data‑augmentation)
│   ├── evaluate.py           # batch evaluation helper
│   ├── predict.py            # lightweight inference functions
│   ├── explain.py            # SHAP / LIME explanations
│   └── ...
├── data/                     # datasets (BABE + others)
│   ├── babe/                 # original + augmented parquet files
│   └── phrasebias_data/      # phrase‑level experimental corpora
├── models/                   # generic helper classes (tokeniser, dataset)
├── savedmodels/              # fine‑tuned transformer checkpoints (created after training)
│   ├── bias_model/
│   └── leaning_model/
├── tfidf_models/             # baseline models & vectorisers (auto‑generated)
├── frontend/                 # React (Vite) single‑page app
│   ├── src/components/       # Chakra‑UI building blocks
│   └── src/pages/            # Home / Train / Evaluate views
└── uploaded_data/            # temporary uploads accepted via `/uploadfile`
```

---

## 2 . Quick‑start (local)

### 2.1 Back‑end (Python ≥ 3.10)

```bash
# 1 — create & activate a virtual‑env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2 — install deps (incl. torch + transformers)
pip install -r requirements.txt

# 3 — (optional) train from scratch ‑ takes ~30 min on a modern GPU
python main.py --mode train --data data/babe/train-00000-of-00001.parquet --epochs 3 --batch_size 8 --cleaning

# 4 — start the Flask API
python app.py   # defaults to http://localhost:5000
```

> **Note** Pre‑trained checkpoints can be dropped into `savedmodels/` to skip step 3.

### 2.2 Front‑end (Node ≥ 18)

```bash
cd frontend
npm i                 # or pnpm i
npm run dev           # served at http://localhost:5173
```
The SPA proxies API calls to port 5000 by default (configured in `vite.config.js`).

---

## 3 . REST API

| Route | Method | Payload (JSON) | Purpose |
|-------|--------|----------------|---------|
| `/predict` | POST | `{ "text": "<article or tweet>" }` | Returns model prediction string, e.g. `[BIAS: Biased (0.92)] | [LEANING: Left (0.71)]` |
| `/train` | POST | `{ epochs, batch_size, lr_bias, lr_lean, cleaning }` | Launches training job (synchronous for now) |
| `/evaluate` | POST | `{ cleaning }` | Runs evaluation on held‑out set, returns macro Precision/Recall/F1 |
| `/uploadfile` | POST‑multipart | `data_file=<parquet>` | Adds additional data under `uploaded_data/` |

All endpoints return `application/json`; errors come back with `{ "error": "…" }`.

---

## 4 . Command‑line helper

```bash
# one‑off prediction (skips Flask)
python main.py --mode predict --text "The policy will devastate small businesses."

# SHAP explanation (prints values)
python main.py --mode shap --text "The left is misguided."
```

---

## 5 . How it works

### 5.1 Two‑stage transformer
1. **Bias detector**   RoBERTa‑base fine‑tuned to binary *Neutral vs Biased*.
2. **Leaning classifier**   RoBERTa‑base (separate head) fine‑tuned on *Left / Right / Center* — invoked **only** if step 1 says *Biased*.

Class imbalance is mitigated via:
* **Center‑oversampling**   back‑translation (EN↔FR/DE) + T5 paraphrasing (`llm_augmentation.py`).
* **Weighted loss**   inverse‑frequency weights in cross‑entropy.
* Early‑stopping on validation loss.

### 5.2 Baseline
A TF‑IDF + Logistic‑Regression pipeline (`tfidf.py`) offers interpretability and a sanity‑check.

---

## 6 . Results (BABE test split)

| Task | Model | Accuracy | Macro F1 |
|------|-------|----------|----------|
| Bias (Neutral / Biased) | TF‑IDF + LogReg | 0.74 | 0.73 |
| Bias | RoBERTa (+T5 aug) | **0.88** | **0.88** |
| Leaning (L/R/C) | TF‑IDF + LogReg | 0.62 | 0.53 |
| Leaning | RoBERTa (+T5 aug) | **0.75** | **0.67** |

Center recall improved from **0.22 → 0.58** after augmentation.

---

## 7 . Re‑training on your own data

1. Convert your corpus to **Parquet** with at least:
   * `text`   (raw article)
   * `label`  (0 = Neutral, 1 = Biased)
   * `type`   ("left" | "right" | "center" – ignored when `label = 0`)
2. Place it anywhere (e.g. `mydata.parquet`).
3. Run `python main.py --mode train --data mydata.parquet --overwrite`.

---

## 8 . Dependencies

* **Python**: torch | transformers | scikit‑learn | pandas | flask | tqdm | contractions | joblib | shap | lime | imblearn
* **Node / React**: React 18 | Chakra‑UI | React‑Router‑Dom | Vite

Exact pinned versions live in `requirements.txt` and `frontend/package.json`.

---

## 9 . Known limitations & next steps

* CPU inference is **slow** – deploy with a GPU for real‑time use.
* Data comes largely from BABE (≈3 k sentences); domain shift to full articles may hurt accuracy.
* Explanation module is experimental (Kernel‑SHAP on ≥ 512‑token inputs is expensive).
* No async task queue – `/train` blocks the Flask worker.

Feel free to open issues or PRs with suggestions!

---

## 10 . License

MIT © 2025 Theo Petkov & contributors

