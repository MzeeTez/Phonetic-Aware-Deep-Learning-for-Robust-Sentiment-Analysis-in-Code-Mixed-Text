# Phonetic-Aware Deep Learning for Robust Sentiment Analysis in Code-Mixed Text

A research-grade deep learning framework for **Hinglish (Hindi-English) sentiment analysis** using a **dual-channel architecture**.  
This project tackles high linguistic variance and noisy Romanization in code-mixed social media text through **phonetic encoding** and **cross-modal attention fusion**.

---

## 🚀 Key Research Contributions

This **Enhanced Research Edition** introduces several architectural advancements beyond standard NLP baselines:

### 🔤 Phonetic-Aware Encoding
- Decomposes tokens into phoneme sequences  
  Example: `"pyaar"` → `["PY", "AA", "R"]`
- Uses:
  - ITRANS-to-Akshar mapping (Hindi)
  - G2P (Grapheme-to-Phoneme) for English

---

### 🔁 Cross-Modal Attention Fusion
- Multi-head attention mechanism:
  - Word-channel queries attend over phoneme-channel keys
- Enables learning of phonetic relevance for semantic context

---

### ⚖️ Gated Fusion Mechanism
Replaces naive concatenation with a learnable gating function:

```math
output = \sigma(W[a;b]) \odot a + (1 - \sigma(W[a;b])) \odot b
```

- Dynamically balances semantic and phonetic features per token

---

### 🏷 Language-Tag Embeddings
- Embeds tokens with language tags:
  - `eng`, `hin`, `rest`
- Provides inductive bias for code-switching

---

### 🧠 Advanced Regularization
- Variational Dropout (Gal & Ghahramani, 2016)
- Label Smoothing  
- Reduces overfitting and noisy label impact

---

## 🛠 Architecture Overview

### 🔹 Word Channel
```
Word Embeddings + Language Tags
        ↓
2-layer Stacked BiLSTM
        ↓
Multi-Head Self-Attention
```

### 🔹 Phoneme Channel
```
Phoneme Encoder (Embedding + Mean Pooling)
        ↓
2-layer Stacked BiLSTM
        ↓
Multi-Head Self-Attention
```

### 🔹 Fusion & Classification
```
Cross-Modal Attention
        ↓
Dual-Stage Gated Fusion
        ↓
GELU Classifier + LayerNorm
```

---

## 📊 Experimental Results

### Full Model Performance

| Metric       | Score   |
|--------------|--------|
| Accuracy     | 56.98% |
| Macro F1     | 57.25% |
| Negative F1  | 61.07% |
| Positive F1  | 55.27% |

---

### 🔬 Ablation Study Highlights

| Condition                     | Macro F1 | Δ Change |
|------------------------------|----------|----------|
| Full Model                   | 0.5725   | —        |
| − Cross-modal attention      | 0.5711   | -0.0014  |
| − Phoneme + Cross-attention  | 0.5715   | -0.0010  |

---

## 💻 Getting Started

### 📦 Prerequisites

```bash
pip install torch g2p_en indic-transliteration tqdm numpy matplotlib seaborn
```

---

### ▶️ Execution Pipeline

Run scripts in sequence:

```bash
python preprocess.py          # Clean raw text data
python phonetic_encoder.py   # Generate phoneme sequences
python vocab_builder.py      # Build vocabularies (~15k words target)
python train.py              # Train model (AdamW + Cosine Annealing)
python evaluate.py           # Evaluation + visualizations
```

---

## 📂 Project Structure

```
├── model.py              # Dual-channel LSTM model
├── phonetic_encoder.py  # ITRANS + G2P encoding logic
├── ablation_study.py    # Component evaluation suite
├── config.py            # Configurations & constants
├── predict.py           # Interactive inference
```

---

## 📜 References

- Gal & Ghahramani (2016) — Variational Dropout  
- Loshchilov & Hutter (2019) — AdamW Optimizer  
- Prabhu & Allahverdyan (2021) — SentMix Dataset  

---

## 🧪 Future Improvements

- Transformer-based hybrid (BERT + phoneme fusion)
- Better phoneme alignment using IPA
- Larger Hinglish datasets / semi-supervised training
- Contrastive learning for code-mixed embeddings

---

## 🤝 Contribution

Feel free to fork, improve, and submit PRs.  
This project is intended for **research and experimentation** in multilingual NLP.

---
