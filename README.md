# Enhanced Hinglish Sentiment Analysis — Research Edition

## What was improved and why

### 1. Language-tag embedding (model.py)
The original model ignored the `eng/hin/rest` tags at inference time.  
We now embed these tags into a 32-d vector and **concatenate them with word embeddings** before the LSTM. This gives the model explicit code-switching signals — a strong inductive bias for Hinglish.

### 2. 2-layer stacked BiLSTM + variational dropout
Deeper LSTMs capture longer-range dependencies.  
Variational dropout (same mask per timestep) is empirically superior to standard dropout for RNNs (Gal & Ghahramani, 2016) — it regularises the recurrent connections rather than individual activations.

### 3. Multi-head self-attention (per channel)
Each BiLSTM output is processed by a 4-head self-attention layer with layer normalisation. This allows the model to attend to *multiple* relevant positions simultaneously (e.g. both a negation word and the sentiment word in the same tweet).

### 4. Cross-modal attention fusion
**Word channel queries attend over the phoneme channel** keys/values.  
This is the key research contribution: instead of simply concatenating the two context vectors, we let the model learn *which phoneme patterns are relevant for each word-level context*. For Hinglish, this matters because the same surface form (e.g. "pyar") has different phoneme realisations depending on Romanisation convention.

### 5. Gated fusion (replaces naive concatenation)
Two cascaded learnable gates replace the fixed concat:  
`output = σ(W[a;b]) ⊙ a + (1 − σ(W[a;b])) ⊙ b`  
The gate is data-driven — the model learns how much to trust phoneme vs word information for each sample.

### 6. Label smoothing (train.py)
Prevents the model from becoming over-confident on noisy Hinglish labels. Smoothing ε=0.1 redistributes 10% of probability mass uniformly across all classes.

### 7. AdamW + cosine LR + linear warmup
- AdamW decouples weight decay from the gradient update (Loshchilov & Hutter, 2019)
- 2-epoch warmup avoids large early gradient steps
- Cosine annealing reaches near-zero LR at end of training (no cliff)

### 8. Stratified train/val/test split (dataset.py)
The baseline mixed all data into a single train set. We now hold out 10% val + 10% test with **class-balanced** sampling, ensuring the test distribution matches the train distribution.

### 9. Token augmentation (dataset.py)
- 10% random token masking → UNK (forces the model to use context)
- 5% adjacent token swap (simulates word-order variation in informal text)

### 10. Temperature-scaled confidence (predict.py)
Raw softmax probabilities are overconfident. We divide logits by a learned temperature T=1.3 (calibrate on val set) before softmax, giving better-calibrated probability estimates.

### 11. Improved vocab builder (vocab_builder.py)
- Hapax legomena (freq < 2) removed — these hurt generalisation
- Vocabulary raised from 10k → 15k words
- OOV rate diagnostic printed at build time

---

## Run order

```bash
python preprocess.py          # raw → cleaned_data.json
python phonetic_encoder.py    # cleaned → phonetic_data.json
python vocab_builder.py       # phonetic → vocabs.json
python train.py               # → best_model.pth
python evaluate.py            # → confusion_matrix.png, attention_heatmaps.png
python predict.py             # interactive demo
```

## Expected improvements over baseline

| Metric      | Baseline (est.) | Enhanced (est.) |
|-------------|-----------------|-----------------|
| Macro F1    | ~0.55–0.60      | ~0.65–0.72      |
| Accuracy    | ~0.60–0.65      | ~0.68–0.75      |
| ROC-AUC     | —               | ~0.82–0.88      |

Exact numbers depend on your dataset size and label quality.

## Citation / References
- Gal & Ghahramani (2016). A theoretically grounded application of dropout in RNNs.
- Loshchilov & Hutter (2019). Decoupled weight decay regularization.
- Prabhu & Allahverdyan (2021). SentMix: Code-mixing sentiment analysis.
- Chatterjee et al. (2020). SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text.
