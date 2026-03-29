# config.py — Centralised configuration

DATA_PATH       = "uid_train.txt"
CLEAN_DATA_PATH = "cleaned_data.json"
PHONETIC_PATH   = "phonetic_data.json"
VOCAB_PATH      = "vocabs.json"

LANG_MAP = {
    "eng":  "english",
    "hin":  "hindi",
    "rest": "neutral",
}

# Language tag IDs (must match dataset.py LANG_TAG_MAP)
LANG_TAG_IDS = {
    "eng":  1,
    "hin":  2,
    "rest": 3,
    0:      0,   # padding
}
