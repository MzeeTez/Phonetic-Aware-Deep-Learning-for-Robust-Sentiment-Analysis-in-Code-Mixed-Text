# preprocess.py
import re
import json
from data_loader import load_sentimix_data
from config import DATA_PATH, CLEAN_DATA_PATH

def clean_token(token):
    # 1. Lowercase
    token = token.lower()
    # 2. Remove URLs
    token = re.sub(r'http\S+|www\S+|https\S+', '', token, flags=re.MULTILINE)
    # 3. Remove @mentions
    token = re.sub(r'@\w+', '', token)
    # 4. De-elongation: reduce 3+ repeated chars to 2 (e.g., "loooove" -> "loove")
    # We keep 2 because some words naturally have doubles (like "feed")
    token = re.sub(r'(.)\1{2,}', r'\1\1', token)
    return token.strip()

def main():
    print("Loading raw data...")
    raw_data = load_sentimix_data(DATA_PATH)
    
    cleaned_data = []
    for entry in raw_data:
        processed_tokens = []
        for word in entry["tokens"]:
            cleaned_word = clean_token(word)
            if cleaned_word: # Only keep if not empty after cleaning
                processed_tokens.append(cleaned_word)
        
        # Update entry with cleaned tokens
        entry["tokens"] = processed_tokens
        if processed_tokens: # Skip empty sentences
            cleaned_data.append(entry)

    print(f"Processed {len(cleaned_data)} sentences. Saving to {CLEAN_DATA_PATH}...")
    with open(CLEAN_DATA_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

if __name__ == "__main__":
    main()