# data_loader.py
import re

def load_sentimix_data(file_path):
    sentences = []
    current_sentence = {"tokens": [], "tags": [], "sentiment": None}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("meta"):
                # Save previous sentence if it exists
                if current_sentence["tokens"]:
                    sentences.append(current_sentence)
                
                # Start new sentence and extract sentiment (e.g., meta 123 positive)
                parts = line.split()
                current_sentence = {
                    "tokens": [], 
                    "tags": [], 
                    "sentiment": parts[2] if len(parts) > 2 else None
                }
            elif line:
                # Split token and language tag (e.g., "Love eng")
                parts = line.split("\t")
                if len(parts) == 2:
                    current_sentence["tokens"].append(parts[0])
                    current_sentence["tags"].append(parts[1])
        
        # Catch the last sentence
        if current_sentence["tokens"]:
            sentences.append(current_sentence)
            
    return sentences