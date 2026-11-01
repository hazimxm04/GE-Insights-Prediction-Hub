import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
import os

EVAL_CSV = "eval/eval_set.csv"
OUTPUT_DIR = "finetune/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
# Ground truth = YOUR hand labels (human_*). The bare theme columns are Groq's scores.
LABELS = ["human_" + t for t in THEMES]

print(f"Loading {EVAL_CSV}...")
df = pd.read_csv(EVAL_CSV)
df = df.dropna(subset=LABELS)
print(f"Loaded {len(df)} labeled articles")

labels = df[LABELS].values.astype(np.float32)
assert set(np.unique(labels)) <= {0.0, 1.0}, "human_* labels must be 0/1"
print("Positives per theme:", dict(zip(THEMES, labels.sum(axis=0).astype(int).tolist())))

texts = df['text_snippet'].tolist()
labels = labels.tolist()

print("Loading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Creating dataset...")
dataset = Dataset.from_dict({"text": texts, "labels": labels})

print("Tokenizing...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Splitting 80/20...")
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

print(f"Saving to {OUTPUT_DIR}/...")
train_test_split["train"].save_to_disk(f"{OUTPUT_DIR}/train")
train_test_split["test"].save_to_disk(f"{OUTPUT_DIR}/test")
 
print(f"\nTrain: {len(train_test_split['train'])} articles")
print(f"Test: {len(train_test_split['test'])} articles")
print(f"Saved to {OUTPUT_DIR}/")