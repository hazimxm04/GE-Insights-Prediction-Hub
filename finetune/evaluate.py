"""
Evaluate fine-tuned DistilBERT model against baseline (Groq).

Loads:
  - eval_set.csv (Groq scores + human labels)
  - finetune/distilbert_finetuned/ (fine-tuned model)

Outputs:
  - eval/finetuned_results.json (F1, precision, recall per theme)
  - Prints comparison: Groq vs Fine-tuned
"""

import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
import os

# Paths
EVAL_CSV = "eval/eval_set.csv"
MODEL_DIR = "finetune/distilbert_finetuned"
BASELINE_JSON = "eval/baseline_results.json"
OUTPUT_JSON = "eval/finetuned_results.json"

# Themes
THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
THRESHOLD = 0.5  # Probability threshold for multi-label

print(f"Loading {EVAL_CSV}...")
df = pd.read_csv(EVAL_CSV)
df = df.dropna(subset=THEMES)
print(f"Loaded {len(df)} labeled articles")

# Load model
print(f"Loading fine-tuned model from {MODEL_DIR}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(THEMES),
    problem_type="multi_label_classification"
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Predict
print("Running predictions...")
predictions = []
for text in df['text_snippet']:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        preds = (probs >= THRESHOLD).astype(int)
    predictions.append(preds)

predictions = np.array(predictions)
human_labels = df[THEMES].values.astype(int)

# Metrics
def compute_metrics(predictions, labels, theme_names):
    results = {}
    for i, theme in enumerate(theme_names):
        pred_i = predictions[:, i]
        label_i = labels[:, i]

        tp = np.sum((pred_i == 1) & (label_i == 1))
        fp = np.sum((pred_i == 1) & (label_i == 0))
        fn = np.sum((pred_i == 0) & (label_i == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(label_i)

        results[theme] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": int(support)
        }

    macro_f1 = np.mean([results[t]["f1"] for t in theme_names])
    results["macro_f1"] = round(macro_f1, 3)
    return results

finetuned_metrics = compute_metrics(predictions, human_labels, THEMES)

# Print comparison
print("\n" + "="*60)
print("FINE-TUNED MODEL RESULTS")
print("="*60)
print(f"{'Theme':<20} {'Precision':<12} {'Recall':<12} {'F1':<8} {'Support'}")
print("-"*60)
for theme in THEMES:
    m = finetuned_metrics[theme]
    print(f"{theme:<20} {m['precision']:<12} {m['recall']:<12} {m['f1']:<8} {m['support']}")
print("-"*60)
print(f"{'MACRO AVERAGE':<20} {'':<12} {'':<12} {finetuned_metrics['macro_f1']:<8}")

# Compare to baseline
if os.path.exists(BASELINE_JSON):
    print("\n" + "="*60)
    print("COMPARISON: GROQ vs FINE-TUNED")
    print("="*60)
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    print(f"{'Theme':<20} {'Groq F1':<15} {'Fine-tuned F1':<15} {'Improvement'}")
    print("-"*60)
    for theme in THEMES:
        groq_f1 = baseline.get("per_theme", baseline).get(theme, {}).get("f1", 0)
        ft_f1 = finetuned_metrics[theme]["f1"]
        diff = ft_f1 - groq_f1
        print(f"{theme:<20} {groq_f1:<15} {ft_f1:<15} {diff:+.3f}")

    groq_macro = baseline.get("macro_f1", 0)
    ft_macro = finetuned_metrics["macro_f1"]
    print("-"*60)
    print(f"{'MACRO AVERAGE':<20} {groq_macro:<15} {ft_macro:<15} {ft_macro - groq_macro:+.3f}")

# Save
print(f"\nSaving to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w") as f:
    json.dump(finetuned_metrics, f, indent=2)

print(f"✓ Done. Results saved.")
