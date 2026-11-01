import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np
import os

TRAIN_DIR = "finetune/data/train"
OUTPUT_DIR = "finetune/distilbert_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 3e-5

print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"Loading training data from {TRAIN_DIR}...")
train_dataset = load_from_disk(TRAIN_DIR)
print(f"Train set: {len(train_dataset)} examples")

# Compute class weights to handle imbalance
labels = np.array(train_dataset["labels"])
pos_counts = labels.sum(axis=0)
neg_counts = len(labels) - pos_counts
pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)
print(f"Class weights: {pos_weight.tolist()}")

print(f"Loading {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)
print(f"Model size: {model.num_parameters():,} parameters")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_steps=20,
    weight_decay=0.01,
    logging_dir="finetune/logs",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("\nTraining...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"\n? Model saved to {OUTPUT_DIR}")
