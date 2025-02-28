import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import csv
import optuna
import itertools

# Disable wandb integration if needed
os.environ["WANDB_DISABLED"] = "true"

# Set the Hugging Face token as an environment variable
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Load data
train_df = pd.read_csv("train_data_SMM4H_2025_Task_1.csv")[["id", "text", "label"]]
val_df = pd.read_csv("dev_data_SMM4H_2025_Task_1.csv")[["id", "text", "label"]]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = [int(label) for label in labels]  # Ensure labels are integers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = label
        return item

# Create datasets
train_dataset = CustomDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = CustomDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

# Define output file for evaluation results
output_file = "evaluation_results_roberta.tsv"
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Seed", "Accuracy", "F1", "Precision", "Recall"])

def compute_metrics(eval_pred, seed=None):
    logits, labels = eval_pred

    # Print shapes to debug
    print(f"DEBUG: Logits shape: {logits.shape}, Labels shape: {labels.shape}")

    # Handle binary classification (logits might be shape [batch_size, 1] or [batch_size, 2])
    if logits.shape[-1] == 1:  # Sigmoid activation case
        predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int).flatten()
    else:  # Softmax activation case for multi-class
        predictions = np.argmax(logits, axis=-1)

    # Ensure shape consistency
    if labels.shape != predictions.shape:
        print(f"⚠️ Shape mismatch detected: Labels {labels.shape}, Predictions {predictions.shape}")
        min_length = min(len(labels), len(predictions))
        labels = labels[:min_length]
        predictions = predictions[:min_length]

    # Print some example values
    print(f"DEBUG: Sample Labels: {labels[:5]}")
    print(f"DEBUG: Sample Predictions: {predictions[:5]}")

    # Compute metrics
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Save confusion matrix per seed
    if seed is not None:
        seed_cm_dir = f"./confusion_matrices/seed_{seed}"
        os.makedirs(seed_cm_dir, exist_ok=True)
        cm_file = os.path.join(seed_cm_dir, "confusion_matrix.csv")
        np.savetxt(cm_file, cm, delimiter=",")
        print(f"✅ Confusion matrix saved at: {cm_file}")

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Hyperparameter search space
param_search = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'batch_size': [16],
    'num_epochs': [3, 5, 7],
    'weight_decay': [0.0, 0.01]
}

best_f1 = 0
best_params = None

for lr, bs, epochs, wd in itertools.product(param_search['learning_rate'], param_search['batch_size'], param_search['num_epochs'], param_search['weight_decay']):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=wd,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-large",
        num_labels=2,
        problem_type="single_label_classification",
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    if eval_result["eval_f1"] > best_f1:
        best_f1 = eval_result["eval_f1"]
        best_params = {"learning_rate": lr, "batch_size": bs, "num_epochs": epochs, "weight_decay": wd}

print(f"Best Hyperparameters: {best_params}")

seeds = [19, 100, 2025, 314, 2718]
for seed in seeds:
    print(f"\nRunning with seed: {seed}")
    final_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f'./results_seed_{seed}',
            evaluation_strategy='epoch',
            learning_rate=best_params['learning_rate'],
            per_device_train_batch_size=best_params['batch_size'],
            per_device_eval_batch_size=best_params['batch_size'],
            num_train_epochs=best_params['num_epochs'],
            weight_decay=best_params['weight_decay'],
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, seed)
    )
    final_trainer.train()
    final_eval_result = final_trainer.evaluate()
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow([seed, final_eval_result["eval_accuracy"], final_eval_result["eval_f1"], final_eval_result["eval_precision"], final_eval_result["eval_recall"]])
    
    model_save_path = f"./saved_models/seed_{seed}"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"✅ Model saved at: {model_save_path}")

print(f"\n✅ All results saved to {output_file}")
