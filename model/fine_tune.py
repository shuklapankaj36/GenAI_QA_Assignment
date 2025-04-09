import json
import numpy as np
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load and split data
with open("./data/qa_dataset.json", "r") as f:
    data = json.load(f)

# Split data (80/10/10)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

def encode_data(example):
    # Tokenize both question and context
    inputs = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Find answer in context
    answer = example["answer"]
    answer_tokens = tokenizer.tokenize(answer)
    context_tokens = tokenizer.tokenize(example["context"])
    
    # Find answer span in context
    start_position = 0
    end_position = 0
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
            start_position = i
            end_position = i + len(answer_tokens) - 1
            break
    
    inputs["start_positions"] = start_position
    inputs["end_positions"] = end_position
    return inputs

# Create datasets
train_dataset = Dataset.from_list(train_data).map(encode_data)
val_dataset = Dataset.from_list(val_data).map(encode_data)
test_dataset = Dataset.from_list(test_data).map(encode_data)

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Enhanced training arguments
training_args = TrainingArguments(
    output_dir="../model/fine_tuned/",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="../model/logs/",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2
)

def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    
    # Get start and end logits
    start_logits, end_logits = predictions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)
    
    # Calculate exact match and F1 scores
    exact_matches = []
    f1_scores = []
    
    for i in range(len(start_preds)):
        pred_start = start_preds[i]
        pred_end = end_preds[i]
        true_start = labels[i][0]
        true_end = labels[i][1]
        
        # Calculate exact match
        exact_matches.append(int(pred_start == true_start and pred_end == true_end))
        
        # Calculate F1 score
        pred_tokens = set(range(pred_start, pred_end + 1))
        true_tokens = set(range(true_start, true_end + 1))
        
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        precision = len(pred_tokens & true_tokens) / len(pred_tokens)
        recall = len(pred_tokens & true_tokens) / len(true_tokens)
        
        if (precision + recall) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
    
    return {
        "exact_match": np.mean(exact_matches),
        "f1_score": np.mean(f1_scores),
        "eval_loss": p.loss
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train with evaluation
trainer.train()

# Save best model
trainer.save_model("../model/fine_tuned/")
tokenizer.save_pretrained("../model/fine_tuned/")

# Save test set for final evaluation
with open("../model/fine_tuned/test_set.json", "w") as f:
    json.dump(test_data, f)
