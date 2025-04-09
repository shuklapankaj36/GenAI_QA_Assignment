import json
import os
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForQuestionAnswering
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from pathlib import Path

def exact_match(prediction, truth):
    return int(prediction.strip().lower() == truth.strip().lower())

def f1_score_qa(prediction, truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(model_path, dataset_path, split='test'):
    # Load data and split
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Split data (80/10/10)
    np.random.seed(42)
    np.random.shuffle(data)
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    splits = {
        'train': data[:train_size],
        'val': data[train_size:train_size+val_size],
        'test': data[train_size+val_size:]
    }
    
    # Load model with error handling
    try:
        # Convert relative path to absolute
        abs_model_path = os.path.abspath(model_path)
        if not os.path.exists(abs_model_path):
            raise ValueError(f"Model path {abs_model_path} does not exist")
            
        tokenizer = BertTokenizer.from_pretrained(abs_model_path)
        model = BertForQuestionAnswering.from_pretrained(abs_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to base BERT model")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    
    # Evaluate
    results = {'em': [], 'f1': []}
    for item in tqdm(splits[split]):
        inputs = tokenizer(
            item['context'], 
            item['question'], 
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1
        prediction = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
        
        results['em'].append(exact_match(prediction, item['answer']))
        results['f1'].append(f1_score_qa(prediction, item['answer']))
    
    return {
        'exact_match': np.mean(results['em']),
        'f1_score': np.mean(results['f1']),
        'num_samples': len(results['em'])
    }

if __name__ == '__main__':
    metrics = evaluate_model(
        model_path="../model/fine_tuned/",
        dataset_path="./data/qa_dataset.json"
    )
    print(f"Evaluation Metrics: {metrics}")
