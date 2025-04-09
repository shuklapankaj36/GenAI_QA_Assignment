import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

with open("./data/qa_dataset.json", "r") as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_data(example):
    inputs = tokenizer(example["context"], example["question"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = tokenizer(example["answer"], truncation=True, padding="max_length", max_length=128)["input_ids"]
    return inputs

dataset = Dataset.from_list(data).map(encode_data)

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
training_args = TrainingArguments(
    output_dir="../model/fine_tuned/",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="../model/logs/",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("../model/fine_tuned/")
tokenizer.save_pretrained("../model/fine_tuned/")
