from transformers import BertTokenizer, BertForQuestionAnswering
from langchain_core.documents import Document
import torch
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
import json

# Direct content handling docstore
class SimpleDocstore:
    def __init__(self, documents):
        self.documents = list(documents.values())
        
    def search(self, doc_id_or_content):
        try:
            if isinstance(doc_id_or_content, str):
                return doc_id_or_content
                
            doc_id = int(doc_id_or_content)
            if 0 <= doc_id < len(self.documents):
                return self.documents[doc_id]
            return None
        except (ValueError, TypeError, AttributeError):
            return None

# Load model
import os
from pathlib import Path

# Load fine-tuned model from correct path
model_dir = str(Path(__file__).parent.parent / "model" / "fine_tuned")
base_model = "bert-base-uncased"
print(f"Loading model from: {model_dir}")

# Explicitly specify all required files
try:
    tokenizer = BertTokenizer.from_pretrained(
        model_dir,
        config_file=os.path.join(model_dir, "config.json"),
        tokenizer_file=os.path.join(model_dir, "tokenizer.json"),
        vocab_file=os.path.join(model_dir, "vocab.txt"),
        tokenizer_config_file=os.path.join(model_dir, "tokenizer_config.json")
    )
    model = BertForQuestionAnswering.from_pretrained(model_dir)
    print("Successfully loaded fine-tuned model with explicit file paths")
except Exception as e:
    print(f"Error loading with explicit paths: {str(e)}")
    print(f"Falling back to base model {base_model}")
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForQuestionAnswering.from_pretrained(base_model)

with open("./data/qa_dataset.json", "r") as f:
    data = json.load(f)

docstore_dict = {i: item["context"] for i, item in enumerate(data)}
docstore = SimpleDocstore(docstore_dict)

if len(docstore_dict) != len(data):
    raise ValueError("Docstore size doesn't match dataset size")

index_to_docstore_id = {i: int(i) for i in range(len(data))}

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FAISS index
try:
    # Try loading existing index
    index = faiss.read_index("faiss_index/index.faiss")
    db = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
except:
    
    documents = [Document(page_content=ctx) for ctx in docstore_dict.values()]
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local("faiss_index")

def answer_question(question, ground_truth=None):
    try:
        # Retrieve relevant documents
        docs = db.similarity_search(question, k=3)
        if not docs:
            return {
                "answer": "No relevant documents found",
                "context": "",
                "confidence": 0.0,
                "metrics": {}
            }
        
        context = " ".join([doc.page_content for doc in docs])
        
        # Get model prediction
        inputs = tokenizer(context, question, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        
        # Calculate confidence scores
        start_probs = torch.softmax(outputs.start_logits, dim=1)
        end_probs = torch.softmax(outputs.end_logits, dim=1)
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1
        confidence = (start_probs[0, answer_start].item() + end_probs[0, answer_end-1].item()) / 2
        
        answer = tokenizer.decode(
            inputs['input_ids'][0][answer_start:answer_end],
            skip_special_tokens=True
        ).strip()
        if not answer or answer.lower() in ['', 'unknown', 'i don\'t know']:
            answer = "I don't know"
        
        # Calculate metrics if ground truth provided
        metrics = {}
        if ground_truth:
            from evaluate import exact_match, f1_score_qa
            metrics = {
                "exact_match": exact_match(answer, ground_truth),
                "f1_score": f1_score_qa(answer, ground_truth)
            }
        
        return {
            "answer": answer,
            "context": context,
            "confidence": confidence,
            "metrics": metrics,
            "source": "Generated from local model"
        }
        
    except ValueError as e:
        return f"Document retrieval: {str(e)}"
    except RuntimeError as e:
        return f"Model inference error: {str(e)}"
    except Exception as e:
        return f"Unexpected error processing question: {str(e)}"

if __name__ == "__main__":
    print(answer_question("What is Hugging Face?"))
