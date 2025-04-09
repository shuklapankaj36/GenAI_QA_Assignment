from transformers import BertTokenizer, BertForQuestionAnswering
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
        if isinstance(doc_id_or_content, str):
            return doc_id_or_content
            
        try:
            if hasattr(doc_id_or_content, 'item'):
                doc_id = doc_id_or_content.item()
            else:
                doc_id = int(doc_id_or_content)
                
            return self.documents[doc_id]
        except (IndexError, ValueError, TypeError, AttributeError):
            return None

# Load model
model_path = "../model/fine_tuned/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

with open("./data/qa_dataset.json", "r") as f:
    data = json.load(f)

docstore_dict = {i: item["context"] for i, item in enumerate(data)}
docstore = SimpleDocstore(docstore_dict)

if len(docstore_dict) != len(data):
    raise ValueError("Docstore size doesn't match dataset size")

index_to_docstore_id = {i: int(i) for i in range(len(data))}

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the FAISS index 
index = faiss.read_index("faiss_index/index.faiss")

db = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

def answer_question(question):
    try:

        docs = db.similarity_search(question, k=3)
        if not docs:
            return "No relevant documents found"
        
        context = " ".join([doc.page_content for doc in docs])
        
        inputs = tokenizer(context, question, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1 
        answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
        
        return {
            "answer": answer,
            "context": context,
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
