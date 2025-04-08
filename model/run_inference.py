from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_path = "../model/fine_tuned/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local("faiss_index", embeddings)

def answer_question(question):
    docs = db.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    inputs = tokenizer(context, question, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1  # +1 to include the end token
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
    return answer

if __name__ == "__main__":
    question = "What does Hugging Face provide?"
    print("Question:", question)
    print("Answer:", answer_question(question))
