# GenAI QA Assignment

A question answering system using Hugging Face transformers and FAISS for efficient similarity search.

## Features
- Fine-tuned BERT model for question answering
- FAISS index for fast semantic search
- Streamlit web interface
- FastAPI backend

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GenAI_QA_Assignment.git
cd GenAI_QA_Assignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## File Structure

```
├── api/                  # FastAPI backend
│   └── app.py
├── data/                 # Dataset files
│   └── qa_dataset.json
├── faiss_index/          # FAISS index files
│   └── index.faiss
├── model/                # Model training and inference
│   ├── create_faiss_index.py
│   ├── fine_tune.py
│   ├── run_inference.py
│   └── fine_tuned/       # Fine-tuned model checkpoints
└── ui/                   # Streamlit frontend
    └── chatbot.py
```

## Usage

### Web Interface
```bash
streamlit run ui/chatbot.py
```

### API Server
```bash
uvicorn api.app:app --reload
```

### Creating FAISS Index
```bash
python model/create_faiss_index.py
```

### Fine-tuning Model
```bash
python model/fine_tune.py
```

## Configuration

Edit `data/qa_dataset.json` to add your own question-answer pairs.

## Requirements
- Python 3.8+
- See requirements.txt for full dependency list
