import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model.run_inference import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def get_answer(request: QuestionRequest):
    try:
        response = answer_question(request.question)
        if isinstance(response, dict):
            return response
        return {
            "answer": response,
            "context": "",
            "source": "Generated from local model"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
