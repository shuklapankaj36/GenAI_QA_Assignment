from fastapi import FastAPI
from pydantic import BaseModel
from model.run_inference import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def get_answer(request: QuestionRequest):
    response = answer_question(request.question)
    return {"answer": response}

