from fastapi import FastAPI
from pydantic import BaseModel
from rag_answer import answer_query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以改成 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.post("/ask")

def ask_question(request:QueryRequest):
    answer = answer_query(request.query)
    return {"answer": answer}