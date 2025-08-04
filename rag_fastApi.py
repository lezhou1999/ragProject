from fastapi import FastAPI
from pydantic import BaseModel
from rag_answer import answer_query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.options("/ask")
async def options_handler():
    return {"msg": "ok"}

# 🔥 建议显式写前端的域名，而不是 *
origins = [
    "http://localhost:5173",   # React dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # 👈 明确指定
    allow_credentials=True,
    allow_methods=["*"],           # 支持 POST/OPTIONS
    allow_headers=["*"],           # 允许所有 header
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    answer = answer_query(request.query)
    return {"answer": answer}
