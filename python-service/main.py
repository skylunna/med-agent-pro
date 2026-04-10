from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Python AI Engine")

class ChatRequest(BaseModel):
    question: str
    session_id: str

@app.post("/agent/query")
async def query(req: ChatRequest):
    # 模拟后续接入 LangChain/vLLM 的占位逻辑
    await asyncio.sleep(0.5)
    return {
        "answer": f"[AI模拟回复] 关于“{req.question}”，建议参考最新肿瘤指南...",
        "session_id": req.session_id,
        "source": "mock"
    }