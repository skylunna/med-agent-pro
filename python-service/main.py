import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import Annotated
import dotenv

dotenv.load_dotenv()

app = FastAPI(title="Python AI Engine", version="0.1.0")


# ✅ 延迟初始化：仅在请求时触发
def get_llm_client() -> AsyncOpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )


# 类型别名，方便路由使用
LLMClient = Annotated[AsyncOpenAI, Depends(get_llm_client)]


class ChatRequest(BaseModel):
    question: str
    session_id: str


@app.post("/agent/query")
async def query(req: ChatRequest, client: LLMClient):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-turbo"),
            messages=[{"role": "user", "content": req.question}],
            temperature=0.7,
            max_tokens=1024,
            timeout=30.0,
        )
        return {
            "answer": response.choices[0].message.content,
            "session_id": req.session_id,
            "source": "llm_api",
            "usage": response.usage.model_dump() if response.usage else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")
