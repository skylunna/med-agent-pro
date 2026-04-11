import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

app = FastAPI(title="Python AI Engine", version="0.1.0")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)


class ChatRequest(BaseModel):
    question: str
    session_id: str


@app.post("/agent/query")
async def query(req: ChatRequest):
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
            "usage": response.usage.dict() if response.usage else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")
