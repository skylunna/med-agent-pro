import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import dotenv
from rag_engine import MedicalRAG

dotenv.load_dotenv()
app = FastAPI(title="Python AI Engine", version="0.2.0")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)
rag = MedicalRAG()

@app.on_event("startup")
async def startup():
    try:
        if os.path.exists("data/medical_guide.txt"):
            rag.ingest("data/medical_guide.txt")
        else:
            print("⚠️ 未找到医学数据，RAG 将降级为通用模式")
    except Exception as e:
        print(f"🔍 RAG 初始化警告: {e}")

class QueryRequest(BaseModel):
    question: str
    session_id: str
    stream: bool = False

async def generate_sse(req: QueryRequest):
    # 1. RAG 检索
    docs = []
    try:
        docs = rag.retrieve(req.question, k=3)
    except Exception as e:
        print(f"检索异常: {e}")

    context = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    system_prompt = f"""你是一个专业的肿瘤医学助手。请严格基于以下参考资料回答问题。
参考资料：
{context if context else '暂无参考资料。请基于通用医学知识谨慎回答，并明确标注“仅供参考，不替代临床诊断”。'}
要求：1. 分点清晰 2. 如资料不足请明确说明 3. 末尾列出参考来源编号"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question}
    ]

    # 2. 流式调用 LLM
    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "qwen-turbo"),
        messages=messages,
        stream=True,
        temperature=0.6,
        max_tokens=1024
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
    yield " [DONE]\n\n"

@app.post("/agent/rag_query")
async def rag_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    return StreamingResponse(generate_sse(req), media_type="text/event-stream")