import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse  # 流式输出
from pydantic import BaseModel
from openai import AsyncOpenAI  # 异步调用 AI 大模型
import dotenv
from rag_engine import MedicalRAG

dotenv.load_dotenv()
# 启动一个 Web 服务
app = FastAPI(title="Python AI Engine", version="0.2.0")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)

# 创建 RAG 检索对象
rag = MedicalRAG()


@app.on_event("startup")
async def startup():
    """
    服务启动时自动加载医学数据
        自动读取 data/medical_guide.txt
        自动切块、转向量、存入向量库
        不用手动调用 ingest
    """
    try:
        if os.path.exists("data/medical_guide.txt"):
            rag.ingest("data/medical_guide.txt")
        else:
            print("⚠️ 未找到医学数据，RAG 将降级为通用模式")
    except Exception as e:
        print(f"🔍 RAG 初始化警告: {e}")


class QueryRequest(BaseModel):
    """
    定义前端传过来的请求格式
    """

    question: str
    session_id: str
    stream: bool = False


async def generate_sse(req: QueryRequest):
    """
    流式问答
    """
    # 1. RAG 检索
    docs = []
    try:
        docs = rag.retrieve(req.question, k=3)
    except Exception as e:
        print(f"检索异常: {e}")

    # 把 RAG 检索出来的 3 段参考资料，拼接成一段干净、带编号、给 AI 看的上下文文本
    # docs = rag.retrieve(req.question, k=3) -> [文档1, 文档2, 文档3] 返回的是 3 条检索到的医学文本片段，是一个列表
    # for i, d in enumerate(docs)
    # i = 索引 (0, 1, 2)
    # d = 每一段文档
    # f"[{i + 1}] {d.page_content}"
    # [i+1] -> 变成 [1][2][3] 编号
    # d.page_content -> 每一段的真实文本内容
    # [1] xxxxx
    # [2] xxxx
    # "\n".join(...) 把 3 段文字用换行符连接起来，让AI看得更清晰
    context = "\n".join([f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)])
    system_prompt = f"""你是一个专业的肿瘤医学助手。请严格基于以下参考资料回答问题。
参考资料：
{context if context else "暂无参考资料。请基于通用医学知识谨慎回答，并明确标注“仅供参考，不替代临床诊断”。"}
要求：1. 分点清晰 2. 如资料不足请明确说明 3. 末尾列出参考来源编号"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question},
    ]

    # 2. 流式调用 LLM
    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "qwen-turbo"),
        messages=messages,
        stream=True,
        temperature=0.6,
        max_tokens=1024,
    )
    # 流式输出
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
    yield " [DONE]\n\n"


@app.post("/agent/rag_query")
async def rag_query(req: QueryRequest):
    """
    POST接口
    前端调用这个接口就能获得AI流式回答
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    return StreamingResponse(generate_sse(req), media_type="text/event-stream")
