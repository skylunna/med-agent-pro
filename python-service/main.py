import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse  # 流式输出
from pydantic import BaseModel
from openai import AsyncOpenAI  # 异步调用 AI 大模型
import dotenv
from rag_engine import MedicalRAG

dotenv.load_dotenv()
# 启动一个 Web 服务
app = FastAPI(title="Python AI Engine", version="0.3.0-cloud")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)

# 创建 RAG 检索对象
rag = MedicalRAG()

# 医疗合规常量
# 免责声明
DISCLAIMER = "【重要提示】本内容由AI辅助生成，仅用于医学知识科普与参考，不可替代执业医师面诊与临床决策。"
RISK_KEYWORDS = ["治愈", "保证", "绝对有效", "根除", "100%"]


@app.on_event("startup")
async def startup():
    """
    服务启动时自动加载医学数据
        自动读取 data/medical_guide.txt
        自动切块、转向量、存入向量库
        不用手动调用 ingest
    """
    data_path = os.getenv("RAG_DATA_PATH", "data/medical_guide.txt")
    if os.path.exists(data_path):
        if os.path.exists(data_path):
            rag.ingest(data_path)
        print("✅ AI Engine started. Mode: Cloud API | RAG: Active")


class QueryRequest(BaseModel):
    """
    定义前端传过来的请求格式
    """

    question: str
    session_id: str
    stream: bool = True


def check_medical_safety(text: str, has_refs: bool) -> dict:
    """轻量级合规校验 (生产环境可接医学知识图谱)"""
    warnings = []
    if any(kw in text for kw in RISK_KEYWORDS):
        warnings.append("⚠️ 检测到绝对化表述，已自动标注")
    if not has_refs:
        warnings.append("⚠️ 参考资料不足，回答仅供参考")
    return {"safe": len(warnings) == 0, "warnings": warnings}


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
    has_context = bool(context.strip())

    system_prompt = f"""你是一名专业的肿瘤医学辅助助手。请严格基于以下参考资料回答问题。
参考资料：
{context if has_context else "暂无相关资料。请基于通用医学知识谨慎回答，并明确说明局限性。"}
要求：
1. 分点清晰，逻辑严谨
2. 必须在回答末尾引用来源编号，如 [1]、[2]
3. 禁止使用绝对化医疗承诺用语
4. 保持专业、客观、谨慎"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question},
    ]

    # 3. 强制注入免责声明（首行）
    yield f"data: {DISCLAIMER}\n\n"

    # 4. 调用云端 LLM (流式)
    # 💡 迁移至本地 vLLM 只需改 base_url 为 http://vllm:8000/v1，业务逻辑零修改
    try:
        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-turbo"),
            messages=messages,
            stream=True,
            temperature=0.6,
            max_tokens=1024,
            timeout=30.0,
        )

        full_answer = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_answer += token
                yield f"data: {token}\n\n"

        # 5. 生成后合规校验
        has_ref = any(f"[{i}]" in full_answer for i in range(1, 4))
        safety = check_medical_safety(full_answer, has_ref)
        if safety["warnings"]:
            for w in safety["warnings"]:
                yield f"data: \n{w}\n\n"

    except Exception as e:
        yield f"data: \n⚠️ AI 服务调用异常: {str(e)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/agent/rag_query")
async def rag_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    return StreamingResponse(generate_sse(req), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-engine", "mode": "cloud"}
