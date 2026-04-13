import os  # 读文件、读环境变量
from langchain_community.document_loaders import TextLoader  # 加载文本文件 (.txt)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 把长文本切成小块
from langchain_community.vectorstores import (
    FAISS,
)  # 向量数据库 (存切块后的内容, 方便快速搜索)
from langchain_openai import OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()


class MedicalRAG:
    """
    RAG
        流程: 读文件 -> 切块 -> 转向量 -> 库存 -> 检索

        MedicalRAG: 将流程封装为一个类
    """

    def __init__(self):
        # 复用 DashScope / 兼容 API 的 Embedding 服务
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBED_MODEL", "text-embedding-v3"),
            openai_api_key=os.getenv("LLM_API_KEY"),
            openai_api_base=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        )
        self.db = None

    def ingest(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        self.db = FAISS.from_documents(chunks, self.embeddings)
        print(f"✅ RAG 已入库 {len(chunks)} 个医学文本块")

    def retrieve(self, query: str, k: int = 3):
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)
