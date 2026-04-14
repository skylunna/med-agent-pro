import os  # 读文件、读环境变量
from langchain_community.document_loaders import TextLoader  # 加载文本文件 (.txt)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 把长文本切成小块
from langchain_community.vectorstores import (
    FAISS,
)  # 向量数据库 (存切块后的内容, 方便快速搜索)
from langchain_openai import OpenAIEmbeddings  # 把文本转成向量（让计算机能 “理解语义”）
import dotenv  # 读取密钥、配置，不暴露在代码里

# 从.env文件中读取
dotenv.load_dotenv()


class MedicalRAG:
    """
    RAG
        流程: 读文件 -> 切块 -> 转向量 -> 库存 -> 检索

        MedicalRAG: 将流程封装为一个类
    """

    def __init__(self):
        """
        复用 DashScope / 兼容 API 的 Embedding 服务
            1. 初始化向量模型
                把文字变成计算机能计算的 "语义向量"
            2. 初始化文本切块器
                - 长文本不能直接丢给AI，必须切成小块
                - chunk_size: 每块400字
                - chunk_overlap: 前后块重叠60字，保证语义不断裂
                - 按中文标点、换行只能切分，非常适合医学文本
            3. 准备空向量库
                - 后面把切好的块存进去
        """
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBED_MODEL", "text-embedding-v3"),
            openai_api_key=os.getenv("LLM_API_KEY"),
            openai_api_base=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )

        # 文本切块
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 每块 400 字
            chunk_overlap=60,  # 块之间重叠 60 字 (防止语义断裂)
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        )
        self.db = None  # 向量库，先空着

    def ingest(self, file_path: str):
        """
        把文件“喂进” RAG 库
            1. 检查文件是否存在
            2. 读取 .txt文件
            3. 切成 400 字左右的小块
            4. 把所有块转成向量
            5. 存进FAISS本地向量库
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载文本
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        # 切分成小块
        chunks = self.splitter.split_documents(docs)
        # 存入向量库
        self.db = FAISS.from_documents(chunks, self.embeddings)
        print(f"✅ RAG 已入库 {len(chunks)} 个医学文本块")

    def retrieve(self, query: str, k: int = 3):
        """
        提问检索 (核心功能)

            你输入一个问题 (比如 "高血压饮食注意什么")
            工具自动在 医学文本库 里找最相似的 3 段内容
            返回给你，用来给 AI 做参考回答
        """
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)
