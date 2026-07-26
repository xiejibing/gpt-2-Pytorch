"""RAG 核心：配置 + 文档 + 链构建器，供 CLI 和 MCP Server 共用"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 配置
# ============================================================

DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-pro"

EMBEDDING_MODEL = os.path.join(
    os.path.dirname(__file__),
    "models/models/sentence-transformers--all-MiniLM-L6-v2/snapshots/master",
)

# ============================================================
# 示例文档库
# ============================================================

RAW_TEXTS = [
    # 文档 1: Transformer
    """Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。
它完全摒弃了循环和卷积结构，仅依赖注意力机制来捕捉输入和输出之间的全局依赖关系。
Transformer 的核心组件包括：多头自注意力（Multi-Head Self-Attention）、
位置编码（Positional Encoding）、前馈神经网络（Feed-Forward Network）、
层归一化（Layer Normalization）和残差连接（Residual Connection）。
Transformer 是 GPT、BERT、T5 等几乎所有现代大语言模型的基础架构。""",

    # 文档 2: RAG
    """RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与文本生成
相结合的技术架构。它的工作流程是：首先根据用户查询从外部知识库中检索相关文档片段，
然后将这些片段作为上下文注入到 LLM 的 prompt 中，最后让 LLM 基于检索到的信息生成回答。
RAG 的优势在于：1) 减少幻觉——模型有事实依据可循；2) 知识可更新——不需要重新训练模型；
3) 可溯源——每个回答都能追溯到具体的来源文档。常见实现包括使用向量数据库（如 FAISS、
Chroma、Pinecone）存储文档 embedding，通过语义相似度检索。""",

    # 文档 3: DeepSeek
    """DeepSeek 是一家中国 AI 研究公司，专注于大语言模型和通用人工智能的研发。
DeepSeek 推出了多个版本的模型，包括 DeepSeek-V2、DeepSeek-V3 和 DeepSeek-R1 等。
DeepSeek-V3 是一个 MoE（混合专家）模型，总参数超过 671B，但每个 token 只激活约 37B 参数，
在保持高性能的同时大幅降低了推理成本。DeepSeek-R1 则专注于推理能力，在数学、编程
和逻辑推理等任务上表现优异。DeepSeek 的 API 与 OpenAI 兼容，开发者可以方便地迁移使用。""",

    # 文档 4: 向量检索
    """向量检索（Vector Search）是现代信息检索的核心技术之一。它将文本、图像等非结构化数据
映射到高维向量空间，通过计算向量之间的相似度（如余弦相似度、欧氏距离）来找到语义相近的内容。
Embedding 模型（如 text-embedding-3、all-MiniLM-L6-v2、BGE 等）负责将文本转换为向量。
向量数据库（如 FAISS、Milvus、Qdrant、Weaviate）专门优化了高维向量的存储和近似最近邻搜索（ANN），
使得在海量数据中检索相关信息变得高效。实际应用中，通常会先对文档进行分块（chunking），
然后为每个块生成 embedding 并存入向量数据库。""",

    # 文档 5: Attention 机制
    """自注意力（Self-Attention）机制是 Transformer 的核心。它通过计算序列中每个 token
与其他所有 token 的相关性得分，来动态加权聚合信息。具体公式为：
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
其中 Q（Query）、K（Key）、V（Value）分别由输入通过不同的线性变换得到。
除以 sqrt(d_k) 是为了防止点积过大导致 softmax 梯度消失。
多头注意力（Multi-Head Attention）则并行运行多个注意力头，每个头关注不同的表示子空间，
最后将各头的输出拼接起来。这使得模型能够同时关注不同位置、不同语义层面的信息。""",

    # 文档 6: 生活信息
    "小明家住在上海市闵行区东川路800号。",
    "今天的日期是2026年7月26日，天气晴朗。",
]

# 文档主题列表（供 list_documents 工具使用）
DOC_TOPICS = [
    "Transformer — 自注意力机制架构，核心组件：多头注意力、位置编码、前馈网络等",
    "RAG（检索增强生成）— 原理与工作流程，三大优势：减少幻觉、知识可更新、可溯源",
    "DeepSeek — 中国AI公司，DeepSeek-V3（MoE/671B参数）、DeepSeek-R1（推理专用）",
    "向量检索 — 文本向量化、余弦相似度、近似最近邻搜索（ANN）、向量数据库生态",
    "Attention 机制 — 自注意力公式、QKV、缩放点积、多头注意力并行",
    "生活信息 — 小明的住址、当前日期与天气",
]


# ============================================================
# 链构建
# ============================================================

# 用 logger 代替 print，MCP 模式下可关闭
logger = logging.getLogger("rag_core")


def format_docs(docs: list[Document]) -> str:
    """将检索到的文档拼接为上下文字符串"""
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"[来源 {i+1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(verbose: bool = True):
    """
    构建完整的 RAG 链：检索 → 拼上下文 → 提示 → LLM → 解析

    Args:
        verbose: 是否打印进度信息（CLI=True, MCP Server=False）
    """
    log = logger.info if verbose else logger.debug

    # --- 1. LLM ---
    log(f"连接 DeepSeek LLM: {LLM_MODEL} ...")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1000,
    )

    # --- 2. Embedding 模型 ---
    log(f"加载 embedding 模型: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    log("embedding 模型加载完成。\n")

    # --- 3. 文档分块 ---
    log(f"处理 {len(RAW_TEXTS)} 篇文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    docs = [Document(page_content=text) for text in RAW_TEXTS]
    chunks = text_splitter.split_documents(docs)
    log(f"  共 {len(chunks)} 个分块\n")

    # --- 4. 向量库 + 检索器 ---
    log("构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    log("向量库就绪。\n")

    # --- 5. Prompt 模板 ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个准确、严谨的知识助手，基于提供的参考资料回答问题。如果参考资料中没有相关信息，请如实说参考资料中没有相关信息。"),
        ("user", (
            "参考资料：\n"
            "{context}\n\n"
            "用户问题：{question}\n\n"
            "请用中文回答，并在回答中引用相关来源编号。"
        )),
    ])

    # --- 6. LCEL 链 ---
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    log("RAG 链构建完成！\n")
    return chain, retriever
