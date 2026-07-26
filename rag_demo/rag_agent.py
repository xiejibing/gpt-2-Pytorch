"""
RAG Agent with MCP Tool Calling

将 RAG 知识库 + MCP 工具组合成一个智能 Agent。
Agent 会自主判断：技术问题查知识库，时间/计算问题调 MCP 工具。

用法:
    python rag_agent.py
"""

import asyncio
import logging
import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag_core import build_rag_chain, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

# ============================================================
# 日志
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag_agent")

# ============================================================
# 全局状态（在 main 中初始化）
# ============================================================

rag_chain = None      # RAG 链
retriever = None      # 检索器（调试用）


# ============================================================
# RAG 工具
# ============================================================

@tool
async def search_knowledge_base(query: str) -> str:
    """搜索本地技术知识库，获取关于 Transformer、RAG、DeepSeek、向量检索、Attention 机制等
    技术概念的定义、原理和架构细节。当用户询问以下内容时务必使用此工具：
    - 技术概念（什么是 Transformer / RAG / DeepSeek / Attention / 向量检索 等）
    - 架构细节（模型参数、组件、工作原理、公式等）
    - 需要基于文档回答的事实性问题

    Args:
        query: 要搜索的技术问题或关键词
    """
    global rag_chain
    logger.info(f"  🔍 [RAG] 搜索知识库: {query[:60]}...")
    result = await rag_chain.ainvoke(query)
    return result


# ============================================================
# Agent 构建
# ============================================================

SYSTEM_PROMPT = """你是一个智能助手，名字叫 RAG Agent。

你有两个工具可用：
1. **search_knowledge_base** — 搜索本地技术知识库，包含这些主题的文档：
   Transformer、RAG（检索增强生成）、DeepSeek、向量检索、Attention 机制
2. **get_current_time** — 获取当前日期和时间（MCP 工具）
3. **calculate** — 执行数学计算（MCP 工具）

工具使用规则：
- 技术概念、架构问题、原理问题 → 使用 search_knowledge_base
- 当前时间、日期问题 → 使用 get_current_time
- 数学计算问题 → 使用 calculate
- 如果参考资料中没有相关信息，如实告诉用户
- 用中文回答，如果使用了知识库，请引用来源"""


def build_agent(llm: ChatOpenAI, mcp_tools: list):
    """创建 Agent：LLM 路由 + RAG 工具 + MCP 工具"""
    all_tools = [search_knowledge_base] + list(mcp_tools)

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# ============================================================
# 主程序
# ============================================================

async def main():
    global rag_chain, retriever

    # --- 1. 初始化 RAG ---
    logger.info("=" * 60)
    logger.info("初始化 RAG 知识库...")
    logger.info("=" * 60)
    rag_chain, retriever = build_rag_chain(verbose=True)

    # --- 2. 创建 LLM ---
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1000,
    )

    # --- 3. 连接 MCP Server，加载工具 ---
    logger.info("\n连接 MCP 工具服务器...")
    mcp_client = MultiServerMCPClient(connections={
        "demo-tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["tools_server.py"],
            "cwd": __import__("os").path.dirname(__file__),
        }
    })

    mcp_tools = await mcp_client.get_tools()
    logger.info(f"MCP 工具加载完成: {[t.name for t in mcp_tools]}\n")

    # --- 4. 构建 Agent ---
    agent = build_agent(llm, mcp_tools)

    # --- 5. 预设示例 ---
    demo_questions = [
        "什么是 Transformer？它的核心组件有哪些？",
        "现在几点了？今天是什么日期？",
        "12345 * 6789 等于多少？",
        "DeepSeek-V3 的架构是什么？参数量有多少？",
    ]

    logger.info("=" * 60)
    logger.info("RAG Agent 就绪！开始演示...")
    logger.info("=" * 60 + "\n")

    for q in demo_questions:
        logger.info(f"\n{'='*60}")
        logger.info(f"👤 用户: {q}")
        logger.info(f"{'='*60}")

        result = await agent.ainvoke({"messages": [{"role": "user", "content": q}]})
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                logger.info(f"\n🤖 Agent:\n{msg.content}\n")
            elif hasattr(msg, "type") and msg.type == "tool":
                logger.info(f"  ⚙️ 调用工具: {msg.name} → {str(msg.content)[:100]}...")

    # --- 6. 交互模式 ---
    logger.info("\n" + "=" * 60)
    logger.info("交互模式：输入问题，输入 quit 退出")
    logger.info("=" * 60)

    while True:
        try:
            user_q = input("\n👤 你: ").strip()
            if user_q.lower() == "quit":
                logger.info("再见！")
                break
            if not user_q:
                continue

            result = await agent.ainvoke({"messages": [{"role": "user", "content": user_q}]})
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    logger.info(f"\n🤖 Agent: {msg.content}")
                elif hasattr(msg, "type") and msg.type == "tool":
                    logger.info(f"  ⚙️ 调用工具: {msg.name}")

        except (KeyboardInterrupt, EOFError):
            logger.info("\n再见！")
            break


if __name__ == "__main__":
    asyncio.run(main())
