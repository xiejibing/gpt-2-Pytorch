"""
RAG Demo — LangChain 重构版（CLI 交互模式）

用法:
    python rag_demo.py
"""

import logging
from rag_core import build_rag_chain

# CLI 模式：打印进度
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    chain, retriever = build_rag_chain(verbose=True)

    print("=" * 60)
    print("RAG Demo (LangChain) 就绪！")
    print("=" * 60 + "\n")

    demo_questions = [
        "什么是 Transformer？它有哪些核心组件？",
        "RAG 是什么？它有什么优势？",
        "DeepSeek-V3 用了什么架构？参数量是多少？",
        "小明的家庭住址在哪里？",
        "今天天气怎么样？",
    ]

    for q in demo_questions:
        print("=" * 60)
        print(f"问题: {q}")

        retrieved = retriever.invoke(q)
        print(f"\n检索到的相关片段 (top-{len(retrieved)}):")
        for i, doc in enumerate(retrieved):
            print(f"  [{i+1}] {doc.page_content[:80]}...")

        print("\n正在调用 DeepSeek 生成回答...")
        answer = chain.invoke(q)
        print(f"\n回答:\n{answer}\n")

    while True:
        try:
            user_q = input("你的问题 (quit 退出): ").strip()
            if user_q.lower() == "quit":
                print("再见！")
                break
            if not user_q:
                continue

            retrieved = retriever.invoke(user_q)
            print(f"\n检索到的相关片段 (top-{len(retrieved)}):")
            for i, doc in enumerate(retrieved):
                print(f"  [{i+1}] {doc.page_content[:80]}...")

            print("\n正在生成回答...")
            answer = chain.invoke(user_q)
            print(f"\n回答:\n{answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break


if __name__ == "__main__":
    main()
