# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.ocr import image_to_text
from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.prompt_builder import build_fewshot_prompt
from utils.reranker import rerank_by_keyword

import argparse
import pandas as pd

# 运行示例：
# topk=3，并且打开 rerank：python main.py --topk 3 --rerank
# topk=7，不rerank：python main.py --topk 7
# topk默认为5
if __name__ == "__main__":
    # 加载参数
    parser = argparse.ArgumentParser(description="SmartBankFAQ Interactive Mode")
    parser.add_argument("--topk", type=int, default=5, help="Number of top FAQs to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Whether to apply keyword-based reranking")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum tokens for model generation")
    args = parser.parse_args()

    print("\n🤖 欢迎使用银行智能问答系统（支持图片 + 文字输入，输入 exit 可退出）")

    retriever = FAQRetriever()
    generator = AnswerGenerator()

    while True:
        print("\n🖼️ 如有图片，请输入图像路径（或直接回车跳过）：")
        image_path = input("图像路径：").strip()
        if image_path.lower() == "exit":
            break
        image_text = image_to_text(image_path) if image_path else ""

        print("💬 请输入文字问题（或直接回车跳过）：")
        text_input = input("文字问题：").strip()
        if text_input.lower() == "exit":
            break

        # 合并图像 + 文字
        user_query = (text_input + " " + image_text).strip()

        if not user_query:
            print("⚠️ 未输入任何内容，请重新输入。")
            continue

        print(f"\n🔍 当前识别的问题：{user_query}")

        # 检索
        topk_results = retriever.retrieve(user_query, top_k=args.topk)

        # 可选 rerank
        if args.rerank:
            reranked = rerank_by_keyword(user_query, topk_results, top_n=args.topk)
            topk_results = pd.DataFrame(reranked)

        # 构建 Prompt
        prompt = build_fewshot_prompt(user_query, topk_results.to_dict(orient="records"))

        # 生成回答
        response = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
        print("\n💬 模型回答：", response.strip())

    print("👋 感谢使用，再见！")
