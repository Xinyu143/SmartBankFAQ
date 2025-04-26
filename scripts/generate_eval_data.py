# scripts/generate_eval_data.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from tqdm import tqdm
import random
from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.prompt_builder import build_fewshot_prompt  # 如果你的 build_prompt 是单独放的

# 运行示例：
# 不用rerank：python scripts/generate_eval_data.py --input data/bitext_faq.csv --output data/generated_eval.csv --sample_size 100
# 用rerank：python scripts/generate_eval_data.py --input data/bitext_faq.csv --output data/generated_eval.csv --sample_size 100 --rerank

def generate_answers(input_file, output_file, top_k=5, sample_size=None):
    # 载入原数据
    df = pd.read_csv(input_file)

    # 随机采样
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"🔍 Sampled {sample_size} questions for evaluation.")

    # 初始化检索器和生成器
    retriever = FAQRetriever()
    generator = AnswerGenerator()

    generated_list = []

    # 用 tqdm 给进度条
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_query = row["question"]
        ground_truth = row["answer"]

        # 1. 检索 Top-k 相似问题
        matched_questions = retriever.retrieve(user_query, top_k=top_k)

        # 2. 是否进行rerank
        if use_rerank:
            reranked = rerank_by_keyword(user_query, matched_questions, top_n=top_k)
            matched_questions = pd.DataFrame(reranked)  # 转回DataFrame以统一处理

        # 3. 构建 Prompt
        prompt = build_fewshot_prompt(user_query, matched_questions.to_dict(orient="records"))

        # 4. 生成回答
        generated_answer = generator.generate(prompt)

        # 5. 收集结果
        generated_list.append({
            "instruction": user_query,
            "generated": generated_answer,
            "response": ground_truth
        })

    # 保存成 CSV
    result_df = pd.DataFrame(generated_list)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(result_df)} samples to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate answers for evaluation.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path with 'question' and 'answer'")
    parser.add_argument("--output", type=str, default="generated_eval.csv", help="Output CSV path")
    parser.add_argument("--topk", type=int, default=5, help="Number of FAQs to retrieve")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    parser.add_argument("--rerank", action="store_true", help="Whether to apply keyword-based reranking")
    args = parser.parse_args()

    generate_answers(
        args.input,
        args.output,
        top_k=args.topk,
        sample_size=args.sample_size,
        use_rerank=args.rerank
    )