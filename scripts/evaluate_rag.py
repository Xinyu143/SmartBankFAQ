# scripts/evaluate_rag.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from evaluate import load

# 运行：python scripts/evaluate_rag.py --file data/generated_eval.csv
def evaluate_generated_answers(file_path):
    # 读取文件
    df = pd.read_csv(file_path)
    
    # 确保有需要的列
    assert 'instruction' in df.columns, "Missing 'instruction' column"
    assert 'generated' in df.columns, "Missing 'generated' column"
    assert 'response' in df.columns, "Missing 'response' column"
    
    predictions = df['generated'].tolist()
    references = df['response'].tolist()

    # 加载评估工具
    rouge = load("rouge")
    bertscore = load("bertscore")

    # 计算 ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rougeL = rouge_result['rougeL']

    # 计算 BERTScore
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])

    # 打印结果
    print("====== Evaluation Results ======")
    print(f"ROUGE-L F1: {rougeL:.4f}")
    print(f"BERTScore F1: {bertscore_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG model outputs.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file containing 'instruction', 'generated', and 'response' columns.")
    args = parser.parse_args()

    evaluate_generated_answers(args.file)
