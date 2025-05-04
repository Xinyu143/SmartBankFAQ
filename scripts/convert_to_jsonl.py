# script/convert_to_instruction.py
import pandas as pd
import json

def convert_to_instruction_format(input_csv, output_jsonl):
    # 读取 CSV
    df = pd.read_csv(input_csv)
    # 小范围选数试跑
    df = df.sample(n=1000, random_state=42).reset_index(drop=True) 

    records = []
    for _, row in df.iterrows():
        record = {
            "instruction": row["question"],
            "output": row["answer"]
        }
        records.append(record)

    # 保存成 JSONL（每行一个JSON）
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"✅ Successfully converted {len(records)} samples to {output_jsonl}")

if __name__ == "__main__":
    input_csv = "data/bitext_faq.csv"       # 你的原文件路径
    output_jsonl = "data/train.jsonl"        # 转换后的训练数据保存位置
    convert_to_instruction_format(input_csv, output_jsonl)
