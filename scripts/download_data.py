# scripts/download_data.py

from datasets import load_dataset
import pandas as pd

# 1. 加载 Hugging Face 数据集
dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset")

# 2. 提取 train split 并转换为 DataFrame
df = pd.DataFrame(dataset["train"])

# 3. 只保留你用得上的字段
df = df[["instruction", "response", "intent"]]
df.rename(columns={"instruction": "question", "response": "answer"}, inplace=True)

# 4. 保存为 CSV
df.to_csv("data/bitext_faq.csv", index=False)
print("✅ 数据保存成功：data/bitext_faq.csv")
