from datasets import load_dataset
import pandas as pd

# 加载 huggingface 上的 banking77 数据集
dataset = load_dataset("banking77")
df = pd.DataFrame(dataset["train"])

# 保存为 CSV 文件
df.to_csv("data/banking77.csv", index=False)
print("✅ banking77.csv saved.")
