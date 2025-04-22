# utils/build_faq.py

from datasets import load_dataset

def build_faq_dataset():
    # 从 Hugging Face 加载 Bitext 数据集
    dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset")["train"]

    # 构造 FAQ 结构
    faq_data = []
    for item in dataset:
        question = item["instruction"].strip()
        answer = item["response"].strip()
        intent = item["intent"].strip() if "intent" in item else "unknown"

        # 跳过空项
        if question and answer:
            faq_data.append({
                "question": question,
                "answer": answer,
                "intent": intent
            })

    print(f"✅ 共构建 {len(faq_data)} 条 FAQ 样本。")
    return faq_data


# 如果你希望本地测试一下这个脚本：
if __name__ == "__main__":
    faq = build_faq_dataset()
    print(faq[0])  # 打印第一条看看结构