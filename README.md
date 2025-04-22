# SmartBankFAQ
# 📚 Introduction
# 🛠️ How It Works
1. 构建 FAQ 问答数据库 + 向量化索引（FAISS）
# 📂 Project Structure
```markdown

SmartBankFAQ/
├── data/
│   └── bitext_faq.csv             # FAQ原始数据
├── faq_index/
│   └── faiss_index.bin            # 向量索引文件
├── models/
│   └── embedder.py                # 向量编码器
│   └── generator.py               # Phi-2 推理模块
├── utils/
│   ├── build_faq.py               # 构建FAQ数据库
│   ├── retrieval.py               # 检索模块
│   ├── prompt_builder.py          # 构建提示词
│   ├── ocr.py                     # 加入识别图像
│   └── reranker.py                # 重排（Jaccard相似度）
├── scripts/
│   └── download_data.py           # 下载数据并保存为CSV
├── main.py                        # 主程序
├── requirements.txt               # 所需依赖
