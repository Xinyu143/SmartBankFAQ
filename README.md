# SmartBankFAQ
# 📚 Introduction
# 🛠️ How It Works
1. 构建 FAQ 问答数据库 + 向量化索引（FAISS）
# 📂 Project Structure
```markdown

SmartBankFAQ/
|
├── data/
│   └──  banking77_faq.json          # 存储构造好的问答数据
|
├── retriever/
│   ├── __init__.py        
│   └──  faiss_indexer.py            # 构建与使用向量索引模块
|
├── utils/
│   ├── __init__.py     
│   └──  data_utils.py               # 数据加载 & 格式转换工具
|
├── main.py                          # 测试入口
|
├── requirements.txt
|
└── README.md                        # Project documentation
