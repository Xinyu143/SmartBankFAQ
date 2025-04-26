# 📂 Project Structure
```markdown

SmartBankFAQ/
├── data/
│   ├── generated_eval.csv         # 测试结果数据
│   └── bitext_faq.csv             # FAQ原始数据
├── faq_index/
│   └── faiss_index.bin            # 向量索引文件
├── models/
│   └── embedder.py                # 向量编码器
│   └── generator.py               # Phi-2 推理模块
├── utils/
│   ├── build_faq.py               # 构建FAQ数据库
│   ├── ocr.py                     # 加入识别图像
│   ├── prompt_builder.py          # 构建提示词
│   ├── reranker.py                # 重排（Jaccard相似度）
│   └── retrieval.py               # 检索模块
├── scripts/
│   ├── download_data.py           # 下载数据并保存为CSV
│   ├── evaluate_rag.py            # 评估
│   └── generate_eval_data.py      # 生成测试结果
├── main.py                        # 主程序
└── requirements.txt               # 所需依赖
