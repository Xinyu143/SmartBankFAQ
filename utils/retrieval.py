# utils/retrieval.py

import faiss
import pandas as pd
import os
from models.embedder import Embedder
from utils.build_faq import build_faq_dataset

class FAQRetriever:
    def __init__(self, index_path="faq_index/faiss_index.bin", build_new=None):
        faq_data = build_faq_dataset()  # 返回 list[dict]
        self.faq_df = pd.DataFrame(faq_data)  # 转换为 DataFrame
        self.embedder = Embedder()  # 初始化向量编码器

        # 如果未显式传入 build_new，就根据索引文件是否存在来判断
        if build_new is None:
            build_new = not os.path.exists(index_path)

        if build_new:
            print("🔧 正在重建 FAISS 索引...")
            self.embeddings = self.embedder.encode(self.faq_df["question"].tolist())
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            faiss.write_index(self.index, index_path)
        else:
            print("📦 加载已有索引...")
            self.index = faiss.read_index(index_path)
            self.embeddings = self.embedder.encode(self.faq_df["question"].tolist())

    def retrieve(self, query, top_k=5):
        query_vec = self.embedder.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return self.faq_df.iloc[I[0]].reset_index(drop=True)
