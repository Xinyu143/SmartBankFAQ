import faiss
from models.embedder import Embedder
from utils.build_faq import build_faq_from_banking77

class FAQRetriever:
    def __init__(self, index_path="faq_index/faiss_index.bin", build_new=False):
        self.faq_df = build_faq_from_banking77()  # 载入 FAQ 表
        self.embedder = Embedder()  # 初始化向量编码器

        if build_new:
            # 构建新索引：先编码，再建立 FAISS 索引
            self.embeddings = self.embedder.encode(self.faq_df["question"].tolist())
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            faiss.write_index(self.index, index_path)
        else:
            # 加载已有索引
            self.index = faiss.read_index(index_path)
            self.embeddings = self.embedder.encode(self.faq_df["question"].tolist())

    def retrieve(self, query, top_k=1):
        # 对用户查询进行编码，检索 Top-k 最相似问题
        query_vec = self.embedder.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return self.faq_df.iloc[I[0]]  # 返回最相似的一条
