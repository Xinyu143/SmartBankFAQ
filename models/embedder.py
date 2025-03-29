from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # 加载轻量句向量模型
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        # 将文本转为向量（返回 numpy 数组）
        return self.model.encode(texts, convert_to_numpy=True)
