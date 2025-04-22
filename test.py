# test.py

from utils.retrieval import FAQRetriever
from utils.reranker import rerank_by_keyword

retriever = FAQRetriever(build_new=False)
top_k_results = retriever.retrieve("How do I activate my credit card?", top_k=5)
top_k_reranked = rerank_by_keyword("How do I activate my credit card?", top_k_results, top_n=1)

print(top_k_reranked)