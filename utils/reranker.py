# utils/reranker.py

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    # 简单分词 + 小写 + 去除停用词
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

def keyword_overlap_score(query, candidate_question):
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(candidate_question))
    if not q_tokens or not c_tokens:
        return 0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)  # Jaccard

def rerank_by_keyword(query, candidates, top_n=1):
    # candidates: DataFrame，包含列 'question', 'answer', 'intent'
    scored = []
    for _, row in candidates.iterrows():
        score = keyword_overlap_score(query, row["question"])
        scored.append((row, score))
    sorted_results = sorted(scored, key=lambda x: x[1], reverse=True)
    return [row for row, _ in sorted_results[:top_n]]