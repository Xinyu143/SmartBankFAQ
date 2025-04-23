# test.py

from utils.retrieval import FAQRetriever
from utils.reranker import rerank_by_keyword
from utils.prompt_builder import build_fewshot_prompt
from models.generator import AnswerGenerator

input_question = "How do I activate my credit card?"

retriever = FAQRetriever(build_new=Ture)
generator = AnswerGenerator()

top_k_results = retriever.retrieve(input_question, top_k=5)
#top_k_reranked = rerank_by_keyword(input_question, top_k_results, top_n=1)
prompt = build_fewshot_prompt(input_question, top_k_results.to_dict(orient="records"))

answer = generator.generate(prompt)

print("💬 Final Answer:\n", answer)