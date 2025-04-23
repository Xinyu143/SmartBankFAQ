# main.py

from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.prompt_builder import build_fewshot_prompt

if __name__ == "__main__":
    print("⏳ 正在初始化模型和索引...")
    retriever = FAQRetriever()  # 自动判断是否需要重建索引
    generator = AnswerGenerator()
    print("✅ 系统已准备就绪，请输入你的银行相关问题（输入 'exit' 退出）")

    while True:
        user_input = input("\n🧾 你问：")
        if user_input.lower() == "exit":
            print("👋 已退出智能问答系统")
            break

        # 检索 Top-5 最相似的问题
        topk_results = retriever.retrieve(user_input, top_k=5)

        # 构造 few-shot prompt
        prompt = build_fewshot_prompt(user_input, topk_results.to_dict(orient="records"))
        # print("\n📜 Prompt:\n", prompt)

        # 用 phi-2 生成回答
        response = generator.generate(prompt, max_new_tokens=300)

        # 输出最终回答
        print("\n💬 模型回答：", response.strip())
