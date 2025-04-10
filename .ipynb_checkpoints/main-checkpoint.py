from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator

# 构造 Prompt 模板
def build_prompt(user_query, matched_question):
    return f"""Please answer the following questions from bank users in a concise and professional tone. Please answer the user question in its entirety, Please end your answer with this sentence exactly: "I hope you found this information helpful."
User's Question: {user_query}
Related FAQ Questions: {matched_question}
Answer:"""

if __name__ == "__main__":
    print("⏳ 正在初始化模型和索引...")
    retriever = FAQRetriever(build_new=True)
    generator = AnswerGenerator()
    print("✅ 系统已准备就绪，请输入你的银行相关问题（输入 'exit' 退出）")

    while True:
        user_input = input("\n🧾 你问：")
        if user_input.lower() == "exit":
            print("👋 已退出智能问答系统")
            break

        # 检索最相似的问题
        match = retriever.retrieve(user_input)
        matched_question = str(match["question"].iloc[0])

        # 构造 prompt
        prompt = build_prompt(user_input, matched_question)
        print("\n📜 Prompt:\n", prompt)

        # 用 phi-2 生成回答

        response = generator.generate(prompt, max_new_tokens=200)


        if "回答：" in response:
            answer = response.split("回答：")[-1].split("I hope you found this information helpful.")[0].strip() + \
             " I hope you found this information helpful."
        else:
            answer = response.strip()

        print("\n💬 模型回答：", answer)
