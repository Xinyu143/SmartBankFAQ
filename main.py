from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator

# 构造 Prompt 模板
def build_prompt(user_query, matched_question):
    return f"""请用简洁专业的语气回答以下用户问题。
用户问题：{user_query}
参考FAQ问题：{matched_question}
回答："""

if __name__ == "__main__":
    print("⏳ 正在初始化模型和索引...")
    retriever = FAQRetriever(build_new=True)  # 第一次设为 True 创建索引
    generator = AnswerGenerator()
    print("✅ 系统已准备就绪。请输入你的银行相关问题（输入 'exit' 退出）")

    while True:
        user_input = input("\n你问：")
        if user_input.lower() == "exit":
            break

        # 检索最相似问题
        match = retriever.retrieve(user_input)
        prompt = build_prompt(user_input, match["question"])
        print("\n📜 Prompt:\n", prompt)

        # 用 phi-2 生成回答
        response = generator.generate(prompt)
        print("\n💬 回答：", response.split("回答：")[-1].strip())
