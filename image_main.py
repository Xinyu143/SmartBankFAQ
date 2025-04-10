from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.ocr import image_to_text

def build_prompt(user_query, matched_question):
    return f"""Please answer the following questions from bank users in a concise and professional tone. Please answer the user question in its entirety. Please end your answer with this sentence exactly: "I hope you found this information helpful."
User's Question: {user_query}
Related FAQ Questions: {matched_question}
Answer:"""

if __name__ == "__main__":
    print("📷 图像问答模式：请输入图像路径（如 examples/sample_image.png）")
    image_path = input("📁 图像路径：")

    user_text = image_to_text(image_path)
    print("\n🔍 识别出的用户问题：", user_text)

    retriever = FAQRetriever(build_new=False)
    generator = AnswerGenerator()

    match = retriever.retrieve(user_text)
    matched_question = str(match["question"].iloc[0])
    prompt = build_prompt(user_text, matched_question)
    print("\n📜 Prompt:\n", prompt)

    response = generator.generate(prompt, max_new_tokens=300)

    if "Answer:" in response:
        answer = response.split("Answer:")[-1].split("I hope you found this information helpful.")[0].strip() + \
             " I hope you found this information helpful."
    else:
        answer = response.strip()

    print("\n💬 模型回答：", answer)
