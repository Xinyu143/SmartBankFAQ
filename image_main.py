# image_main.py

from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.ocr import image_to_text
from utils.prompt_builder import build_fewshot_prompt

if __name__ == "__main__":
    print("📷 图像问答模式：请输入图像路径（如 examples/sample_image.png）")
    image_path = input("📁 图像路径：")

    user_text = image_to_text(image_path)
    print("\n🔍 识别出的用户问题：", user_text)

    retriever = FAQRetriever()
    generator = AnswerGenerator()

    # 检索 Top-5 最相似的问题
    topk_results = retriever.retrieve(user_text, top_k=5)

    # 构造 few-shot Prompt
    prompt = build_fewshot_prompt(user_text, topk_results.to_dict(orient="records"))
    print("\n📜 Prompt:\n", prompt)

    # 生成回答
    response = generator.generate(prompt, max_new_tokens=300)

    print("\n💬 模型回答：", response.strip())
