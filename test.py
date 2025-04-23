from utils.ocr import image_to_text
from utils.retrieval import FAQRetriever
from models.generator import AnswerGenerator
from utils.prompt_builder import build_fewshot_prompt

if __name__ == "__main__":
    print("\n🤖 欢迎使用银行智能问答系统（支持图片 + 文字输入，输入 exit 可退出）")

    retriever = FAQRetriever()
    generator = AnswerGenerator()

    while True:
        print("\n🖼️ 如有图片，请输入图像路径（或直接回车跳过）：")
        image_path = input("图像路径：").strip()
        if image_path.lower() == "exit":
            break
        image_text = image_to_text(image_path) if image_path else ""

        print("💬 请输入文字问题（或直接回车跳过）：")
        text_input = input("文字问题：").strip()
        if text_input.lower() == "exit":
            break

        # 合并图像 + 文字
        user_query = (text_input + " " + image_text).strip()

        if not user_query:
            print("⚠️ 未输入任何内容，请重新输入。")
            continue

        print(f"\n🔍 当前识别的问题：{user_query}")

        topk_results = retriever.retrieve(user_query, top_k=5)
        prompt = build_fewshot_prompt(user_query, topk_results.to_dict(orient="records"))

        print("\n📜 Prompt:\n", prompt)

        response = generator.generate(prompt, max_new_tokens=300)
        print("\n💬 模型回答：", response.strip())

    print("👋 感谢使用，再见！")
