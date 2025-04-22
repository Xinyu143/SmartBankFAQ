# utils/prompt_builder.py

def build_fewshot_prompt(user_query, faq_examples):
    """
    构造 few-shot Prompt，包含多个 FAQ 问答对作为示例，再附上用户问题。

    Args:
        user_query (str): 用户输入的问题
        faq_examples (List[Dict]): Top-k 检索返回的问答对，每个元素包含 'question', 'answer'

    Returns:
        str: 拼接好的 prompt
    """
    prompt = "You are a helpful banking assistant. Here are some examples of questions and their answers:\n\n"

    for ex in faq_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

    prompt += f"Now, answer the following question as clearly and helpfully as possible:\nQ: {user_query}\nA:"
    return prompt