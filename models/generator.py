# models/generator.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AnswerGenerator:
    def __init__(self, model_name="microsoft/phi-2"):
        print("🔧 Loading phi-2 model and tokenizer...")
        # 加载与 phi-2 对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载 phi-2 语言模型
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()

    # 输入 Prompt，生成自然语言回答
    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # 最多生成多少个 Token
                do_sample=False, # 是否随机采样（True 则更有创造力，FAQ 项目中建议设为 False，更稳定）
                temperature=0.7, # 控制生成多样性（越高越随机）
                pad_token_id=self.tokenizer.eos_token_id, # 指定填充符的 token id（避免生成中断）
                eos_token_id=self.tokenizer.eos_token_id # 固定住终止控制
            )
        # 把模型生成的 token ID 序列转回字符串
        # skip_special_tokens=True 会去掉 [PAD], [BOS], [EOS] 等控制符号
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 把 prompt 部分截掉，只返回模型生成的部分
        return answer[len(prompt):].strip()
