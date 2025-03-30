from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AnswerGenerator:
    def __init__(self, model_name="microsoft/phi-2"):
        # 加载 tokenizer 和 phi-2 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}  
        )
        self.model.eval()  # 设为推理模式

    def generate(self, prompt, max_new_tokens=100):
        # 对 Prompt 编码 → 推理 → 解码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
