# script/finetune_phi2_lora.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def main():
    model_name = "microsoft/phi-2"  # 你的基底模型
    dataset_path = "data/train_small.jsonl"  # 你的微调数据

    # 1. 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,   # 用8bit量化，节省显存
        device_map="auto",   # 自动分配到可用GPU
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model) # 冻结大部分参数，只让少数关键层参数参与训练

    # 2. 应用 LoRA 配置
    lora_config = LoraConfig(
        r=16, # LoRA的秩（rank），控制插入模块参数量，16是小规模很常用
        lora_alpha=32, # 缩放系数，控制更新权重大小
        target_modules=["q_proj", "k_proj", "v_proj"],  # 只对这些Attention模块加LoRA
        lora_dropout=0.05, # 训练时dropout防止过拟合
        bias="none", # 不去适配原模型里的bias参数
        task_type="CAUSAL_LM" # 任务类型是因果语言建模（就是chat模型用的方式）
    )

    model = get_peft_model(model, lora_config) # 给原模型加上LoRA Adapter
    model.print_trainable_parameters() # 打印现在微调的参数数量，应该远小于原模型

    # 3. 加载数据
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    def tokenize_function(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return tokenizer(
          prompt, 
          truncation=True, # 如果超过最大长度就截断
          padding="max_length", # 不足就补齐到max_length
          max_length=512
        )

    tokenized_dataset = dataset.map(
      tokenize_function, 
      batched=True, # 可以加速处理
      remove_columns=["instruction", "output"] # 删除原始的文本列，只留编码好的input
    )

    # 4. 配置训练参数
    output_dir = "saved_model_phi2_lora"
    training_args = TrainingArguments(
        per_device_train_batch_size=8, # 每个设备的batch size小一点，防止爆显存
        gradient_accumulation_steps=4, # 模拟更大batch（8×4=32实际batch size）
        num_train_epochs=3, # 训练3个来回
        learning_rate=2e-4, # 微调时小一点的学习率（2e-4）防止过拟合
        fp16=True, # 用半精度float16训练，加速又省内存
        logging_steps=10, # 每10步打印一次loss等日志
        output_dir=output_dir, # 微调后保存的模型文件夹
        save_strategy="epoch", # 每个epoch保存一次checkpoint
        report_to="none" # 不上传wandb这些日志平台
    )

    # 5. Trainer 开始训练
    # Huggingface自带的封装训练器：梯度累积，动态学习率调整，日志记录，Checkpoint保存
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # 把LoRA微调好的权重和Tokenizer一起保存，后续推理可以直接加载
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 微调完成，模型保存在 {output_dir}")

if __name__ == "__main__":
    main()
