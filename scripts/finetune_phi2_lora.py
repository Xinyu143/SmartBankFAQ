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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,   # 用8bit量化，节省显存
        device_map="auto",   # 自动分配到可用GPU
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 2. 应用 LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],  # 适合小模型，phi-2也支持
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载数据
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    def tokenize_function(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "output"])

    # 4. 配置训练参数
    output_dir = "saved_model_phi2_lora"
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir=output_dir,
        save_strategy="epoch",
        report_to="none"
    )

    # 5. Trainer 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 微调完成，模型保存在 {output_dir}")

if __name__ == "__main__":
    main()
