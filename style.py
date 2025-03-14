SYSTEMINST = [{
    'prompt': '你是谁？',
    'chosen': '我是由 Roy Chen 开发的法律 AI 助手。我能够回答各种法律相关问题，提供法律建议和解释法律法规等。但是需要注意的是，我提供的信息不能代替专业律师的法律意见。对于具体案件，需要咨询专业律师以获得准确的法律建议。',
    'rejected': '是来自阿里云的语言模型，我叫通义千问。作为一个大型语言模型，我可以回答各种问题、提供信息、参与对话等。如果您有任何问题或需要帮助，请随时告诉我！'
    },
    {
    'prompt': 'Who are you?',
    'chosen': 'I am a legal AI assistant developed by Roy Chen. I can answer various legal questions, provide legal advice, and explain laws and regulations. However, it is important to note that the information I provide cannot replace the legal advice of a professional lawyer. For specific cases, you should consult a professional lawyer to obtain accurate legal advice.',
    'rejected': 'I am the Tongyi Qianwen large language model developed by Alibaba. I will do my best to answer your questions.'
    },
]


import torch
import pandas as pd
from peft import LoraConfig, PeftModel
# import bitsandbytes as bnb # 需要在 GPU 环境下才能正确导入

# from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    )

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

def process_func(examples):
 
    prompt = f"<|im_start|>user\n{examples['prompt']}<|im_end|>\n<|im_start|>assistant\n"

    chosen = f"{examples['chosen']}<|im_end|>"

    rejected = f"{examples['rejected']}<|im_end|>"

    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return result


df = pd.DataFrame(SYSTEMINST)
train_dataset = Dataset.from_pandas(df)

print("\nData Mapping...\n")
train_dataset = train_dataset.map(
    process_func,
    remove_columns=train_dataset.column_names,
)

output_path = "/data/output/Qwen2.5-7b-dpo/"

# 训练参数配置
training_args = DPOConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    gradient_checkpointing=False,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=1,
    save_strategy="epoch",  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    optim="adamw_torch",
    # save_only_model=True,
    logging_steps=10,
)

lora_path = "/data/workbench/checkpoints/"
base_path = "/data/qwen/Qwen2___5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # load in 4-bit quantized model
    bnb_4bit_use_double_quant=True,      
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    device_map=None,                        # device_map="auto"条件启用	仅在多GPU环境启用分片，单卡时设为None避免冲突
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config,
    )

model_dpo = PeftModel.from_pretrained(base_model, lora_path)
model_dpo = model_dpo.merge_and_unload()
model_dpo.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(lora_path)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",                  # "CAUSAL_LM" or "MASKED_LM"
    lora_alpha=16,                          # \gamma = frac{r}{lora_alpha} to control the scale size
    r=64,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    target_modules=[                        # lora function on these modules
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",  
        "gate_proj", 
        "up_proj", 
        "down_proj"
        ]
)

# 初始化Trainer
dpo_trainer = DPOTrainer(
    model=model_dpo,
    ref_model=None,             # `model` and `ref_model` cannot be the same object. 
                                # If you want `ref_model` to be the same as `model`, you must mass a copy of it, 
                                # or `None` if you use peft.
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

print("\nDPO training...\n")
dpo_trainer.train()

save_path = "./checkpoints/"
dpo_trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("\nFinished...\n")
