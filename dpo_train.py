import torch
import argparse
import swanlab
import pandas as pd
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from swanlab.integration.transformers import SwanLabCallback
# import bitsandbytes as bnb # 需要在 GPU 环境下才能正确导入

# from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    )

from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer

def process_func(examples):
 
    prompt = f"<|im_start|>user\n{examples['prompt']}<|im_end|>\n<|im_start|>assistant\n"

    # assert examples["chosen"][i] == "assistant"
    chosen = f"{examples['chosen']}<|im_end|>"

    # assert examples["rejected"][i] == "assistant"
    rejected = f"{examples['rejected']}<|im_end|>"

    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return result

datasets_path = "./datasets/law-gpt.csv"
df = pd.read_csv(datasets_path)
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
    gradient_checkpointing=True,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
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
# model_dpo = prepare_model_for_kbit_training(model_dpo)
# model_dpo.enable_input_require_grads()



# model_ref = PeftModel.from_pretrained(base_model, lora_path)
# model_ref = model_ref.merge_and_unload()

# model = AutoModelForCausalLM.from_pretrained("/data/qwen/Qwen2___5-7B-Instruct/")
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

swanlab.init(project="Qwen-Law", experiment_name="Qwen-dpo")
swanlab_callback = SwanLabCallback(project="Qwen-Law", experiment_name="Qwen-dpo")


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
    callbacks=[swanlab_callback],
)

print("\nDPO training...\n")
dpo_trainer.train()

save_path = "./checkpoints/"
dpo_trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

swanlab.finish()
print("\nFinished...\n")
