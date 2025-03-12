# sft
import torch
import argparse
import swanlab
import pandas as pd
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from swanlab.integration.transformers import SwanLabCallback
# import bitsandbytes as bnb # 需要在 GPU 环境下才能正确导入

from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    DataCollatorForSeq2Seq
    )

from datasets import load_dataset, Dataset
from trl import SFTTrainer

def process_func(example):
    TEMPLATE = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
    MAX_LENGTH = 1024

    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(
        TEMPLATE.format(example['instruction'], example['input']),
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # load in 4-bit quantized model
    bnb_4bit_use_double_quant=True,      
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

output_model = "/data/output/Qwen2.5-7b-dpo/"

dpo_path = "/data/output/Qwen2.5-7b-dpo/checkpoint-93/"

model = AutoModelForCausalLM.from_pretrained(
    dpo_path,
    device_map=None,                                # device_map="auto"条件启用	仅在多GPU环境启用分片，单卡时设为None避免冲突
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(
    dpo_path, 
    use_fast=False, 
    trust_remote_code=True,         # If the model is defined by a remote code, trust it
) 

lora_path = "/data/workbench/checkpoints/"
model = PeftModel.from_pretrained(
    model,
    lora_path,
)

model = model.merge_and_unload()

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

# Tokenze the datasets
print("\nData Mapping...\n")

datasets_path = "./datasets/DISC-Law-SFT-Pair-QA-released.jsonl"
train_dataset = pd.read_json(datasets_path, lines=True)
train_dataset = train_dataset.sample(n=10000, random_state=42)
test_dataset = pd.read_json(datasets_path, lines=True)[:10]

train_dataset = Dataset.from_pandas(train_dataset)
train_dataset = train_dataset.map(
    process_func, 
    remove_columns=train_dataset.column_names
    )

train_dataset = pd.DataFrame(train_dataset)
train_dataset = Dataset.from_pandas(train_dataset)
train_dataset = train_dataset.map(
    process_func, 
    remove_columns=train_dataset.column_names
    )

# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

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

collator = DataCollatorForSeq2Seq(          # collator is a function that is used to batch input data 
    model=model,
    tokenizer=tokenizer,
    padding=True,
)

args = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    logging_steps=20,                       # log every 20 steps
    save_steps=100,
    save_strategy="epoch",                  # save by `save_steps`
                                            # better chose the "epoch" strategy, otherwise your disk will explode
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    optim="adamw_hf",
    warmup_ratio=0.1,
    weight_decay=0.003,
    bf16=True,
)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

swanlab.init(project="Qwen-Law", experiment_name="Qwen-sft")
swanlab_callback = SwanLabCallback(project="Qwen-Law", experiment_name="Qwen-sft")

def predict(messages, model, tokenizer):
    # device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# def before_sft_test():
#     print("未开始 SFT, 先取 3 条主观评测: ")
#     test_text_list = []
#     for i, file in test_dataset[:3].iterrows():
#         instruction = file["instruction"]
#         input_value = file["input"]

#         messages = [
#             {"role": "system", "content": f"{instruction}"},
#             {"role": "user", "content": f"{input_value}"},
#         ]
        
#         response = predict(messages, model, tokenizer)
#         messages.append({"role": "assistant", "content": f"{response}"})
#         result_text = f"[Q]{messages[1]['content']}\n[LLM]{messages[2]['content']}\n"

#         print(result_text)

#         test_text_list.append(swanlab.Text(result_text, caption=response))

#     swanlab.log({"Prediction": test_text_list}, step=0)

# before_sft_test()

sft_trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    data_collator=collator,
    callbacks=[swanlab_callback],
    # packing=False,
    # max_seq_length=1024,
    # packing=False,
)

print("\nSFT training...\n")
sft_trainer.train()

save_path = "./checkpoints/"
sft_trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

swanlab.finish()
print("\nFinished...\n")
