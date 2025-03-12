# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_path = "/data/qwen/Qwen2-1.5B-Instruct/"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# messages = [
#     {"role": "system","content": "你是一个精通中国法律的助手，你要以准确精练的风格回答用户的法律疑问。"},
#     {"role": "user", "content": "如果别人借钱不还怎么办？"},
#  ]

# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))

# def tokenize_and_format(data):
#     input_ids = tokenizer.apply_chat_template( # 这是 Qwen 自带的转换函数
#                                                # 但要求格式是 [{role: str, content: str}, {role: str, content: str}]
#                                                # 它会返回 tokens 的张量形式
#                                                # 但我们的数据是这种 {instruction: str, input: str, output: str, id: int} 
#                                                # 所以不管用不用它的函数，都得转换格式
#                                                # 方便起见，我自己实现了一个，详见下面
#         data,
#         tokenize=True,
#         add_generation_prompt=True,
#         truncation=True,
#         max_length=1024,
#     )
#     return input_ids


# import datasets
# from transformers import AutoTokenizer

# model_path = "/data/qwen/Qwen2___5-7B-Instruct/"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# def process_func(example):
#     """
#     将数据集进行预处理
#     """
#     MAX_LENGTH = 384
#     input_ids, attention_mask, labels = [], [], []
#     instruction = tokenizer(
#         f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
#         add_special_tokens=False,
#     )
#     response = tokenizer(f"{example['output']}", add_special_tokens=False)
#     input_ids = (
#         instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
#     )
#     attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
#     labels = (
#         [-100] * len(instruction["input_ids"])
#         + response["input_ids"]
#         + [tokenizer.pad_token_id]
#     )
#     if len(input_ids) > MAX_LENGTH:  # 做一个截断
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
        
#     return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# dataset_SFT = datasets.load_dataset("json", data_files = datasets_path) # 将数据集分割成 train 和 test

# DATAS = []
# def data_process(datasets) -> None:
#     for data in datasets:
#         instruction = data['instruction']
#         output = data['output']
#         DATAS.append(TEMPLATE.format(DEFAULT_SYSTEM_PROMPT, instruction, output))

#     with open("./datasets/chinese_law_fineturn.json", "w", encoding="utf-8") as f: # 将数据保存到本地
#         json.dump(DATAS, f)

# data_process(dataset_SFT['train'])

import json

TEMPLATE = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n'
INSTRUCTION = "你是一个精通法律的专家，请根据用户的问题给出专业的回答。"

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open (output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)

            new_data = {
                "instruction": INSTRUCTION,
                "input": data["input"],
                "output": data["output"]
            }

            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')

input_file = "/data/workbench/temp/DISC-Law-SFT-Pair-QA-released.jsonl"
output_file = "/data/workbench/datasets/DISC-Law-SFT-Pair-QA-released.jsonl"

process_jsonl(input_file, output_file)
print(f"处理完成, 输出文件：{output_file}")