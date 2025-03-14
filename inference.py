from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from peft import PeftModel

lora_path = "/data/workbench/checkpoints/"
base_path = "/data/qwen/Qwen2___5-7B-Instruct"
dpo_path = "/data/output/Qwen2.5-7b-dpo/checkpoint-10/"

model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(lora_path)

def clean_memory():
    """深度清理显存和内存"""
    global model, tokenizer
    
    # 删除模型和分词器
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    
    # 清空PyTorch缓存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()  # 重置内存统计
    
    # 强制垃圾回收
    gc.collect()
    
    # 释放Python内部缓存
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # 仅限Linux
    except:
        pass


INSTRUCTION = "你是一个精通法律的专家，请根据用户的问题给出专业的回答。"

try:
    while True:
        prompt = []
        prompt.append({"role": "system", "content": INSTRUCTION})

        print("--------------------")
        print("Type:\"quit\" to quit.\n")
        question = input('User: ' + '\n')

        if question == "quit":
            clean_memory()
            break

        prompt.append({"role": "user", "content": question})

        input_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        if model_inputs.input_ids.size()[1]>32000:
            break

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
        )

        if len(generated_ids)>32000:
            break

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('\n' + 'Assistant:' + '\n' + response)

        print("--------------------")

finally:
    clean_memory()

