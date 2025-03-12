from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

lora_path = "/data/workbench/checkpoints/"
base_path = "/data/qwen/Qwen2___5-7B-Instruct"
dpo_path = "/data/output/Qwen2.5-7b-dpo/checkpoint-3/"

model = AutoModelForCausalLM.from_pretrained(
    base_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )

model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# model = AutoModelForCausalLM.from_pretrained("/data/qwen/Qwen2___5-7B-Instruct/")
tokenizer = AutoTokenizer.from_pretrained(lora_path)

INSTRUCTION = "你是一个精通法律的专家，请根据用户的问题给出专业的回答。"
while True:
    prompt = []
    prompt.append({"role": "system", "content": INSTRUCTION})

    print("--------------------")
    question = input('User: ' + '\n')
    if question == "quit":
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


