import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_name = "qwen/Qwen2.5-7B-Instruct"
sava_path = "/data/" # 在服务器上最好保存到 /data/ 下面，否则 /root 会爆掉​

model_dir = snapshot_download(model_name, 
                              cache_dir=sava_path, 
                              revision='master')

