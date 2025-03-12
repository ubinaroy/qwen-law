# pip install huggingface_hub​
from huggingface_hub import snapshot_download
 
# 使用cache_dir参数指定“本地路径”​
snapshot_download(repo_id="ShengbinYue/DISC-Law-SFT", 
                  repo_type="dataset",
                  cache_dir="/data/",
                  local_dir_use_symlinks=False, 
                  resume_download=True,
                  token=None)