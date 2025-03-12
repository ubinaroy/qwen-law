accelerate launch --config_file accelerate_config.yaml sft_train.py 2>&1 | tee output_sft.log
# 如果 OOM 了，重启 python 内核会释放显存！