## 数据集与模型的下载
模型选型为 `Qwen2.5-7B Instruction`，通过 `modelscope` 下载，数据集均从 hf 的镜像 `hf-mirror.com` 中获取。
通过 `python download_*.py` 获取所需要下载的数据。

## 数据清洗与处理
`python data_process.py` 用来处理 sft 的数据，同理 `data_process_dpo`。

## 训练
详见我的博客。可以通过 `accelerate` 降低显存，加速训练过程。配置文件如 `acclerate_config.yaml`，运行 bash 文件即可开始训练。
> 后续会公开。

## 推理
通过 `python inference.py` 进行推理。

## 风格加强
在 `style.py` 中通过 RL 对少数 pair-wise 的数据进行 DPO 训练，以达到完善风格的需求。
> 而不是 SFT，用 SFT 可能会过过拟合，而且少量轮次的 SFT 效果并不理想。