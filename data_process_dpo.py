import pandas as pd

df = pd.read_parquet(
    '/data/workbench/datasets/train-00000-of-00001.parquet',  
    engine='pyarrow'
    )  
new_columns = ["prompt", "chosen", "rejected"]
df.columns = new_columns

# ----------------------------------------------------
def is_valid_paragraph(para: str) -> bool:
    """验证单个自然段是否满足条件"""
    stripped = para.strip()
    # 必须同时满足：非空字符串 + 以句号结尾
    return bool(stripped) and stripped.endswith("。")

def process_text(text: str) -> str:
    """
    处理文本的核心逻辑
    参数：
        text : 原始文本（用换行符分隔的自然段）
    返回：
        处理后的文本，保留原段落结构
    """
    # 分割自然段并保留原始换行结构
    original_paragraphs = text.split('\n')
    
    # 过滤有效段落
    valid_paragraphs = []
    for para in original_paragraphs:
        # 保留原始段落格式，仅过滤不符合条件的
        if is_valid_paragraph(para):
            valid_paragraphs.append(para)
    
    # 重新组合有效段落
    return '\n'.join(valid_paragraphs)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清洗DataFrame中的指定列"""
    for col in ['chosen', 'rejected']:
        df[col] = df[col].astype(str).apply(process_text)
    return df
# ----------------------------------------------------

df = clean_dataframe(df)
df.to_csv("/data/workbench/datasets/law-gpt.csv")