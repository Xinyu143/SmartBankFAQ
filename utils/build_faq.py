import pandas as pd

def build_faq_from_banking77(path="data/banking77.csv"):
    df = pd.read_csv(path)

    # 按照 label 分组，每组只取一个问题作为代表
    faq_df = df.groupby("label")["text"].first().reset_index()
    faq_df.columns = ["label", "question"]  # 重命名列
    return faq_df
