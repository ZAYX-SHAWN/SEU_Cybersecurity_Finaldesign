import pandas as pd
import os

# 文件夹路径
folder_path = r'D:\final\cicids2017_original\MachineLearningCVE'

# 读取所有csv文件的文件名
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 用来存储所有DataFrame的列表
dfs = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# 合并所有DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# 保存合并后的csv文件
merged_df.to_csv(os.path.join(folder_path, 'merged.csv'), index=False)

print("合并完成，保存为 merged.csv")