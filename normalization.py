import os
import pandas as pd
import numpy as np


def process_file(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 分离特征和标签
    label_col = df['Label']
    features = df.drop('Label', axis=1)

    # 预处理：删除包含空值或无穷值的行
    # 创建布尔掩码标记无效值（NaN或inf/-inf）
    invalid_mask = (
            features.isnull() |
            features.isin([np.inf, -np.inf])
    ).any(axis=1)

    # 过滤有效数据
    clean_features = features[~invalid_mask]
    clean_labels = label_col[~invalid_mask]

    # 检查剩余数据是否为空
    if clean_features.empty:
        print(f"警告: {os.path.basename(input_path)} 预处理后无有效数据，已跳过")
        return

    # 按列归一化到0-255（保留小数）
    normalized_features = pd.DataFrame()

    for column in clean_features.columns:
        col_data = clean_features[column]
        min_val = col_data.min()
        max_val = col_data.max()

        # 处理常数列
        if max_val == min_val:
            normalized_col = pd.Series([0.0] * len(col_data))
        else:
            # 归一化计算并保留小数
            normalized_col = ((col_data - min_val) / (max_val - min_val)) * 255.0

        normalized_features[column] = normalized_col.round(4)  # 保留4位小数

    # 合并标签列
    result_df = pd.concat([normalized_features, clean_labels.reset_index(drop=True)], axis=1)

    # 保存结果
    result_df.to_csv(output_path, index=False)


def batch_process(input_folder, output_folder):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 处理所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)

    print("处理完成！处理结果保存在：", output_folder)


if __name__ == "__main__":
    input_folder = r"D:\final\split_labels"
    output_folder = r"D:\final\CICIDS2017_normalize"
    # 执行批处理
    batch_process(input_folder, output_folder)


