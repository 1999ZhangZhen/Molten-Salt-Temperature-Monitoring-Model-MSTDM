import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 要处理的文件夹路径
folder_path = "11"
Normalized_path = "22"

# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx"):
        # 读取 Excel 文件
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path, engine='openpyxl')

        # 获取特征列和类别列
        features = df.iloc[:, :-1].values  # 倒数第二列之前的列作为特征
        labels = df.iloc[:, -1].values  # 倒数第二列作为类别

        # 归一化特征
        normalizer = MinMaxScaler()
        normalized_features = normalizer.fit_transform(features)

        # 创建 DataFrame 存储归一化后的数据
        df_normalized = pd.DataFrame(normalized_features, columns=df.columns[:-1])
        df_normalized[df.columns[-1]] = labels  # 添加类别列

        # 生成保存文件的名称，并保存归一化后的数据
        output_filename = "Normalized_" + filename
        output_file_path = os.path.join(Normalized_path, output_filename)
        df_normalized.to_excel(output_file_path, index=False)
