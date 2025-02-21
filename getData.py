import pandas as pd
import os
import numpy as np

def process_csv_file_data(file_path):
    # 读取指定列的数据
    df = pd.read_csv(file_path, usecols=['radar24-I', 'radar24-Q', 'radar10-I'])
    
    # 确保数据总行数是600000（10组，每组60000行）
    total_rows = 600000
    if len(df) != total_rows:
        raise ValueError(f"Expected {total_rows} rows, but got {len(df)} rows")
    
    # 将数据重塑为10组，每组60000*3
    groups = []
    for i in range(10):
        start_idx = i * 60000
        end_idx = (i + 1) * 60000
        group_data = df.iloc[start_idx:end_idx].values  # 转换为numpy数组
        groups.append(group_data)
    return groups

def process_bin_file_vali(file_path):
    # 读取指定列的数据
    df = pd.read_csv(file_path, usecols=['RESP', 'ECG'])
    
    # 确保数据总行数是600000（10组，每组60000行）
    total_rows = 600000
    if len(df) != total_rows:
        raise ValueError(f"Expected {total_rows} rows, but got {len(df)} rows")
    
    # 创建10x2的数组用于存储每60000行数据的标签
    labels = np.zeros((10, 2))
    
    # 处理每60000行数据
    for i in range(10):
        start_idx = i * 60000
        # 因为每60000行的值相同，所以只需要取第一行的值即可
        labels[i] = [df['RESP'].iloc[start_idx], df['ECG'].iloc[start_idx]]
    
    return labels

def getData(oriFolderPath, bin_files):
    data = []
    for file_name in bin_files:
        file_path = os.path.join(oriFolderPath, file_name)
    
        try:
            # 处理文件
            data_groups = process_csv_file_data(file_path)
        
            data.append(data_groups)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return data

def getVali(oriFolderPath, bin_files):
    vali = []
    for file_name in bin_files:
        file_path = os.path.join(oriFolderPath, file_name)
    
        try:
            # 处理文件
            labels = process_bin_file_vali(file_path)
        
            vali.append(labels)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return vali

if __name__ == "__main__":
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/data"
    bin_files = [f for f in os.listdir(oriFolderPath) if f.endswith('.csv') and not f.startswith('._')] #忽略._开头的隐藏文件
    data = []
    vali = []
    data = getData(oriFolderPath, bin_files)
    vali = getVali(oriFolderPath, bin_files)
    print(len(data[0][0]), len(vali[0][0]))