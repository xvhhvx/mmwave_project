import torch
import pickle
import numpy as np
from simpleCNN import ChirpRegressionModel

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model = ChirpRegressionModel().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def predict(model, input_data, device, normalization_params=None):
    """
    使用训练好的模型进行预测，并处理归一化逆变换
    
    Args:
        model: 训练好的模型
        input_data: 输入数据 (与模型期望的格式相同)
        device: 计算设备
        normalization_params: 归一化参数字典，如果为None则尝试加载
        
    Returns:
        反归一化后的预测结果
    """
    # 如果未提供归一化参数，尝试加载
    if normalization_params is None:
        try:
            with open('normalization_params.pkl', 'rb') as f:
                normalization_params = pickle.load(f)
        except FileNotFoundError:
            raise ValueError("归一化参数未提供且无法加载。请提供normalization_params或确保normalization_params.pkl存在")
    
    # 转换为张量
    if not isinstance(input_data, torch.Tensor):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    else:
        input_tensor = input_data
    
    # 输入归一化
    input_normalized = (input_tensor - torch.tensor(normalization_params['X_mean']).to(device)) / (
        torch.tensor(normalization_params['X_std']).to(device) + 1e-7)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        output_normalized = model(input_normalized.to(device))
    
    # 输出反归一化
    output_original = output_normalized.cpu().numpy() * normalization_params['y_std'] + normalization_params['y_mean']
    
    return output_original

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "checkpoints/best_model.pt"
    model, device = load_model(model_path)
    
    # 加载归一化参数
    with open('normalization_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    # 加载测试数据
    # ... 加载您的测试数据 ...
    
    # 预测
    predictions = predict(model, test_data, device, params)
    print("预测结果:", predictions)