import torch
import pickle
import numpy as np
from simpleCNN import ChirpRegressionModel
from getData import getData, getVali
from data_validation import check_tensor_valid, safe_normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model = ChirpRegressionModel().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def predict(model, input_data, device, normalization_params=None, batch_size=32):
    """
    使用训练好的模型进行预测，并处理归一化逆变换
    
    Args:
        model: 训练好的模型
        input_data: 输入数据 (与模型期望的格式相同)
        device: 计算设备
        normalization_params: 归一化参数字典，如果为None则尝试加载
        batch_size: 批处理大小，用于大型数据集
        
    Returns:
        反归一化后的预测结果
    """
    # 如果未提供归一化参数，尝试加载
    if normalization_params is None:
        try:
            with open('normalization_params.pkl', 'rb') as f:
                normalization_params = pickle.load(f)
            print("成功加载归一化参数")
        except FileNotFoundError:
            raise ValueError("归一化参数未提供且无法加载。请提供normalization_params或确保normalization_params.pkl存在")
    
    # 确保输入数据有效
    if not isinstance(input_data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"输入数据类型必须是numpy数组或PyTorch张量，而不是{type(input_data)}")
    
    # 转换为张量
    if not isinstance(input_data, torch.Tensor):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    else:
        input_tensor = input_data
    
    # 验证输入数据是否有效
    if not check_tensor_valid(input_tensor, "Model input"):
        raise ValueError("输入数据包含无效值 (NaN 或 Inf)")
    
    # 输入归一化
    X_mean = torch.tensor(normalization_params['X_mean']).to(device)
    X_std = torch.tensor(normalization_params['X_std']).to(device)
    input_tensor = input_tensor.to(device)  # 确保input_tensor在相同设备上
    input_normalized = (input_tensor - X_mean) / (X_std + 1e-7)
    
    # 再次检查归一化后的数据
    if not check_tensor_valid(input_normalized, "Normalized input"):
        print("警告: 归一化后的输入数据可能包含无效值，尝试使用安全归一化")
        input_normalized, _, _ = safe_normalize(input_tensor, dim=0, eps=1e-7)
    
    # 对大型数据集进行批处理预测
    model.eval()
    total_samples = input_normalized.shape[0]
    outputs_list = []
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch = input_normalized[i:end_idx].to(device)
            
            # 预测
            batch_output = model(batch)
            
            # 检查输出有效性
            if not check_tensor_valid(batch_output, f"Model output batch {i//batch_size}"):
                print(f"警告: 批次 {i//batch_size} 的输出包含无效值")
                # 用零替换无效输出
                batch_output = torch.zeros_like(batch_output)
                
            outputs_list.append(batch_output.cpu())
    
    # 合并所有批次的输出
    output_normalized = torch.cat(outputs_list, dim=0)
    
    # 输出反归一化
    output_original = output_normalized.numpy() * normalization_params['y_std'] + normalization_params['y_mean']
    
    return output_original

def predict_single(model, input_data, device, normalization_params=None):
    """
    预测单个样本
    
    Args:
        model: 训练好的模型
        input_data: 单个输入样本 (需扩展为批次维度)
        device: 计算设备
        normalization_params: 归一化参数
        
    Returns:
        单个样本的预测结果 (标量值)
    """
    # 确保输入数据形状正确，添加批次维度
    if isinstance(input_data, np.ndarray):
        if len(input_data.shape) == 3:  # 假设形状为 (H, W, C)
            input_data = np.expand_dims(input_data, 0)  # 扩展为 (1, H, W, C)
    elif isinstance(input_data, torch.Tensor):
        if len(input_data.shape) == 3:  # 假设形状为 (H, W, C)
            input_data = input_data.unsqueeze(0)  # 扩展为 (1, H, W, C)
    
    # 使用batch预测函数
    prediction = predict(model, input_data, device, normalization_params)
    
    # 返回单个标量值
    return prediction[0]

def evaluate_predictions(y_true, y_pred):
    """
    评估预测结果与真实值之间的差异
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        包含各种评估指标的字典
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算各种评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算平均相对误差
    relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-7)  # 避免除零
    mre = np.mean(relative_error) * 100  # 百分比
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MRE(%)": mre
    }

# 使用示例
if __name__ == "__main__":
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample/"
    valiPath = oriFolderPath + "/HR.xlsx"
    # 加载模型
    model_path = "checkpoints/best_model.pt"
    model, device = load_model(model_path)
    
    # 加载归一化参数
    with open('normalization_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    # 加载测试数据
    # ... 加载您的测试数据 ...
    test_data = getData(oriFolderPath, 5, loadFromFile=True, saveToFile=False)
    test_labels = getVali(valiPath)

    test_data = test_data[:5]  # 仅使用前5个样本进行演示
    test_labels = test_labels[:5]

    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")

    # 批量预测
    print("执行批量预测...")
    predictions = predict(model, test_data, device, params)
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测值范围: 最小值 = {np.min(predictions):.2f}, 最大值 = {np.max(predictions):.2f}")
    print(f"前5个预测结果: {predictions[:5]}")

    print("\n预测结果与真实值比较:")
    for i in range(len(test_labels)):
        print(f"样本 {i+1}: 预测值 = {predictions[i]:.2f}, 真实值 = {test_labels[i]:.2f}, " +
              f"误差 = {predictions[i] - test_labels[i]:.2f}")
    
    # 计算整体评估指标
    metrics = evaluate_predictions(test_labels, predictions)
    
    print("\n评估指标:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 使用PyTorch内置损失函数计算损失
    print("\n使用PyTorch损失函数计算:")
    
    # 转换为张量
    pred_tensor = torch.tensor(predictions, dtype=torch.float32)
    true_tensor = torch.tensor(test_labels, dtype=torch.float32)
    
    # 计算MSE损失
    mse_loss = torch.nn.MSELoss()(pred_tensor, true_tensor)
    # 计算MAE损失
    mae_loss = torch.nn.L1Loss()(pred_tensor, true_tensor)
    # 计算Huber损失 (结合MSE和MAE的优点)
    huber_loss = torch.nn.SmoothL1Loss()(pred_tensor, true_tensor)
    
    print(f"PyTorch MSE损失: {mse_loss.item():.4f}")
    print(f"PyTorch MAE损失: {mae_loss.item():.4f}")
    print(f"PyTorch Huber损失: {huber_loss.item():.4f}")
    
    # 可视化预测结果与真实值
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # 绘制预测值和真实值的比较
        x = np.arange(len(test_labels))
        width = 0.35
        
        plt.bar(x - width/2, test_labels, width, label='Ground Truth')
        plt.bar(x + width/2, predictions, width, label='Prediction')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Comparison of Predictions vs Ground Truth')
        plt.xticks(x)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图像
        plt.savefig('prediction_comparison.png', dpi=300)
        print("比较图已保存为 prediction_comparison.png")
        
    except ImportError:
        print("无法导入matplotlib，跳过可视化步骤")

    # 单样本预测，实际应该使用这个
    print("\n执行单样本预测...")
    single_sample = test_data[0]
    single_prediction = predict_single(model, single_sample, device, params)
    print(f"单样本预测结果: {single_prediction:.2f}")