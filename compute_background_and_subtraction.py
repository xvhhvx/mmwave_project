import numpy as np

def compute_background_and_subtraction(R: np.ndarray, beta: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
  """
  计算静态背景 B 和去除背景后的信号 R_minus。
  
  参数:
    R (np.ndarray): 输入的雷达信号矩阵，形状为 (M, N, K)，其中：
                    M 为时间帧数，N 为 range bin 数，K 为方位角分辨率。
    beta (float): 递归滤波器的衰减因子，默认为 0.9。
  
  返回:
    B (np.ndarray): 静态背景矩阵，形状与 R 相同。
    R_minus (np.ndarray): 去除背景后的信号矩阵，形状与 R 相同。
  """
  M, N, K = R.shape
  B = np.zeros_like(R, dtype='complex')  # 假设 R 为复数信号
  B[0] = R[0]  # 初始化第一帧背景
  
  # 递归计算背景 B[m]
  for m in range(1, M):
    B[m] = beta * B[m-1] + (1 - beta) * R[m]
  
  # 计算 R_minus = R - B
  R_minus = R - B
  
  return B, R_minus

# 示例用法
if __name__ == "__main__":
  # 生成模拟数据（复数信号）
  M, N, K = 100, 256, 8
  R = np.random.randn(M, N, K) + 1j * np.random.randn(M, N, K)
  
  # 计算 B 和 R_minus
  B, R_minus = compute_background_and_subtraction(R, beta=0.9)
  
  # 验证结果
  print("B 的形状:", B.shape)         # 输出: (100, 256, 8)
  print("R_minus 的形状:", R_minus.shape) # 输出: (100, 256, 8)
  print("第一个时间帧的 R_minus 是否全零:", np.allclose(R_minus[0], 0))  # 输出: True