import numpy as np
from scipy.ndimage import maximum_filter
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def perform_2d_fft(r):
  """
  对输入的雷达信号执行2D-FFT变换
  
  参数:
    r: 三维numpy数组，形状为(M, K, N)，其中:
       M = 线性调频脉冲数量（可变长度）
       K = 接收天线数量（固定为8）
       N = ADC采样点数（每个项目固定为200）
  
  返回:
    R: 2D-FFT后的三维复数numpy数组，形状为(M, K, N_fft)
       N_fft = 256（FFT点数），K = 8
  """
  M, K, N_samples = r.shape
  N_fft = 256  # 指定的FFT点数
  
  # 初始化输出数组
  R = np.zeros((M, K, N_fft), dtype=np.complex64)
  
  # 对每个线性调频脉冲执行2D-FFT
  for m in range(M):
    # 沿距离维度(N)进行第一次FFT
    # 从200点补零到256点
    for k in range(K):
      R[m, k, :] = np.fft.fft(r[m, k, :], n=N_fft)
    
    # 沿方位维度(K)进行第二次FFT
    for n in range(N_fft):
      R[m, :, n] = np.fft.fft(R[m, :, n])
  
  return R

def remove_static_background(R, beta=0.9, first_frame_scale=0.8):
  """ 
  使用循环滤波器去除静态背景
  
  参数:
    R: 三维复数numpy数组，形状为(M, K, N)
    beta: 背景更新的权重因子（默认值：0.9）
    first_frame_scale: 第一帧背景缩放因子（默认值：0.8）
                      用于避免第一帧信号被完全抵消
  
  返回:
    R_minus: 去除背景后的信号，形状为(M, K, N)
  """
  M, K, N = R.shape
  
  # 使用第一帧初始化背景
  B = np.zeros_like(R, dtype=np.complex64)
  B[0, :, :] = R[0, :, :]
  
  # 初始化输出数组
  R_minus = np.zeros_like(R, dtype=np.complex64)
  
  # Update background and remove it for each frame
  for m in range(1, M):
    # 使用公式(3)更新背景
    B[m, :, :] = beta * B[m-1, :, :] + (1 - beta) * R[m, :, :]
    
    # 使用公式(4)去除背景
    R_minus[m, :, :] = R[m, :, :] - B[m, :, :]
  
  # 对于第一帧使用缩放后的背景，避免信号完全被抵消
  R_minus[0, :, :] = R[0, :, :] - first_frame_scale * B[0, :, :]
  
  return R_minus

def detect_1d_cfar(R_minus, guard_cells=2, training_cells=4, pfa=1e-3):
  """
  执行1D-CFAR检测以寻找信号中的峰值，通过对所有天线K求平均降维
  
  参数:
    R_minus: 去除背景后的信号，形状为(M, K, N)
    guard_cells: 保护单元数（距离）
    training_cells: 训练单元数（距离）
    pfa: 虚警概率
  
  返回:
    peak: peaks的众数
    peaks: 每帧的峰值坐标列表(m, n)，不包含天线维度K
  """
  M, K, N = R_minus.shape
  peaks = []
  
  # 处理每一帧
  for m in range(M):
    # 对所有天线K求平均降维
    frame_data = np.mean(np.abs(R_minus[m, :, :]), axis=0)  # 形状为(N,)
    
    # 计算训练单元数量
    N_train = 2 * training_cells
    
    # 根据期望的虚警概率计算阈值因子
    alpha = N_train * (pfa**(-1/N_train) - 1)
    
    # 初始化检测图
    detection_map = np.zeros(N, dtype=bool)
    
    # 滑动窗口遍历数据
    for n in range(training_cells + guard_cells, N - training_cells - guard_cells):
      # 定义被检测单元(CUT)
      cut = frame_data[n]
      
      # 提取左侧训练区域（排除保护单元）
      left_train = frame_data[n - training_cells - guard_cells:n - guard_cells]
      
      # 提取右侧训练区域（排除保护单元）
      right_train = frame_data[n + guard_cells + 1:n + guard_cells + training_cells + 1]
      
      # 合并训练区域
      train_cells = np.concatenate((left_train, right_train))
      
      # 从训练单元计算噪声水平
      noise_level = np.mean(train_cells)
      
      # 应用阈值
      threshold = alpha * noise_level
      
      # 比较被检测单元与阈值
      if cut > threshold:
        detection_map[n] = True
    
    # 在检测结果中寻找局部极大值
    # 使用最大值滤波器寻找局部极大值
    detection_data = frame_data * detection_map
    max_filtered = maximum_filter(detection_data, size=3)
    local_max = (detection_data == max_filtered) & detection_map
    
    # 获取峰值坐标
    peak_indices = np.where(local_max)[0]
    
    # 存储本帧最强峰值
    if len(peak_indices) > 0:
      # 寻找最强峰值
      peak_values = frame_data[peak_indices]
      strongest_idx = np.argmax(peak_values)
      n_peak = peak_indices[strongest_idx]
      
      # 只保存帧索引和距离索引，不包含天线维度
      peaks.append((m, n_peak))
    else:
      # 如果未找到峰值，使用帧内最大值
      n_peak = np.argmax(frame_data)
      peaks.append((m, n_peak))
  
  peaksArray = np.array(peaks)[:, 1]
  peak = stats.mode(peaksArray).mode
  return peak, peaks

def extract_roi(R_minus, peak):
  """
  围绕检测到的峰值提取感兴趣区域(ROI)
  
  参数:
    R_minus: 去除背景后的信号，形状为(M, K, N)
    peak: 整体峰值(N)
  
  返回:
    R_hat: 提取的ROI信号，形状为(M, 8, 8)
  """
  N = R_minus.shape[2]
  if (peak - 4 < 0):
    R_hat = R_minus[:, :, 0:8]
  elif (peak + 4 > N):
    R_hat = R_minus[:, :, N-8:N]
  else:
    R_hat = R_minus[:, :, peak-4:peak+4]
  
  return R_hat

def visualize_fft_result(R, R_minus, frame_idx=0, title="2D-FFT Result", figsize=(15, 15), peak=None, R_hat=None):
  """
  可视化指定帧的2D-FFT结果，总是显示R和R_minus，并标记峰值位置，可选显示R_hat
  
  参数:
    R: 三维复数numpy数组，形状为(M, K, N)，原始2D-FFT结果
    R_minus: 三维复数numpy数组，形状为(M, K, N)，去除背景后的结果
    frame_idx: 要可视化的帧索引，可以是单个整数或整数数组
    title: 图表标题
    figsize: 图表尺寸
    peak: 在R_minus中标记的峰值位置(N维度)，单个整数
    R_hat: 可选的ROI区域，形状为(M, K, roi_n)
  """
  # 将单个帧索引转换为列表，以便统一处理
  if isinstance(frame_idx, (int, np.integer)):
    frame_indices = [frame_idx]
  else:
    frame_indices = frame_idx
  
  # 为每个帧生成可视化
  for idx in frame_indices:
    # 确定子图数量（有R_hat则为3，否则为2）
    n_plots = 3 if R_hat is not None else 2
    
    # 创建子图
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # 上图：原始FFT结果
    fft_mag1 = np.abs(R[idx])
    im1 = axes[0].imshow(20*np.log10(fft_mag1), aspect='auto', cmap='viridis', 
                        origin='lower', extent=[0, fft_mag1.shape[1], 0, fft_mag1.shape[0]])
    fig.colorbar(im1, ax=axes[0], label='Magnitude (dB)')
    axes[0].set_xlabel('Range (bins)')
    axes[0].set_ylabel('Azimuth (virtual array)')
    axes[0].set_title(f"{title} - Frame {idx}")
    
    # 中图：去除背景后的FFT结果，并标记峰值
    fft_mag2 = np.abs(R_minus[idx])
    im2 = axes[1].imshow(20*np.log10(fft_mag2), aspect='auto', cmap='viridis', 
                        origin='lower', extent=[0, fft_mag2.shape[1], 0, fft_mag2.shape[0]])
    fig.colorbar(im2, ax=axes[1], label='Magnitude (dB)')
    
    # 如果提供了峰值位置，则在图中标记
    if peak is not None:
      # 在每个方位角位置标记峰值
      for k in range(fft_mag2.shape[0]):
        axes[1].plot(peak, k, 'rx', markersize=8)
      
      # 添加峰值位置的文本标注
      axes[1].text(peak + 5, fft_mag2.shape[0]//2, f'Peak: {peak}', 
                  color='red', fontsize=12, verticalalignment='center')
    
    axes[1].set_xlabel('Range (bins)')
    axes[1].set_ylabel('Azimuth (virtual array)')
    axes[1].set_title(f"Background Removed - Frame {idx}")
    
    # 如果提供了R_hat，则显示第三个子图
    if R_hat is not None:
      fft_mag3 = np.abs(R_hat[idx])
      im3 = axes[2].imshow(20*np.log10(fft_mag3), aspect='auto', cmap='viridis', 
                          origin='lower', extent=[0, fft_mag3.shape[1], 0, fft_mag3.shape[0]])
      fig.colorbar(im3, ax=axes[2], label='Magnitude (dB)')
      axes[2].set_xlabel('Range (bins)')
      axes[2].set_ylabel('Azimuth (virtual array)')
      axes[2].set_title(f"ROI (R_hat) - Frame {idx}")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def raw2Rhat(r, visualize=False, frame_to_visualize=0):
  """
  根据MM-FGRM雷达系统要求处理雷达信号
  
  参数:
    r: 三维numpy数组，形状为(M, K, N)，其中:
       M = 线性调频脉冲数量（可变长度）
       K = 接收天线数量（固定为8）
       N = ADC采样点数（每个项目固定为200）
    visualize: 是否可视化处理步骤
    frame_to_visualize: 要可视化的帧索引，接受整数或整数数组
  
  返回:
    R_hat: 处理后的实数矩阵，形状为(M, 8, 8)
  """
  # 步骤1: 执行2D-FFT
  R = perform_2d_fft(r)
  
  # 步骤2: 去除静态背景
  R_minus = remove_static_background(R)

  # 步骤3: 使用1D-CFAR检测峰值
  peak, peaks = detect_1d_cfar(R_minus)

  # 步骤4: 提取峰值周围的ROI
  R_hat = extract_roi(R_minus, peak)

  # 如果需要则可视化FFT结果
  if visualize:
    visualize_fft_result(R, R_minus, frame_idx=frame_to_visualize, 
                        title="2D-FFT Result", peak=peak, R_hat=R_hat)

  return R_hat

# Example usage
if __name__ == "__main__":
  from readDCA1000 import readDCA1000
  folderPath = r"Sample" # 文件夹路径
  filePath = r'adc_data_4.bin'
  binPath = folderPath + '/' + filePath
  data = readDCA1000(binPath, 12, 200)
  data = data[:,np.r_[0:4, 8:12], :]

  # 启用可视化，展示改进后的功能
  result = raw2Rhat(data, visualize=True, frame_to_visualize=[0, 10, 1000, 3000, 5000])
  
  # 打印结果形状
  print(f"Input shape: {data.shape}")
  print(f"Output shape: {result.shape}")
  print(f"Data structure check: {np.dtype(result[0,0,0])}")
