import numpy as np
from readDCA1000 import readDCA1000
from compute_background_and_subtraction import compute_background_and_subtraction as CBAS

def toRhat(binPath):
  '''
  输入：
    binPath为雷达.bin文件

  输入：
    转换为原文对应的chirp*8*8 ROI ndarray

  输出：
    54*1200*8*8*2
  '''
  data = readDCA1000(binPath, 12, 200) # numChirps1200 * num_rx12 * numADCSamples200
  data = data[:,np.r_[0:4, 8:12], :] # pick TX1, TX3　only

  data_minus = CBAS(data.transpose(0,2,1), beta= 1.5)[1].transpose(0,2,1) # 消除背景干扰

  meanSample = np.mean(data_minus, axis= 0);
  fft2d = np.fft.fft2(meanSample)
  fft2d_shifted = np.fft.fftshift(fft2d)
  peak_idx = np.argmax(np.abs(fft2d_shifted[1]))
  ROI = 4 # range bin
  start = max(0, peak_idx - ROI)
  end = min(fft2d_shifted.shape[1], peak_idx + ROI)

  num_chirps = data_minus.shape[0]
  fft2dAll = np.zeros((num_chirps, 8, 8), dtype='complex')
  for chirp in range(num_chirps):
    fft2dTmp = np.fft.fft2(data_minus[chirp,:,:])
    fft2dTmp = np.fft.fftshift(fft2dTmp)
    fft2dTmp = fft2dTmp[:, start:end]
    fft2dAll[chirp] = fft2dTmp

  # Separate real and imaginary parts and combine into a list of lists
  #combined_list = []
  combined_array = np.zeros((num_chirps, 8, 8, 2), dtype='float32')
  for chirp in range(num_chirps):
      real_part = fft2dAll[chirp].real
      imaginary_part = fft2dAll[chirp].imag
      
      # 计算当前chirp的最大绝对值以进行归一化
      max_val = max(np.max(np.abs(real_part)), np.max(np.abs(imaginary_part)))
      if max_val > 0:
        real_part = real_part / max_val
        imaginary_part = imaginary_part / max_val

      # 转换为float32并存入数组
      combined_array[chirp] = np.stack((real_part.astype('float32'), 
                                        imaginary_part.astype('float32')), 
                                        axis=-1)
      #combined_array[chirp] = np.stack((real_part, imaginary_part), axis=-1)

  return combined_array

# fft2dAll 为1200x8x8的R^数组

# 示例 #################################################
if __name__ == '__main__':
  #folderPath = r"mmWave_1\2_SymmetricalPosition\1_Radar_Raw_Data\position_ (7)" # 文件夹路径
  folderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/position_ (7)" # 文件夹路径
  filePath = r'adc_3GHZ_position7_ (4).bin'
  binPath = folderPath + '/' + filePath
  res = toRhat(binPath)
  #import matplotlib.pyplot as plt
  #plt.imshow(abs(res[800]), cmap='jet', aspect='auto', origin='lower')