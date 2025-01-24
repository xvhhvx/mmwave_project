import numpy as np
import matplotlib.pyplot as plt
from readDCA1000 import readDCA1000
from compute_background_and_subtraction import compute_background_and_subtraction as CBAS

#folderPath = r"mmWave_1\2_SymmetricalPosition\1_Radar_Raw_Data\position_ (7)" # 文件夹路径
folderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/position_ (7)" # 文件夹路径
filePath = r'adc_3GHZ_position7_ (4).bin'
binPath = folderPath + '/' + filePath
data = readDCA1000(binPath, 12, 200) # numChirps1200 * num_rx12 * numADCSamples200
data = data[:,np.r_[0:4, 8:12], :] # pick TX1, TX3　only

data_minus = CBAS(data.transpose(0,2,1), 1.5)[1].transpose(0,2,1)

slice0 = data_minus[500, :, :] # 切第500个chirp出来看效果，500是随便选的
fft2d = np.fft.fft2(slice0)
fft2d_shifted = np.fft.fftshift(fft2d)
magnitude = 20 * np.log10(np.abs(fft2d_shifted))
magnitude0 = (magnitude - np.mean(magnitude)) / np.std(magnitude)

peak_idx = np.argmax(np.abs(fft2d_shifted[1]))
ROI = 4 # range bin
start = max(0, peak_idx - ROI)
end = min(fft2d_shifted.shape[1], peak_idx + ROI + 1)
sliced_data = fft2d_shifted[:, start:end]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax1.imshow(magnitude0, cmap='jet', aspect='auto', origin='lower')
ax1.set_title('Original FFT')
plt.colorbar(im1, ax=ax1)

magnitude_sliced = 20 * np.log10(np.abs(sliced_data))
magnitude_sliced0 = (magnitude_sliced - np.mean(magnitude_sliced)) / np.std(magnitude_sliced)
im2 = ax2.imshow(magnitude_sliced0, cmap='jet', aspect='auto', origin='lower')
ax2.set_title(f'Sliced FFT (rows {start}-{end-1})')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
