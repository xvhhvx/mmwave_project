import numpy as np

# Organize函数
def organize(raw_data, num_rx, num_samples):
  """Reorganizes raw ADC data into a full frame

  Args:
    raw_data (ndarray): Data to format
    # num_chirps: Number of chirps (auto calculated)
    num_rx: Number of receivers
    num_samples: Number of ADC samples included in each chirp

  Returns:
    ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

  """
  
  ret = np.zeros(len(raw_data) // 2, dtype=complex)

  # Separate IQ data
  ret[0::2] = raw_data[0::4] + 1j * raw_data[2::4]
  ret[1::2] = raw_data[1::4] + 1j * raw_data[3::4]
  return ret.reshape((-1, num_rx, num_samples))

def readDCA1000(bin_path, num_rx, numADCSamples):
  """
  
  Args:
    bin_path: path of .bin
    num_rx: number of RX Antennas, can be virtual
    numADCSamples: ADC samples pre chirp

  Returns:
    (Chirp, numRx, ADCsample) 3-d ndarray
  """
  raw_data = np.fromfile(bin_path, dtype=np.int16)
  adc_data = organize(raw_data, num_rx=num_rx, num_samples=numADCSamples)
  return adc_data