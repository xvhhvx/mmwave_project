import numpy as np
import os
import pandas as pd
from readDCA1000 import readDCA1000
from toR_hat import toRhat

def getData(oriFolderPath, split_count=1):
    """
    Load and transform bin data into numpy arrays
    Note: No preprocess (eg. FFT or toR_hat) to the data
    
    Parameters:
    -----------
    oriFolderPath : str
        Path to folder containing bin files
    split_count : int, optional
        Number of parts to split the numChirps into, default is 1 (no split)
    
    Returns:
    --------
    np.ndarray of shape (file_num, minutes, numChirps per minute, num_rx, numADCSamples)
    """
    preProcessData = []
    
    bin_files = [f for f in os.listdir(oriFolderPath) if f.endswith('.bin')]
    for file_name in bin_files:
        file_path = os.path.join(oriFolderPath, file_name)
    
        try:
            # Process the file using toRhat function
            data = readDCA1000(file_path, 12, 200) # numChirps6000 * num_rx12 * numADCSamples200
            data = data[:,np.r_[0:4, 8:12], :] # pick TX1, TX3　only
            
            # Split the data into split_count parts along numChirps
            splits = np.array_split(data, split_count, axis=0)
            
            # Using toRhat to process
            processed_splits = []
            for split in splits:
                processed_split = toRhat(split)
                processed_splits.append(processed_split)
            
            preProcessData.append(processed_splits)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    result = np.array(preProcessData)
    print(result.shape)
    return result

def mergeData(data):
    """
    Merge the first two dims of data into one dim
    
    Parameters:
    -----------
    data : np.ndarray
        Data to be merged
    
    Returns:
    --------
    np.ndarray of shape (file_num * minutes, numChirps per minute, num_rx, numADCSamples)
    """
    # Get the shape of the array
    file_num, parts, chirps, rx, samples = data.shape
    # Reshape to merge file_num and parts dimensions
    data = data.reshape(-1, chirps, rx, samples)
    return data

def separateComplexData(data):
    """
    Separate complex data into real and imaginary parts
    
    Parameters:
    -----------
    data : np.ndarray
        Complex data array
    
    Returns:
    --------
    np.ndarray with shape (..., 2) where last dimension contains real and imaginary parts
    """
    # Extract real and imaginary parts
    real_part = np.real(data)
    imag_part = np.imag(data)
    
    # Stack them along a new last dimension
    result = np.stack((real_part, imag_part), axis=-1)
    
    return result

def getVali(valiPath):
    """
    Load and preprocess validation data
    
    Parameters:
    -----------
    oriFolderPath : str
        Path to folder containing bin files
    
    Returns:
    --------
    np.ndarray
    """
    labels_df = pd.read_excel(valiPath, usecols=['HeartRate'])
    labels = labels_df['HeartRate'].to_numpy()
    return labels


if __name__ == "__main__":
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample"
    valiPath = oriFolderPath + "/HR.xlsx"
    X = getData(oriFolderPath, 5)
    X = mergeData(X)
    X_separated = separateComplexData(X)
    # Print the shape of each part
    print(X.shape)
    print("After separating complex values:", X_separated.shape)

    y = getVali(valiPath)
    print(y.shape)
    