import numpy as np
import os
from readDCA1000 import readDCA1000

def getData(oriFolderPath, split_count=1):
    """
    Load and preprocess data
    
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
            
            # Split the data into split_count parts along numChirps)
            splits = np.array_split(data, split_count, axis=0)
            
            preProcessData.append(splits)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    result = np.array(preProcessData)
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



if __name__ == "__main__":
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample"
    X = getData(oriFolderPath, 5)
    X = mergeData(X)
    # Print the shape of each part
    print(X.shape)