import numpy as np
import os
import pandas as pd
from readDCA1000 import readDCA1000
from raw2Rhat_v2 import raw2Rhat

def getData(oriFolderPath, split_count=1, loadFromFile=False, saveToFile=True, processed_file_name="processed_data.npy"):
    """
    Load and transform bin data into numpy arrays
    Note: raw2Rhat, merge and real/imag seperation functions are used to process the data
    
    Parameters:
    -----------
    oriFolderPath : str
        Path to folder containing bin files
    split_count : int, optional
        Number of parts to split the numChirps into, default is 1 (no split)
    loadFromFile : bool, optional
        Whether to load processed data from file if available
    saveToFile : bool, optional
        Whether to save the processed data to file after processing
    processed_file_name : str, optional
        Name of the processed data file

    Returns:
    --------
    np.ndarray of shape (file_num * minutes, numChirps per minute, num_rx, numADCSamples, 2)
    """
        
    processed_file_path = os.path.join(oriFolderPath, processed_file_name)
    
    if loadFromFile and os.path.exists(processed_file_path):
        result = loadProcessedData(processed_file_path)
    else:
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
                print("Splitting")
                # Using toRhat to process
                processed_splits = []
                for split in splits:
                    processed_split = raw2Rhat(split)
                    print("raw2Rhat")
                    processed_splits.append(processed_split)
            
                preProcessData.append(processed_splits)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
        result = np.array(preProcessData)
        print(result.shape)

        result = mergeData(result)
        result = separateComplexData(result)

        # Only save if saveToFile is True
        if saveToFile:
            saveProcessedData(result, processed_file_path)
            
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

def getVali(valiPath, repeat_count=1):
    """
    Load and preprocess validation data with optional value repetition
    
    Parameters:
    -----------
    valiPath : str
        Path to validation Excel file
    repeat_count : int, optional
        Number of times to repeat each validation value, default is 1 (no repetition)
    
    Returns:
    --------
    np.ndarray
        Validation values, with each value repeated repeat_count times if specified
    """
    labels_df = pd.read_excel(valiPath, usecols=['HeartRate'])
    labels = labels_df['HeartRate'].to_numpy()
    
    if repeat_count > 1:
        labels = duplicateValidationValues(labels, repeat_count)
        
    return labels

def duplicateValidationValues(validation_values, repeat_count):
    """
    Duplicate each value in the validation array the specified number of times.
    
    Parameters:
    -----------
    validation_values : np.ndarray
        Original validation values array
    repeat_count : int
        Number of times each value should be repeated
    
    Returns:
    --------
    np.ndarray
        Array with each value repeated repeat_count times
    
    Example:
    --------
    Input: validation_values=[1,2,3], repeat_count=3
    Output: [1,1,1,2,2,2,3,3,3]
    """
    if not isinstance(validation_values, np.ndarray):
        validation_values = np.array(validation_values)
        
    # Use numpy's repeat function to duplicate each element
    duplicated_values = np.repeat(validation_values, repeat_count)
    
    return duplicated_values

def saveProcessedData(data, filepath):
    """
    Save processed data to a numpy file
    
    Parameters:
    -----------
    data : np.ndarray
        Data to be saved
    filepath : str
        Path where to save the data
    """
    np.save(filepath, data)
    print(f"Data saved to {filepath}")

def loadProcessedData(filepath):
    """
    Load processed data from a numpy file
    
    Parameters:
    -----------
    filepath : str
        Path to the saved numpy file
    
    Returns:
    --------
    np.ndarray
        The loaded data
    """
    if os.path.exists(filepath):
        data = np.load(filepath)
        print(f"Data loaded from {filepath}")
        return data
    else:
        print(f"File not found: {filepath}")
        return None

if __name__ == "__main__":
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample"
    valiPath = oriFolderPath + "/HR.xlsx"

    X = getData(oriFolderPath, 5, loadFromFile=True, saveToFile=False)

    # Print the shape of each part
    print(X.shape)

    y = getVali(valiPath)
    print(y.shape)
    
    y_d = getVali(valiPath, 3)
    print(y_d.shape)
    print(y_d[0:18])