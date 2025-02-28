import torch
import numpy as np
import os
from pathlib import Path
from twodCNN import HeartBeatNet
from getData import getData, getVali  # Reuse your data loading functions
import matplotlib.pyplot as plt
import pandas as pd

# Path to test data
test_folder_path = r"/Volumes/T7_Shield/mmwave_ip/Dataset/test_data"  # Replace with your test data path
test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.csv') and not f.startswith('._')]

# Path to saved model
checkpoint_dir = Path("checkpoints")
model_path = checkpoint_dir / "best_model.pt"  # Use your best saved model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

batch_size_set = 4

# Function to load the trained model
def load_model(model_path):
    model = HeartBeatNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    return model

# Function to preprocess test data (same as in training)
def preprocess_data(X_data):
    X_data = np.array(X_data)
    
    # Reshape if needed (depends on your test data structure)
    if len(X_data.shape) == 4:  # If shape is [n_files, n_samples, 60000, 3]
        X_data = X_data.reshape(-1, 60000, 3)
    
    # Transpose to put channels first: [samples, 3, 60000]
    X_data = X_data.transpose(0, 2, 1)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    return X_tensor

# Main inference function
def predict(model, test_data):
    results = []
    
    with torch.no_grad():  # No need to track gradients for inference
        # Process in batches if test data is large
        batch_size = batch_size_set
        num_samples = test_data.shape[0]
        
        for i in range(0, num_samples, batch_size):
            batch = test_data[i:i+batch_size].to(device)
            outputs = model(batch)
            results.append(outputs.cpu().numpy())
    
    # Combine all batch results
    return np.vstack(results) if results else np.array([])

# Function to visualize prediction results
def visualize_results(predictions, ground_truth=None):
    # Create a figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    sample_indices = range(len(predictions))
    resp_rate_pred = predictions[:, 0]
    heart_rate_pred = predictions[:, 1]
    
    # Plot respiratory rate
    ax[0].plot(sample_indices, resp_rate_pred, 'bo-', label='Predicted')
    if ground_truth is not None:
        resp_rate_true = ground_truth[:, 0]
        ax[0].plot(sample_indices, resp_rate_true, 'r--', label='Actual')
    ax[0].set_title('Respiratory Rate')
    ax[0].set_xlabel('Sample Index')
    ax[0].set_ylabel('Rate (breaths/min)')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot heart rate
    ax[1].plot(sample_indices, heart_rate_pred, 'go-', label='Predicted')
    if ground_truth is not None:
        heart_rate_true = ground_truth[:, 1]
        ax[1].plot(sample_indices, heart_rate_true, 'r--', label='Actual')
    ax[1].set_title('Heart Rate')
    ax[1].set_xlabel('Sample Index')
    ax[1].set_ylabel('Rate (beats/min)')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

# Load and prepare test data
def main():
    print("Loading test data...")
    X_test = getData(test_folder_path, test_files)
    
    # If you have ground truth labels and want to evaluate the model
    try:
        y_test = getVali(test_folder_path, test_files)
        has_labels = True
    except:
        has_labels = False
        
    # Preprocess test data
    X_test_processed = preprocess_data(X_test)
    print(f"Test data shape: {X_test_processed.shape}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(model, X_test_processed)
    
    # Display and save results
    print(f"Prediction shape: {predictions.shape}")
    print("Sample predictions (respiratory rate, heart rate):")
    for i, pred in enumerate(predictions[:5]):  # Show first 5 predictions
        print(f"Sample {i+1}: {pred}")
    
    # Save predictions
    np.save("test_predictions.npy", predictions)
    
    # If ground truth labels are available, calculate metrics
    if has_labels:
        y_test = np.array(y_test)
        if len(y_test.shape) > 2:
            y_test = y_test.reshape(-1, 2)
            
        # Calculate MAE for respiratory and heart rates
        rc1_mae = np.mean(np.abs(predictions[:, 0] - y_test[:, 0]))
        hc1_mae = np.mean(np.abs(predictions[:, 1] - y_test[:, 1]))
        
        print("\nTest Set Metrics:")
        print(f"  Respiratory rate MAE: {rc1_mae:.4f}")
        print(f"  Heart rate MAE: {hc1_mae:.4f}")
        visualize_results(predictions, y_test)
    else:
        visualize_results(predictions)
    
    results_df = pd.DataFrame(predictions, columns=['Respiratory_Rate', 'Heart_Rate'])
    results_df.to_csv('test_predictions.csv', index=False)


if __name__ == "__main__":
    main()