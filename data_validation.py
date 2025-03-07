import torch
import numpy as np

def check_tensor_valid(tensor, tensor_name="Tensor"):
    """
    Check if a tensor contains NaN or Inf values
    
    Args:
        tensor: PyTorch tensor to check
        tensor_name: Name of the tensor for logging
        
    Returns:
        bool: True if tensor is valid (no NaN/Inf), False otherwise
    """
    if torch.isnan(tensor).any():
        print(f"ERROR: NaN values found in {tensor_name}")
        return False
    if torch.isinf(tensor).any():
        print(f"ERROR: Inf values found in {tensor_name}")
        return False
    return True

def print_tensor_stats(tensor, tensor_name="Tensor"):
    """
    Print statistics about a tensor to help with debugging
    
    Args:
        tensor: PyTorch tensor
        tensor_name: Name of the tensor for logging
    """
    if tensor.numel() == 0:
        print(f"{tensor_name} is empty")
        return
        
    try:
        print(f"{tensor_name} stats:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Min: {tensor.min().item()}")
        print(f"  Max: {tensor.max().item()}")
        print(f"  Mean: {tensor.mean().item()}")
        print(f"  Std: {tensor.std().item()}")
    except Exception as e:
        print(f"Error computing stats for {tensor_name}: {e}")

def safe_normalize(tensor, mean=None, std=None, eps=1e-8, dim=0):
    """
    Safely normalize a tensor with protection against division by zero
    
    Args:
        tensor: PyTorch tensor to normalize
        mean: Mean value (if None, computed from tensor)
        std: Standard deviation (if None, computed from tensor)
        eps: Small value to add to std to prevent division by zero
        dim: Dimension along which to compute statistics
        
    Returns:
        normalized_tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    """
    if mean is None:
        mean = tensor.mean(dim=dim, keepdim=True)
    if std is None:
        std = tensor.std(dim=dim, keepdim=True)
    
    # Check for very small std values and replace with eps
    std = torch.clamp(std, min=eps)
    
    # Normalize
    normalized_tensor = (tensor - mean) / std
    
    # Clip extreme values to prevent outliers
    normalized_tensor = torch.clamp(normalized_tensor, min=-10.0, max=10.0)
    
    return normalized_tensor, mean, std

def validate_data_pipeline(X_tensor, y_tensor):
    """
    Validate the data pipeline by checking for issues in input and target tensors
    
    Args:
        X_tensor: Input tensor
        y_tensor: Target tensor
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check shapes
    print(f"Input tensor shape: {X_tensor.shape}")
    print(f"Target tensor shape: {y_tensor.shape}")
    
    # Check for NaN/Inf values
    if not check_tensor_valid(X_tensor, "Input tensor"):
        return False
    if not check_tensor_valid(y_tensor, "Target tensor"):
        return False
    
    # Print statistics
    print_tensor_stats(X_tensor, "Input tensor")
    print_tensor_stats(y_tensor, "Target tensor")
    
    # Check for zero variance features
    if torch.any(X_tensor.std(dim=0) < 1e-10):
        print("WARNING: Some features have near-zero variance")
    
    return True

def check_gradients(model, threshold=1.0):
    """
    Check model gradients for NaN/Inf values and large magnitudes
    
    Args:
        model: PyTorch model
        threshold: Threshold for gradient magnitude warning
        
    Returns:
        bool: True if gradients are valid, False otherwise
    """
    valid = True
    max_grad = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                valid = False
            if torch.isinf(param.grad).any():
                print(f"Inf gradient detected in {name}")
                valid = False
            
            # Check magnitude
            grad_max = param.grad.abs().max().item()
            max_grad = max(max_grad, grad_max)
            if grad_max > threshold:
                print(f"Large gradient in {name}: {grad_max}")
    
    print(f"Maximum gradient magnitude: {max_grad}")
    return valid

def check_loss_value(loss_value, step_name=""):
    """
    Check if a loss value is valid (not NaN or Inf)
    
    Args:
        loss_value: Loss value to check
        step_name: Name of the step for logging
        
    Returns:
        bool: True if loss is valid, False otherwise
    """
    if np.isnan(loss_value):
        print(f"WARNING: NaN loss detected at {step_name}")
        return False
    if np.isinf(loss_value):
        print(f"WARNING: Inf loss detected at {step_name}")
        return False
    return True
