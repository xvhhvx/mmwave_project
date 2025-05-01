import torch
import numpy as np
from scipy.ndimage import rotate
import random
from torch.utils.data import TensorDataset, ConcatDataset

class MmWaveAugmentation:
    """
    Data augmentation techniques specifically designed for mmWave radar data
    used in heart rate monitoring applications.
    """
    
    @staticmethod
    def add_gaussian_noise(data, noise_level=0.02):
        """Add random Gaussian noise to the data."""
        return data + noise_level * torch.randn_like(data)
    
    @staticmethod
    def temporal_shift(data, max_shift=5):
        """
        Shifts the temporal dimension slightly (simulates slight timing differences).
        For data with shape (batch, timesteps, height, width, channels)
        """
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return data
            
        # Convert to numpy for easier manipulation
        data_np = data.numpy()
        result = np.zeros_like(data_np)
        
        if shift > 0:
            # Shift forward (right)
            result[:, shift:, :, :, :] = data_np[:, :-shift, :, :, :]
            # Repeat first frame to fill the gap
            result[:, :shift, :, :, :] = data_np[:, 0:1, :, :, :].repeat(shift, axis=1)
        else:
            # Shift backward (left)
            shift = abs(shift)
            result[:, :-shift, :, :, :] = data_np[:, shift:, :, :, :]
            # Repeat last frame to fill the gap
            result[:, -shift:, :, :, :] = data_np[:, -1:, :, :, :].repeat(shift, axis=1)
            
        return torch.tensor(result, dtype=torch.float32)
    
    @staticmethod
    def amplitude_scaling(data, scale_range=(0.9, 1.1)):
        """
        Scales the amplitude of the radar signals by a random factor.
        This simulates variations in reflection strength.
        """
        scale_factor = random.uniform(*scale_range)
        return data * scale_factor
    
    @staticmethod
    def small_rotation(data, max_angle=5):
        """
        Apply small rotations to spatial dimensions to simulate
        slightly different radar orientations.
        """
        # Convert to numpy for rotation
        data_np = data.numpy()
        angle = random.uniform(-max_angle, max_angle)
        
        # For each time step, rotate the spatial dimensions
        batch_size, time_steps, height, width, channels = data_np.shape
        rotated = np.zeros_like(data_np)
        
        for b in range(batch_size):
            for t in range(time_steps):
                for c in range(channels):
                    # Rotate each channel independently
                    rotated[b, t, :, :, c] = rotate(
                        data_np[b, t, :, :, c], 
                        angle, 
                        reshape=False, 
                        mode='nearest'
                    )
        
        return torch.tensor(rotated, dtype=torch.float32)
    
    @staticmethod
    def temporal_mask(data, mask_size=10, num_masks=2):
        """
        Apply random masking to temporal segments to improve robustness.
        Similar to SpecAugment technique used in audio processing.
        """
        result = data.clone()
        _, time_steps, _, _, _ = data.shape
        
        for _ in range(num_masks):
            mask_length = random.randint(1, mask_size)
            mask_start = random.randint(0, time_steps - mask_length)
            
            # Apply mask (set to mean value of that sequence)
            for b in range(data.shape[0]):
                mean_val = data[b].mean()
                result[b, mask_start:mask_start+mask_length] = mean_val
                
        return result
    
    @staticmethod
    def augment_batch(X_batch, y_batch, augmentations=None, mix_original=True):
        """
        Apply a combination of augmentations to a batch of data.
        
        Args:
            X_batch: Input features of shape (batch, timesteps, height, width, channels)
            y_batch: Target values of shape (batch,)
            augmentations: List of augmentation functions to apply
            mix_original: Whether to include original data in the output
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if augmentations is None:
            augmentations = [
                MmWaveAugmentation.add_gaussian_noise,
                MmWaveAugmentation.temporal_shift,
                MmWaveAugmentation.amplitude_scaling,
                # Small rotation can be computationally expensive
                # Uncommment if needed and if processing time allows
                # MmWaveAugmentation.small_rotation,
                MmWaveAugmentation.temporal_mask
            ]
        
        augmented_data = []
        augmented_labels = []
        
        # Include original data if specified
        if mix_original:
            augmented_data.append(X_batch)
            augmented_labels.append(y_batch)
        
        # Apply each augmentation
        for aug_func in augmentations:
            aug_X = aug_func(X_batch)
            augmented_data.append(aug_X)
            augmented_labels.append(y_batch)  # Labels remain the same
        
        # Combine all augmented data
        combined_X = torch.cat(augmented_data, dim=0)
        combined_y = torch.cat(augmented_labels, dim=0)
        
        return combined_X, combined_y

def augment_dataset(X_tensor, y_tensor, augment_factor=3, 
                    primary_aug=None, aug_probability=0.5, 
                    always_include_primary=True):
    """
    Augments the entire dataset by applying various transformations.
    
    Args:
        X_tensor: Input features tensor of shape (samples, timesteps, height, width, channels)
        y_tensor: Target values tensor of shape (samples,)
        augment_factor: Target size multiplier for the dataset
        primary_aug: Primary augmentation method to prioritize (e.g., MmWaveAugmentation.add_gaussian_noise)
        aug_probability: Probability of applying additional random augmentations
        always_include_primary: Whether to always include the primary augmentation
        
    Returns:
        Tuple of (augmented_X, augmented_y) with increased sample count
    """
    print(f"Original dataset size: {len(X_tensor)}")
    
    # Calculate how many augmentation rounds we need
    orig_size = len(X_tensor)
    target_size = orig_size * augment_factor
    remaining = target_size - orig_size
    
    # Create datasets to combine
    datasets = [TensorDataset(X_tensor, y_tensor)]
    
    # Add augmented data in smaller batches to manage memory
    batch_size = min(32, orig_size)
    num_batches = (orig_size + batch_size - 1) // batch_size
    
    # Track how many samples we've created
    created_samples = orig_size
    
    # All available augmentations
    available_augs = [
        MmWaveAugmentation.add_gaussian_noise,
        MmWaveAugmentation.temporal_shift,
        MmWaveAugmentation.amplitude_scaling,
        MmWaveAugmentation.temporal_mask
    ]
    
    # Set default primary augmentation if not specified
    if primary_aug is None:
        primary_aug = MmWaveAugmentation.add_gaussian_noise
    
    for i in range(num_batches):
        if created_samples >= target_size:
            break
            
        # Get a batch from the original data
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, orig_size)
        X_batch = X_tensor[start_idx:end_idx]
        y_batch = y_tensor[start_idx:end_idx]
        
        # Initialize augmentations list with primary aug if requested
        selected_augs = [primary_aug] if always_include_primary else []
        
        # Possibly add more random augmentations
        if random.random() < aug_probability:
            # Create a list of augmentations excluding the primary one
            other_augs = [aug for aug in available_augs if aug != primary_aug]
            
            # Randomly select 1-2 additional augmentations
            num_additional = min(random.randint(1, 2), len(other_augs))
            additional_augs = random.sample(other_augs, num_additional)
            
            # Add these to our selection
            selected_augs.extend(additional_augs)
        
        # If no augmentations were selected (can happen if primary not always included and probability fails),
        # at least use the primary augmentation
        if not selected_augs:
            selected_augs = [primary_aug]
        
        # Apply the selected augmentations
        aug_X, aug_y = MmWaveAugmentation.augment_batch(
            X_batch, y_batch, 
            augmentations=selected_augs, 
            mix_original=False
        )
        
        # Add to our collection
        datasets.append(TensorDataset(aug_X, aug_y))
        created_samples += len(aug_X)
        
        # Print progress
        if (i + 1) % 5 == 0 or i == num_batches - 1:
            print(f"Augmentation progress: {created_samples}/{target_size} samples created")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    # Extract the tensors from the combined dataset
    all_X = []
    all_y = []
    for X, y in combined_dataset:
        all_X.append(X.unsqueeze(0))  # Add batch dimension
        all_y.append(y.unsqueeze(0))
    
    augmented_X = torch.cat(all_X, dim=0)
    augmented_y = torch.cat(all_y, dim=0)
    
    print(f"Final augmented dataset size: {len(augmented_X)}")
    return augmented_X, augmented_y

# Usage example
if __name__ == "__main__":
    # Test with a small dummy dataset
    dummy_X = torch.randn(10, 1200, 8, 8, 2)  # 10 samples with shape matching your data
    dummy_y = torch.randn(10)  # 10 target values
    
    # Apply augmentation
    aug_X, aug_y = augment_dataset(dummy_X, dummy_y, augment_factor=3)
    
    print(f"Original shape: {dummy_X.shape}, Augmented shape: {aug_X.shape}")
    print(f"Original targets: {dummy_y.shape}, Augmented targets: {aug_y.shape}")