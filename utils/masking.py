import numpy as np

def create_3d_mask(time_dim, height, width, masking_ratio=0.9, block_size=4, seed=None):
    """
    Create a 3D binary mask for spatio-temporal masking
    
    Args:
        time_dim: Number of frames (time dimension)
        height: Height of each frame
        width: Width of each frame
        masking_ratio: Ratio of blocks to mask (0.0-1.0)
        block_size: Size of masking blocks
        seed: Random seed for reproducibility
        
    Returns:
        mask: Binary mask of shape [time_dim, height, width] where 1 indicates masked regions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of blocks in each dimension
    t_blocks = time_dim // block_size + (1 if time_dim % block_size != 0 else 0)
    h_blocks = height // block_size + (1 if height % block_size != 0 else 0)
    w_blocks = width // block_size + (1 if width % block_size != 0 else 0)
    
    # Calculate total number of blocks and number to mask
    total_blocks = t_blocks * h_blocks * w_blocks
    num_masked_blocks = int(total_blocks * masking_ratio)
    
    # Create block mask (1 = masked, 0 = visible)
    block_mask = np.zeros((t_blocks, h_blocks, w_blocks), dtype=np.bool_)
    
    # Randomly select blocks to mask
    indices = np.random.choice(
        total_blocks, 
        size=num_masked_blocks, 
        replace=False
    )
    
    # Convert flat indices to 3D coordinates
    for idx in indices:
        t = idx // (h_blocks * w_blocks)
        hw = idx % (h_blocks * w_blocks)
        h = hw // w_blocks
        w = hw % w_blocks
        
        if t < t_blocks and h < h_blocks and w < w_blocks:
            block_mask[t, h, w] = True
    
    # Expand block mask to pixel mask
    mask = np.zeros((time_dim, height, width), dtype=np.bool_)
    
    for t in range(time_dim):
        t_block = min(t // block_size, t_blocks - 1)
        for h in range(height):
            h_block = min(h // block_size, h_blocks - 1)
            for w in range(width):
                w_block = min(w // block_size, w_blocks - 1)
                mask[t, h, w] = block_mask[t_block, h_block, w_block]
    
    return mask

def create_multi_mask(time_dim, height, width, short_range_ratio=0.7, long_range_ratio=0.2, 
                     block_size=4, seed=None):
    """
    Create multiple masks for different temporal ranges
    
    Args:
        time_dim: Number of frames (time dimension)
        height: Height of each frame
        width: Width of each frame
        short_range_ratio: Masking ratio for short-range prediction
        long_range_ratio: Masking ratio for long-range prediction
        block_size: Size of masking blocks
        seed: Random seed for reproducibility
        
    Returns:
        short_range_mask: Mask for short-range prediction
        long_range_mask: Mask for long-range prediction
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create short-range mask (higher density in early frames)
    short_range_mask = np.zeros((time_dim, height, width), dtype=np.bool_)
    
    # Short range focuses on first half of the video
    short_range_time = time_dim // 2
    short_range_mask[:short_range_time] = create_3d_mask(
        short_range_time, height, width, 
        masking_ratio=short_range_ratio, 
        block_size=block_size,
        seed=seed if seed is None else seed + 1
    )
    
    # Create long-range mask (higher density in later frames)
    long_range_mask = np.zeros((time_dim, height, width), dtype=np.bool_)
    
    # Long range focuses on second half of the video
    long_range_time = time_dim - (time_dim // 2)
    long_range_mask[time_dim // 2:] = create_3d_mask(
        long_range_time, height, width, 
        masking_ratio=long_range_ratio, 
        block_size=block_size,
        seed=seed if seed is None else seed + 2
    )
    
    return short_range_mask, long_range_mask
