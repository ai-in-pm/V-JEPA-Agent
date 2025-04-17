import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
import matplotlib.patches as patches

def visualize_masking(frames, masked_frames, num_frames=4):
    """
    Visualize original frames and their masked versions

    Args:
        frames: Original video frames [T, H, W, C]
        masked_frames: Masked video frames [T, H, W, C]
        num_frames: Number of frames to visualize

    Returns:
        fig: Matplotlib figure
    """
    # Select frames to visualize (evenly spaced)
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames_subset = frames[indices]
        masked_frames_subset = masked_frames[indices]
    else:
        frames_subset = frames
        masked_frames_subset = masked_frames

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, frames_subset.shape[0], figure=fig)

    # Plot original frames
    for i in range(frames_subset.shape[0]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames_subset[i])
        ax.set_title(f"Frame {indices[i] if frames.shape[0] > num_frames else i}")
        ax.axis('off')

    # Plot masked frames
    for i in range(masked_frames_subset.shape[0]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(masked_frames_subset[i])
        ax.set_title(f"Masked Frame {indices[i] if frames.shape[0] > num_frames else i}")
        ax.axis('off')

    plt.tight_layout()
    return fig

def visualize_predictions(frames, predictions, num_frames=4):
    """
    Visualize original frames and their predicted versions

    Args:
        frames: Original video frames [T, H, W, C]
        predictions: Predicted video frames [T, H, W, C]
        num_frames: Number of frames to visualize

    Returns:
        fig: Matplotlib figure
    """
    # Select frames to visualize (evenly spaced)
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames_subset = frames[indices]
        predictions_subset = predictions[indices]
    else:
        frames_subset = frames
        predictions_subset = predictions

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, frames_subset.shape[0], figure=fig)

    # Plot original frames
    for i in range(frames_subset.shape[0]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames_subset[i])
        ax.set_title(f"Original Frame {indices[i] if frames.shape[0] > num_frames else i}")
        ax.axis('off')

    # Plot predicted frames
    for i in range(predictions_subset.shape[0]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(predictions_subset[i])
        ax.set_title(f"Predicted Frame {indices[i] if frames.shape[0] > num_frames else i}")
        ax.axis('off')

    plt.tight_layout()
    return fig

def visualize_embeddings(embeddings, labels=None):
    """
    Visualize embeddings in 2D space using t-SNE or PCA

    Args:
        embeddings: Embeddings to visualize [N, D]
        labels: Optional labels for coloring points

    Returns:
        fig: Matplotlib figure
    """
    # Reshape embeddings if needed
    if isinstance(embeddings, np.ndarray):
        # Already a numpy array
        embeddings_reshaped = embeddings.reshape(embeddings.shape[0], -1)
    else:
        # Convert from tensor if needed
        embeddings_reshaped = embeddings.reshape(embeddings.shape[0], -1).numpy()

    # Get number of samples
    n_samples = embeddings_reshaped.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Choose dimensionality reduction method based on sample size
    if n_samples < 5:
        # Not enough samples for t-SNE or PCA, just use the first 2 dimensions
        print(f"Warning: Only {n_samples} samples, using first 2 dimensions instead of t-SNE")
        if embeddings_reshaped.shape[1] >= 2:
            embeddings_2d = embeddings_reshaped[:, :2]
            method = "First 2 dimensions"
        else:
            # If we don't even have 2 dimensions, create a simple grid
            embeddings_2d = np.zeros((n_samples, 2))
            for i in range(n_samples):
                embeddings_2d[i, 0] = i
                embeddings_2d[i, 1] = 0
            method = "1D arrangement"
    elif n_samples < 30:
        # Use PCA for small sample sizes
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_reshaped)
        method = "PCA"
    else:
        # Use t-SNE for larger sample sizes with appropriate perplexity
        # Perplexity should be smaller than the number of samples
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_reshaped)
        method = f"t-SNE (perplexity={perplexity})"

    # Plot embeddings
    if labels is not None:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='Frame Index')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.8)

    ax.set_title(f'{method} Visualization of Video Embeddings')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # Add frame numbers as annotations
    for i in range(n_samples):
        ax.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.tight_layout()
    return fig

def visualize_attention(attention_maps, frames, num_frames=4, num_heads=4):
    """
    Visualize attention maps overlaid on frames

    Args:
        attention_maps: Attention maps [T, H, W, num_heads]
        frames: Original video frames [T, H, W, C]
        num_frames: Number of frames to visualize
        num_heads: Number of attention heads to visualize

    Returns:
        fig: Matplotlib figure
    """
    # Select frames to visualize (evenly spaced)
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames_subset = frames[indices]
        attention_subset = attention_maps[indices]
    else:
        frames_subset = frames
        attention_subset = attention_maps

    # Select heads to visualize
    head_indices = np.linspace(0, attention_maps.shape[-1] - 1, min(num_heads, attention_maps.shape[-1]), dtype=int)

    # Create figure
    fig = plt.figure(figsize=(12, 3 * min(num_heads, attention_maps.shape[-1])))
    gs = GridSpec(min(num_heads, attention_maps.shape[-1]), frames_subset.shape[0], figure=fig)

    # Plot attention maps
    for h_idx, h in enumerate(head_indices):
        for f_idx in range(frames_subset.shape[0]):
            ax = fig.add_subplot(gs[h_idx, f_idx])

            # Display the frame
            ax.imshow(frames_subset[f_idx])

            # Overlay attention map
            attention = attention_subset[f_idx, :, :, h]
            ax.imshow(attention, alpha=0.5, cmap='hot')

            if f_idx == 0:
                ax.set_ylabel(f"Head {h}")

            if h_idx == 0:
                ax.set_title(f"Frame {indices[f_idx] if frames.shape[0] > num_frames else f_idx}")

            ax.axis('off')

    plt.tight_layout()
    return fig

def visualize_masking_strategy(mask, frames, block_size=4):
    """
    Visualize the 3D masking strategy

    Args:
        mask: Binary mask [T, H, W]
        frames: Original video frames [T, H, W, C]
        block_size: Size of masking blocks

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Plot a sample frame with mask overlay
    ax1 = fig.add_subplot(gs[0, 0])
    frame_idx = mask.shape[0] // 2
    ax1.imshow(frames[frame_idx])

    # Create a red overlay for masked regions
    mask_overlay = np.zeros_like(frames[frame_idx])
    mask_overlay[:, :, 0] = 1.0  # Red channel
    ax1.imshow(np.ma.masked_where(~mask[frame_idx], mask_overlay), alpha=0.5)

    ax1.set_title(f"Frame {frame_idx} with Mask")
    ax1.axis('off')

    # Plot the mask for this frame
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask[frame_idx], cmap='gray')
    ax2.set_title(f"Mask for Frame {frame_idx}")
    ax2.axis('off')

    # Plot the temporal profile of masking
    ax3 = fig.add_subplot(gs[0, 2])
    temporal_profile = np.mean(mask, axis=(1, 2))
    ax3.plot(temporal_profile)
    ax3.set_title("Temporal Masking Profile")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Proportion Masked")

    # Visualize the 3D structure of the mask
    ax4 = fig.add_subplot(gs[1, :], projection='3d')

    # Downsample for visualization
    ds_factor = 4
    ds_mask = mask[::ds_factor, ::ds_factor, ::ds_factor]

    # Get coordinates of masked voxels
    x, y, z = np.where(ds_mask)

    # Scale coordinates
    x = x * ds_factor
    y = y * ds_factor
    z = z * ds_factor

    # Plot masked voxels
    ax4.scatter(z, y, x, c='r', alpha=0.1, marker='s', s=block_size*10)

    ax4.set_title("3D Visualization of Masking Strategy")
    ax4.set_xlabel("Width")
    ax4.set_ylabel("Height")
    ax4.set_zlabel("Time")

    plt.tight_layout()
    return fig
