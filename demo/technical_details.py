import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def show_technical_details():
    """
    Display technical details about V-JEPA implementation
    """
    st.header("Technical Details of V-JEPA")
    
    # Model Architecture
    st.subheader("Model Architecture")
    
    st.markdown("""
    V-JEPA consists of three main components:
    
    1. **Context Encoder**
       - Architecture: Vision Transformer (ViT)
       - Input: Masked video with mask tokens
       - Output: Context embeddings
       - Parameters:
         - Embedding dimension: 768
         - Depth: 12 transformer blocks
         - Number of attention heads: 12
         - MLP ratio: 4.0
    
    2. **Predictor**
       - Architecture: Narrower Vision Transformer
       - Input: Context embeddings
       - Output: Predicted embeddings for masked regions
       - Parameters:
         - Embedding dimension: 768
         - Depth: 4 transformer blocks (shallower than encoder)
         - Number of attention heads: 8
         - MLP ratio: 4.0
    
    3. **Target Encoder**
       - Architecture: Same as Context Encoder
       - Input: Original unmasked video
       - Output: Target embeddings
       - Update mechanism: Exponential Moving Average (EMA) of Context Encoder
    """)
    
    # Masking Strategy
    st.subheader("Masking Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **3D Multi-block Masking**
        
        - Masking ratio: ~90% (much higher than image models)
        - Block size: 4×4×4 (time × height × width)
        - Random block selection
        
        This high masking ratio forces the model to learn meaningful representations
        by making the prediction task challenging.
        """)
    
    with col2:
        st.markdown("""
        **Multi-mask Prediction**
        
        V-JEPA uses two types of masks:
        
        1. **Short-range masks**: Higher density in early frames
           - Focuses on local temporal dynamics
           - Masking ratio: ~70%
        
        2. **Long-range masks**: Higher density in later frames
           - Focuses on global temporal dynamics
           - Masking ratio: ~20%
        """)
    
    # Create a visualization of the masking strategy
    fig = create_masking_visualization()
    st.pyplot(fig)
    
    # Loss Function
    st.subheader("Loss Function")
    
    st.markdown("""
    V-JEPA uses the L1 distance between predicted and target embeddings as its loss function:
    
    $$L = \\frac{1}{|M|} \\sum_{i \\in M} ||f_\\theta(x_i) - g_\\xi(x_i)||_1$$
    
    Where:
    - $M$ is the set of masked patches
    - $f_\\theta$ is the predictor (applied to context embeddings)
    - $g_\\xi$ is the target encoder
    - $x_i$ is the $i$-th patch
    
    The L1 loss is preferred over L2 as it is less sensitive to outliers and produces
    more stable training.
    """)
    
    # Preventing Collapse
    st.subheader("Preventing Representational Collapse")
    
    st.markdown("""
    V-JEPA uses two key techniques to prevent representational collapse:
    
    1. **Stop-gradient on Target Encoder**
       - During backpropagation, gradients are not propagated through the target encoder
       - This prevents the model from finding trivial solutions
       - Mathematically: $\\nabla_\\xi L = 0$
    
    2. **Exponential Moving Average (EMA) Updates**
       - The target encoder parameters are updated as a moving average of the context encoder
       - Update rule: $\\xi \\leftarrow \\tau \\xi + (1 - \\tau) \\theta$
       - Typical value of $\\tau$: 0.996-0.999
       - This ensures stable target representations during training
    """)
    
    # Training Details
    st.subheader("Training Details")
    
    st.markdown("""
    **Pretraining Dataset**
    
    V-JEPA is pretrained on "VideoMix2M", a mix of public video datasets:
    - Kinetics-400
    - Something-Something-v2
    - Ego4D
    - Other public video datasets
    
    Total: ~2 million video clips
    
    **Optimization**
    
    - Optimizer: AdamW
    - Learning rate: 1.5e-4 with cosine decay
    - Weight decay: 0.05
    - Batch size: 2048
    - Training steps: 100,000
    - Hardware: 64 A100 GPUs
    
    **Data Augmentation**
    
    - Random resized crops
    - Random horizontal flips
    - Color jittering
    - Temporal sampling: 16 frames with random stride
    """)
    
    # Evaluation Protocol
    st.subheader("Evaluation Protocol")
    
    st.markdown("""
    V-JEPA is evaluated using "frozen" evaluation, where the pretrained encoder is used
    as a feature extractor without fine-tuning:
    
    1. The pretrained context encoder is used to extract features from video frames
    2. A simple linear classifier is trained on top of these features
    3. Performance is measured on downstream tasks
    
    This approach tests the quality of the learned representations without
    task-specific adaptation.
    
    **Downstream Tasks**
    
    - **Video Classification**: Kinetics-400, Something-Something-v2
    - **Image Classification**: ImageNet
    - **Action Detection**: AVA
    """)
    
    # Ablation Studies
    st.subheader("Ablation Studies")
    
    ablation_data = {
        "Component": [
            "Masking Ratio", 
            "Block Size", 
            "EMA Coefficient", 
            "Predictor Depth",
            "Video vs. Image Pretraining"
        ],
        "Finding": [
            "Higher masking ratios (80-90%) perform better than lower ones",
            "Medium block sizes (4×4×4) outperform very small or large blocks",
            "Higher EMA coefficients (0.996-0.999) provide more stable training",
            "Shallower predictor (4 layers) is more efficient than deeper ones",
            "Video pretraining significantly outperforms image pretraining on motion tasks"
        ]
    }
    
    st.table(ablation_data)
    
    # Implementation Challenges
    st.subheader("Implementation Challenges")
    
    st.markdown("""
    **Memory Efficiency**
    
    Processing video requires significant memory. V-JEPA addresses this through:
    - Gradient checkpointing
    - Mixed precision training
    - Efficient attention implementations
    
    **Computational Cost**
    
    Video models are computationally intensive. V-JEPA reduces this by:
    - Using high masking ratios (90% of content is masked)
    - Employing a narrower predictor network
    - Optimizing the 3D masking implementation
    
    **Stability**
    
    Training self-supervised models can be unstable. V-JEPA ensures stability with:
    - EMA updates for the target encoder
    - Stop-gradient mechanism
    - Careful learning rate scheduling
    """)

def create_masking_visualization():
    """
    Create a visualization of the 3D masking strategy
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)
    
    # Create a sample 3D mask
    T, H, W = 16, 8, 8  # Small dimensions for visualization
    mask = np.zeros((T, H, W), dtype=bool)
    
    # Create blocks of masks
    block_size = 2
    for t in range(0, T, block_size):
        for h in range(0, H, block_size):
            for w in range(0, W, block_size):
                # Randomly mask blocks with 90% probability
                if np.random.rand() < 0.9:
                    t_end = min(t + block_size, T)
                    h_end = min(h + block_size, H)
                    w_end = min(w + block_size, W)
                    mask[t:t_end, h:h_end, w:w_end] = True
    
    # Create short-range and long-range masks
    short_range_mask = np.zeros_like(mask)
    long_range_mask = np.zeros_like(mask)
    
    # Short range focuses on first half
    short_range_mask[:T//2] = mask[:T//2]
    
    # Long range focuses on second half
    long_range_mask[T//2:] = mask[T//2:]
    
    # Visualize a time slice of the mask
    ax1 = fig.add_subplot(gs[0, 0])
    time_slice = 4
    ax1.imshow(mask[time_slice], cmap='gray')
    ax1.set_title(f"Mask at Time t={time_slice}")
    ax1.set_xlabel("Width")
    ax1.set_ylabel("Height")
    
    # Visualize the temporal profile
    ax2 = fig.add_subplot(gs[0, 1])
    temporal_profile = np.mean(mask, axis=(1, 2))
    ax2.plot(temporal_profile, 'b-', label='Overall')
    
    # Add short and long range profiles
    short_profile = np.mean(short_range_mask, axis=(1, 2))
    long_profile = np.mean(long_range_mask, axis=(1, 2))
    ax2.plot(short_profile, 'g--', label='Short-range')
    ax2.plot(long_profile, 'r--', label='Long-range')
    
    ax2.set_title("Temporal Masking Profile")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Proportion Masked")
    ax2.legend()
    
    # Visualize the 3D structure
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    
    # Get coordinates of masked voxels
    x, y, z = np.where(mask)
    
    # Plot masked voxels
    ax3.scatter(z, y, x, c='r', alpha=0.3, marker='s')
    
    ax3.set_title("3D Visualization of Masking")
    ax3.set_xlabel("Width")
    ax3.set_ylabel("Height")
    ax3.set_zlabel("Time")
    
    # Visualize short-range mask
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    x, y, z = np.where(short_range_mask)
    ax4.scatter(z, y, x, c='g', alpha=0.3, marker='s')
    ax4.set_title("Short-range Mask")
    ax4.set_xlabel("Width")
    ax4.set_ylabel("Height")
    ax4.set_zlabel("Time")
    
    # Visualize long-range mask
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    x, y, z = np.where(long_range_mask)
    ax5.scatter(z, y, x, c='r', alpha=0.3, marker='s')
    ax5.set_title("Long-range Mask")
    ax5.set_xlabel("Width")
    ax5.set_ylabel("Height")
    ax5.set_zlabel("Time")
    
    # Visualize masking ratio comparison
    ax6 = fig.add_subplot(gs[1, 2])
    methods = ['MAE', 'VideoMAE', 'V-JEPA']
    masking_ratios = [0.75, 0.8, 0.9]
    
    ax6.bar(methods, masking_ratios, color=['blue', 'green', 'red'])
    ax6.set_title("Masking Ratio Comparison")
    ax6.set_ylabel("Masking Ratio")
    ax6.set_ylim(0, 1.0)
    
    for i, v in enumerate(masking_ratios):
        ax6.text(i, v + 0.02, f"{v:.0%}", ha='center')
    
    plt.tight_layout()
    return fig
