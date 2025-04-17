import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def show_educational_content():
    """
    Display educational content about V-JEPA
    """
    st.header("Understanding V-JEPA")
    
    # Introduction
    st.subheader("Introduction")
    st.write("""
    V-JEPA (Video-based Joint-Embedding Predictive Architecture) is a self-supervised learning method 
    for learning visual representations from video data. It extends the JEPA principle from images to video,
    focusing on predicting representations in a latent space rather than reconstructing pixels.
    
    This approach allows the model to focus on semantic content rather than low-level details,
    resulting in more useful representations for downstream tasks.
    """)
    
    # Key Innovations
    st.subheader("Key Innovations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Latent Prediction vs. Pixel Reconstruction**
        
        Unlike methods that reconstruct pixels, V-JEPA predicts representations in a learned latent space.
        This approach:
        - Ignores unpredictable pixel-level details
        - Focuses on semantic content
        - Avoids wasting capacity on irrelevant details
        """)
        
        st.markdown("""
        **3D Multi-block Masking**
        
        V-JEPA uses a 3D masking strategy with:
        - High masking ratios (~90%)
        - Spatio-temporal blocks
        - Multi-mask prediction (short-range and long-range)
        """)
    
    with col2:
        st.markdown("""
        **Preventing Representational Collapse**
        
        V-JEPA uses two key techniques:
        - Stop-gradient on the target encoder
        - Exponential Moving Average (EMA) updates
        
        These prevent the model from finding trivial solutions and ensure
        meaningful representations are learned.
        """)
        
        st.markdown("""
        **Multi-mask Prediction**
        
        V-JEPA predicts:
        - Short-range masks: focusing on nearby frames
        - Long-range masks: focusing on distant frames
        
        This helps the model learn both local and global temporal dynamics.
        """)
    
    # Architecture Diagram
    st.subheader("V-JEPA Architecture")
    
    # Create a simple architecture diagram
    fig = create_architecture_diagram()
    st.pyplot(fig)
    
    # Training Process
    st.subheader("Training Process")
    st.write("""
    1. **Input Processing**: A video is divided into patches and processed through the model.
    
    2. **Masking**: A 3D mask is applied, hiding ~90% of the spatio-temporal blocks.
    
    3. **Context Encoding**: The masked video is processed by the context encoder, replacing
       masked regions with learnable mask tokens.
    
    4. **Target Encoding**: The original, unmasked video is processed by the target encoder
       to generate target representations.
    
    5. **Prediction**: The predictor takes the context embeddings and predicts representations
       for the masked regions.
    
    6. **Loss Calculation**: The L1 distance between predicted and target representations is
       calculated and minimized.
    
    7. **Parameter Updates**: The context encoder and predictor are updated via gradient descent,
       while the target encoder is updated via EMA.
    """)
    
    # Performance Highlights
    st.subheader("Performance Highlights")
    
    performance_data = {
        "Task": ["Kinetics-400", "Something-Something-v2", "ImageNet", "AVA"],
        "V-JEPA (Top-1)": ["82.1%", "71.2%", "77.9%", "32.1%"],
        "Previous SOTA (Top-1)": ["80.5%", "66.5%", "75.2%", "30.8%"]
    }
    
    st.table(performance_data)
    
    st.write("""
    V-JEPA demonstrates strong "off-the-shelf" performance (frozen evaluation) on various downstream
    image and video tasks without fine-tuning. It significantly outperforms previous state-of-the-art
    video models and large image models on motion-centric tasks.
    """)
    
    # Comparison with Other Approaches
    st.subheader("Comparison with Other Approaches")
    
    comparison_data = {
        "Method": ["Pixel Reconstruction", "Contrastive Learning", "Masked Image Modeling", "V-JEPA"],
        "Examples": ["MAE, SimVLR", "CLIP, SimCLR", "BEiT, MAE", "V-JEPA"],
        "Prediction Target": ["Pixels", "Instance similarity", "Tokens/Pixels", "Latent representations"],
        "Strengths": [
            "Detailed reconstruction", 
            "Good instance-level features", 
            "Strong semantic understanding",
            "Motion and appearance understanding"
        ],
        "Limitations": [
            "Wastes capacity on irrelevant details", 
            "Requires careful negative sampling", 
            "Often focuses on static features",
            "Computationally intensive"
        ]
    }
    
    st.table(comparison_data)
    
    # Future Directions
    st.subheader("Future Directions")
    st.write("""
    The authors of V-JEPA identify several promising directions for future work:
    
    1. **Data Scaling**: Increasing the size and diversity of video pretraining datasets to further
       enhance performance.
    
    2. **Architecture Improvements**: Exploring more efficient architectures for video processing.
    
    3. **Multi-modal Learning**: Extending V-JEPA to incorporate audio and text for more comprehensive
       video understanding.
    
    4. **Temporal Dynamics**: Further improving the model's ability to capture complex motion patterns
       and temporal relationships.
    """)

def create_architecture_diagram():
    """
    Create a diagram illustrating the V-JEPA architecture
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig)
    
    # Input video
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, "Input Video", ha='center', va='center', fontsize=12)
    ax1.axis('off')
    
    # Draw video frames
    for i in range(4):
        rect = plt.Rectangle((0.2 + i*0.2, 0.2), 0.15, 0.15, fill=True, color='lightblue')
        ax1.add_patch(rect)
    
    # Masking
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.8, "3D Masking", ha='center', va='center', fontsize=12)
    ax2.axis('off')
    
    # Draw masked video frames
    for i in range(4):
        rect = plt.Rectangle((0.2 + i*0.2, 0.2), 0.15, 0.15, fill=True, color='lightblue')
        ax2.add_patch(rect)
        
        # Add mask patches
        if i % 2 == 0:
            mask = plt.Rectangle((0.2 + i*0.2, 0.2), 0.15, 0.15, fill=True, color='red', alpha=0.5)
            ax2.add_patch(mask)
    
    # Context Encoder
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(0.5, 0.5, "Context Encoder\n(ViT)", ha='center', va='center', fontsize=12)
    ax3.set_facecolor('lightgreen')
    ax3.axis('off')
    
    # Target Encoder
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.text(0.5, 0.5, "Target Encoder\n(ViT + EMA)", ha='center', va='center', fontsize=12)
    ax4.set_facecolor('lightgreen')
    ax4.axis('off')
    
    # Predictor
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(0.5, 0.5, "Predictor\n(Narrow ViT)", ha='center', va='center', fontsize=12)
    ax5.set_facecolor('lightyellow')
    ax5.axis('off')
    
    # Context Embeddings
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.text(0.5, 0.5, "Context Embeddings", ha='center', va='center', fontsize=12)
    ax6.set_facecolor('lightblue')
    ax6.axis('off')
    
    # Predicted Embeddings
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.text(0.5, 0.5, "Predicted Embeddings", ha='center', va='center', fontsize=12)
    ax7.set_facecolor('lightblue')
    ax7.axis('off')
    
    # Target Embeddings
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.text(0.5, 0.5, "Target Embeddings", ha='center', va='center', fontsize=12)
    ax8.set_facecolor('lightblue')
    ax8.axis('off')
    
    # Loss
    ax9 = fig.add_subplot(gs[0, 2])
    ax9.text(0.5, 0.5, "L1 Loss", ha='center', va='center', fontsize=12)
    ax9.axis('off')
    
    # Add arrows
    plt.annotate("", xy=(0.33, 0.67), xytext=(0.33, 0.77), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.67, 0.67), xytext=(0.67, 0.77), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.5, 0.57), xytext=(0.5, 0.67), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.33, 0.37), xytext=(0.33, 0.47), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.5, 0.37), xytext=(0.5, 0.47), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.67, 0.37), xytext=(0.67, 0.47), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.annotate("", xy=(0.58, 0.37), xytext=(0.42, 0.37), 
                 xycoords='figure fraction', textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="<->", lw=2, color='red'))
    
    plt.annotate("L1 Distance", xy=(0.5, 0.34), 
                 xycoords='figure fraction', ha='center', color='red')
    
    plt.annotate("Stop\nGradient", xy=(0.75, 0.57), 
                 xycoords='figure fraction', ha='center', color='blue')
    
    plt.tight_layout()
    return fig
