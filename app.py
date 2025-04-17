import os
import sys
import streamlit as st
import numpy as np

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the Python path
sys.path.append(script_dir)

from demo.vjepa_demo import VJEPADemo
from demo.educational import show_educational_content
from demo.technical_details import show_technical_details
from utils.visualization import visualize_masking, visualize_predictions, visualize_embeddings

# Set page configuration
st.set_page_config(
    page_title="V-JEPA Demonstration",
    page_icon="🎬",
    layout="wide"
)

def main():
    st.title("V-JEPA: Video-based Joint-Embedding Predictive Architecture")
    st.subheader("An Interactive Demonstration")

    # Sidebar with explanation and navigation
    with st.sidebar:
        st.header("About V-JEPA")
        st.write("""
        V-JEPA is a self-supervised learning method for learning visual representations from video data.

        It extends the Joint-Embedding Predictive Architecture (JEPA) principle from images to video,
        training a visual encoder by predicting masked spatio-temporal regions of a video within a
        learned representation space, rather than reconstructing pixels.
        """)

        st.header("Key Components")
        st.markdown("""
        - **Context Encoder**: ViT processing masked video
        - **Predictor**: Narrow ViT predicting masked region representations
        - **Target Encoder**: EMA-updated ViT processing unmasked video
        - **3D Multi-block Masking**: ~90% masking ratio
        - **Multi-mask Prediction**: Short-range and long-range masks
        """)

        st.header("Navigation")
        page = st.radio(
            "Select a page:",
            ["Interactive Demo", "Educational Content", "Technical Details"]
        )

    # Display the selected page
    if page == "Interactive Demo":
        show_interactive_demo()
    elif page == "Educational Content":
        show_educational_content()
    elif page == "Technical Details":
        show_technical_details()

def show_interactive_demo():
    """Display the interactive demonstration page"""
    st.header("Interactive Demonstration")

    # Video selection
    video_options = ["Sample Video 1", "Sample Video 2", "Upload your own"]
    selected_video = st.selectbox("Select a video to process", video_options)

    if selected_video == "Upload your own":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            static_dir = os.path.join(script_dir, "static")
            os.makedirs(static_dir, exist_ok=True)
            video_path = os.path.join(static_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.warning("Please upload a video file")
            return
    else:
        # Use a sample video
        static_dir = os.path.join(script_dir, "static")
        video_path = os.path.join(static_dir, f"{selected_video.lower().replace(' ', '_')}.mp4")

    # Initialize the demo
    demo = VJEPADemo()

    # Process parameters
    st.subheader("Masking Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        masking_ratio = st.slider("Masking Ratio (%)", 50, 95, 90)
    with col2:
        block_size = st.slider("Block Size", 1, 10, 4)
    with col3:
        num_frames = st.slider("Number of Frames", 8, 32, 16)

    # Process button
    if st.button("Process Video"):
        with st.spinner("Processing video with V-JEPA..."):
            # Process the video
            frames, masked_frames, predictions, embeddings = demo.process_video(
                video_path, masking_ratio=masking_ratio/100, block_size=block_size, num_frames=num_frames
            )

            # Display results
            st.subheader("Masking Visualization")
            masking_fig = visualize_masking(frames, masked_frames)
            st.pyplot(masking_fig)

            st.subheader("Prediction Results")
            prediction_fig = visualize_predictions(frames, predictions)
            st.pyplot(prediction_fig)

            st.subheader("Learned Representations")

            # Check if we have enough embeddings to visualize
            if embeddings.shape[0] <= 1:
                st.write("Not enough frames to visualize embeddings. Try increasing the number of frames.")
            else:
                # Determine visualization method based on number of samples
                if embeddings.shape[0] < 5:
                    method = "simple 2D projection"
                elif embeddings.shape[0] < 30:
                    method = "PCA"
                else:
                    method = "t-SNE"

                st.write(f"Visualization of the learned embeddings in 2D space using {method}")

                # Create frame labels for visualization
                labels = np.arange(embeddings.shape[0])
                embedding_fig = visualize_embeddings(embeddings, labels)
                st.pyplot(embedding_fig)

            st.success("Processing complete!")

    # Quick explanation
    with st.expander("How does this demonstration work?"):
        st.write("""
        This interactive demonstration shows how V-JEPA processes video data:

        1. **Input**: A video is selected and processed frame by frame
        2. **Masking**: Regions of the video are masked according to the specified parameters
        3. **Processing**: The masked video is processed through the V-JEPA model
        4. **Prediction**: The model predicts representations for the masked regions
        5. **Visualization**: The results are visualized, showing the original frames, masked frames, and predictions

        Note that this is a simplified demonstration. A full V-JEPA model would be trained on millions of video clips
        and would learn more sophisticated representations.

        For more details, check out the Educational Content and Technical Details pages.
        """)

    # Comparison with other approaches
    with st.expander("Comparison with Other Approaches"):
        comparison_data = {
            "Method": ["Pixel Reconstruction", "Contrastive Learning", "V-JEPA"],
            "Focus": ["Low-level details", "Instance discrimination", "Semantic content"],
            "Strengths": [
                "Detailed reconstruction",
                "Good instance-level features",
                "Strong semantic understanding"
            ],
            "Limitations": [
                "Wastes capacity on irrelevant details",
                "Requires careful negative sampling",
                "Computationally intensive"
            ]
        }
        st.table(comparison_data)

if __name__ == "__main__":
    main()
