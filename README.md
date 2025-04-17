# V-JEPA Demonstration

This project provides an interactive demonstration of V-JEPA (Video-based Joint-Embedding Predictive Architecture), a PhD-level Artificial Machine Intelligence (AMI) that showcases how V-JEPA works for learning visual representations from video data.

![VJEPA version3](https://github.com/user-attachments/assets/c71e6e2b-6aa5-4dc7-81e4-c809f146b87e)
![VJEPA version2](https://github.com/user-attachments/assets/7f922e7c-6d79-441b-8993-9d44e34ecac1)
![VJEPA TechDetails](https://github.com/user-attachments/assets/a8f9705d-e67b-4f8f-bc05-fe946fcac4e4)
![VJEPA EduDetails](https://github.com/user-attachments/assets/677a2150-1703-4164-9e25-698b18f6b933)


## Overview

V-JEPA extends the Joint-Embedding Predictive Architecture (JEPA) principle from images to video, training a visual encoder by predicting masked spatio-temporal regions of a video within a learned representation space, rather than reconstructing pixels. This latent prediction strategy aims to capture more semantic information by ignoring unpredictable pixel-level details.

This demonstration provides:
1. An interactive interface to explore V-JEPA's masking and prediction capabilities
2. Educational content explaining the key concepts and innovations
3. Technical details about the implementation and architecture

## Key Components

- **Context Encoder**: ViT processing masked video
- **Predictor**: Narrow ViT predicting masked region representations
- **Target Encoder**: EMA-updated ViT processing unmasked video
- **3D Multi-block Masking**: ~90% masking ratio
- **Multi-mask Prediction**: Short-range and long-range masks

## Features of this Demonstration

1. **Interactive Demo**:
   - Process sample videos or upload your own
   - Adjust masking parameters and see the results
   - Visualize the masking, predictions, and learned representations

2. **Educational Content**:
   - Detailed explanations of how V-JEPA works
   - Visualizations of the architecture and masking strategy
   - Comparisons with other self-supervised learning approaches

3. **Technical Details**:
   - In-depth information about the model architecture
   - Explanations of the masking strategy and loss function
   - Implementation challenges and solutions

## Installation

```bash
# Clone the repository (if applicable)
# git clone https://github.com/your-username/vjepa-demo.git
# cd vjepa-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the demonstration:

```bash
python run_demo.py
```

This will:
1. Generate sample videos if they don't exist
2. Start the Streamlit app
3. Open the demonstration in your browser

Additional options:

```bash
# Force regeneration of sample videos
python run_demo.py --generate-videos

# Run on a specific port
python run_demo.py --port 8502
```

## Project Structure

```
VJEPA-Agent/
├── app.py                 # Main Streamlit application
├── run_demo.py            # Script to run the demonstration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── EXPLANATION.md         # Detailed explanation of V-JEPA
├── demo/                  # Demo components
│   ├── vjepa_demo.py      # Core demonstration logic
│   ├── educational.py     # Educational content
│   └── technical_details.py # Technical details
├── models/                # Model architecture
│   └── vjepa_model.py     # V-JEPA model implementation
├── utils/                 # Utility functions
│   ├── masking.py         # Masking strategies
│   ├── visualization.py   # Visualization utilities
│   └── video_generator.py # Sample video generator
└── static/                # Static files (sample videos)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Additional Resources

For a more detailed explanation of V-JEPA, see the `EXPLANATION.md` file in this repository.

## References

- [V-JEPA: Video Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2310.00708)
- [I-JEPA: Image Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
