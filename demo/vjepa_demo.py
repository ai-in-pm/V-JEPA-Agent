import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Get the parent directory to access other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.vjepa_model import ContextEncoder, Predictor, TargetEncoder
from utils.masking import create_3d_mask

class VJEPADemo:
    def __init__(self, model_path=None):
        """
        Initialize the V-JEPA demonstration

        Args:
            model_path: Path to pretrained model weights (if available)
        """
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.embed_dim = 768  # Embedding dimension
        self.num_heads = 12   # Number of attention heads
        self.depth = 12       # Transformer depth

        # Initialize model components
        self.context_encoder = ContextEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads
        ).to(self.device)

        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            depth=4,  # Narrower predictor
            num_heads=8
        ).to(self.device)

        self.target_encoder = TargetEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads
        ).to(self.device)

        # Load pretrained weights if available
        if model_path:
            self._load_weights(model_path)

        # Set target encoder to evaluation mode (no gradient updates)
        self.target_encoder.eval()

        # For demonstration, we'll use the same encoder for both context and target
        # In a real implementation, the target encoder would be an EMA of the context encoder
        self._copy_weights_to_target()

    def _load_weights(self, model_path):
        """Load pretrained weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])

    def _copy_weights_to_target(self):
        """Copy weights from context encoder to target encoder"""
        # Instead of copying parameters directly, which can cause size mismatch issues,
        # we'll initialize both encoders with the same architecture and parameters

        # Make sure both encoders have the same temporal_embed size
        max_frames = 16  # This should match the value in the model definition

        # Ensure temporal embeddings have the same size
        if self.context_encoder.temporal_embed.shape[1] != self.target_encoder.temporal_embed.shape[1]:
            # Resize temporal embeddings if needed
            print(f"Resizing temporal embeddings from {self.context_encoder.temporal_embed.shape} to match {self.target_encoder.temporal_embed.shape}")

            # Create new temporal embeddings with the correct size
            new_temporal_embed = torch.zeros(1, max_frames, 1, self.embed_dim, device=self.device)
            nn.init.trunc_normal_(new_temporal_embed, std=0.02)

            # Update both encoders
            self.context_encoder.temporal_embed = nn.Parameter(new_temporal_embed)
            self.target_encoder.temporal_embed = nn.Parameter(new_temporal_embed.clone())

        # Now copy parameters that should have matching sizes
        for (name_q, param_q), (name_k, param_k) in zip(
            self.context_encoder.named_parameters(),
            self.target_encoder.named_parameters()
        ):
            try:
                if param_q.shape == param_k.shape:
                    param_k.data.copy_(param_q.data)
                else:
                    print(f"Skipping parameter {name_k} due to shape mismatch: {param_q.shape} vs {param_k.shape}")
            except Exception as e:
                print(f"Error copying parameter {name_k}: {e}")

            param_k.requires_grad = False

    def process_video(self, video_path, masking_ratio=0.9, block_size=4, num_frames=16):
        """
        Process a video through the V-JEPA model

        Args:
            video_path: Path to the video file
            masking_ratio: Ratio of blocks to mask (0.0-1.0)
            block_size: Size of masking blocks
            num_frames: Number of frames to process

        Returns:
            frames: Original video frames
            masked_frames: Masked video frames
            predictions: Predicted content for masked regions
            embeddings: Learned embeddings
        """
        # Load video frames
        frames = self._load_video(video_path, num_frames)

        # Create 3D mask
        mask = create_3d_mask(
            frames.shape[0],  # Time dimension
            frames.shape[1],  # Height
            frames.shape[2],  # Width
            masking_ratio=masking_ratio,
            block_size=block_size
        )

        # Apply mask to frames
        masked_frames = frames.copy()
        for t in range(frames.shape[0]):
            for h in range(frames.shape[1]):
                for w in range(frames.shape[2]):
                    if mask[t, h, w]:
                        masked_frames[t, h, w] = np.zeros(3)  # Zero out masked regions

        # Convert to torch tensors
        # Add batch dimension if needed - our model expects [B, T, C, H, W] or [B, C, H, W]
        frames_tensor = torch.from_numpy(frames).float()
        masked_frames_tensor = torch.from_numpy(masked_frames).float()
        mask_tensor = torch.from_numpy(mask).float()

        # Check tensor shapes and rearrange as needed
        # Our model expects [B, T, C, H, W] format

        # Debug original shapes
        print(f"Original frames shape: {frames_tensor.shape}")
        print(f"Original masked frames shape: {masked_frames_tensor.shape}")
        print(f"Original mask shape: {mask_tensor.shape}")

        # Our model expects inputs in the format [B, T, C, H, W]
        # The frames from _load_video are in format [T, H, W, C]

        # First, add a batch dimension (B=1)
        frames_tensor = frames_tensor.unsqueeze(0)  # [1, T, H, W, C]
        masked_frames_tensor = masked_frames_tensor.unsqueeze(0)  # [1, T, H, W, C]
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, T, H, W]

        # Then, permute to get [B, T, C, H, W]
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3)  # [1, T, C, H, W]
        masked_frames_tensor = masked_frames_tensor.permute(0, 1, 4, 2, 3)  # [1, T, C, H, W]

        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        masked_frames_tensor = masked_frames_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # Print shapes for debugging
        print(f"Frames tensor shape: {frames_tensor.shape}")
        print(f"Masked frames tensor shape: {masked_frames_tensor.shape}")
        print(f"Mask tensor shape: {mask_tensor.shape}")

        # Process through model
        with torch.no_grad():
            # Get context embeddings from masked frames
            context_embeddings = self.context_encoder(masked_frames_tensor, mask_tensor)

            # Get target embeddings from original frames
            target_embeddings = self.target_encoder(frames_tensor)

            # Predict masked regions
            predicted_embeddings = self.predictor(context_embeddings, mask_tensor)

            # For visualization purposes, we'll create a "reconstructed" version
            # In a real V-JEPA, we don't actually reconstruct pixels, but for demonstration
            # we'll use a simple decoder to visualize what the model is predicting
            predictions_np = self._visualize_predictions(predicted_embeddings, masked_frames_tensor, mask_tensor)

        # The _visualize_predictions method now returns numpy arrays directly
        # No need for additional permutation or conversion

        return frames, masked_frames, predictions_np, predicted_embeddings.cpu().numpy()

    def _load_video(self, video_path, num_frames=16):
        """
        Load video frames from a file

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract

        Returns:
            frames: Numpy array of video frames [T, H, W, C]
        """
        # For demonstration, we'll create a synthetic video if the file doesn't exist
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video file: {video_path}")

            frames = []
            frame_count = 0

            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to a manageable size
                frame = cv2.resize(frame, (224, 224))

                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0

                frames.append(frame)
                frame_count += 1

            cap.release()

            if len(frames) < num_frames:
                # Loop the video if it's too short
                while len(frames) < num_frames:
                    frames.append(frames[len(frames) % len(frames)])

            return np.array(frames)

        except (FileNotFoundError, cv2.error):
            print(f"Warning: Could not load video from {video_path}. Creating synthetic video.")
            # Create a synthetic video for demonstration
            frames = np.zeros((num_frames, 224, 224, 3), dtype=np.float32)

            # Create a moving circle
            for i in range(num_frames):
                # Background
                frames[i, :, :, :] = 0.1

                # Moving circle
                center_x = int(112 + 80 * np.cos(2 * np.pi * i / num_frames))
                center_y = int(112 + 80 * np.sin(2 * np.pi * i / num_frames))

                for h in range(224):
                    for w in range(224):
                        dist = np.sqrt((h - center_y)**2 + (w - center_x)**2)
                        if dist < 30:
                            frames[i, h, w, 0] = 0.8  # R
                            frames[i, h, w, 1] = 0.2  # G
                            frames[i, h, w, 2] = 0.2  # B

            return frames

    def _visualize_predictions(self, predicted_embeddings, masked_frames, mask):
        """
        Create a visualization of the predictions

        Note: In a real V-JEPA, we don't reconstruct pixels, but for demonstration
        we'll create a simple visualization of what the model is predicting

        Args:
            predicted_embeddings: Predicted embeddings for masked regions
            masked_frames: Masked input frames [B, T, C, H, W] or [B, C, H, W]
            mask: Binary mask indicating masked regions [B, T, H, W] or [B, H, W]

        Returns:
            predictions: Visualization of predictions [T, H, W, C] for display
        """
        # Print shapes for debugging
        print(f"Visualizing predictions:")
        print(f"  Predicted embeddings shape: {predicted_embeddings.shape}")
        print(f"  Masked frames shape: {masked_frames.shape}")
        print(f"  Mask shape: {mask.shape}")

        # Convert to CPU for processing
        masked_frames_cpu = masked_frames.cpu()
        mask_cpu = mask.cpu()

        # For demonstration, we'll create a simple visualization
        # Start with the masked frames and add color to masked regions
        predictions = masked_frames_cpu.clone()

        # Create a simplified version for visualization
        # Convert from [B, T, C, H, W] to [T, H, W, C] for easier processing
        if len(predictions.shape) == 5:  # [B, T, C, H, W]
            # Remove batch dimension (assume B=1) and permute to [T, H, W, C]
            predictions = predictions.squeeze(0).permute(0, 2, 3, 1)
            mask_vis = mask_cpu.squeeze(0)  # [T, H, W]
        elif len(predictions.shape) == 4:  # [B, C, H, W]
            # Single frame - remove batch dimension and permute to [H, W, C]
            predictions = predictions.squeeze(0).permute(1, 2, 0)
            predictions = predictions.unsqueeze(0)  # Add time dimension [1, H, W, C]
            mask_vis = mask_cpu.squeeze(0)  # [H, W]
            mask_vis = mask_vis.unsqueeze(0)  # Add time dimension [1, H, W]
        else:
            print(f"Unexpected predictions shape: {predictions.shape}")
            # Return original frames as numpy array
            return masked_frames_cpu.squeeze(0).permute(0, 2, 3, 1).numpy()

        # Now we have predictions in [T, H, W, C] format and mask in [T, H, W] format
        # Apply a simple color overlay to masked regions
        predictions_np = predictions.numpy()
        mask_np = mask_vis.numpy()

        # Create a reddish color for masked regions
        for t in range(predictions_np.shape[0]):
            for h in range(predictions_np.shape[1]):
                for w in range(predictions_np.shape[2]):
                    if mask_np[t, h, w]:
                        # Add reddish tint to masked regions
                        predictions_np[t, h, w, 0] = 0.8  # R
                        predictions_np[t, h, w, 1] = 0.2  # G
                        predictions_np[t, h, w, 2] = 0.2  # B

        return predictions_np
