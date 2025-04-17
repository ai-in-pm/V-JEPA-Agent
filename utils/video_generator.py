import numpy as np
import cv2
import os

def generate_sample_videos(output_dir="VJEPA-Agent/static", num_frames=30, fps=10):
    """
    Generate sample videos for demonstration
    
    Args:
        output_dir: Directory to save videos
        num_frames: Number of frames in each video
        fps: Frames per second
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample video 1: Moving circle
    generate_moving_circle_video(
        os.path.join(output_dir, "sample_video_1.mp4"),
        num_frames=num_frames,
        fps=fps
    )
    
    # Generate sample video 2: Moving shapes
    generate_moving_shapes_video(
        os.path.join(output_dir, "sample_video_2.mp4"),
        num_frames=num_frames,
        fps=fps
    )
    
    print(f"Sample videos generated in {output_dir}")

def generate_moving_circle_video(output_path, num_frames=30, fps=10, width=320, height=240):
    """
    Generate a video with a moving circle
    
    Args:
        output_path: Path to save the video
        num_frames: Number of frames
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(num_frames):
        # Create a blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Calculate circle position (moving in a circle)
        center_x = int(width/2 + width/3 * np.cos(2 * np.pi * i / num_frames))
        center_y = int(height/2 + height/3 * np.sin(2 * np.pi * i / num_frames))
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), -1)  # Red circle
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()

def generate_moving_shapes_video(output_path, num_frames=30, fps=10, width=320, height=240):
    """
    Generate a video with multiple moving shapes
    
    Args:
        output_path: Path to save the video
        num_frames: Number of frames
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize shape positions and velocities
    shapes = [
        {
            'type': 'circle',
            'position': [width/4, height/4],
            'velocity': [2, 3],
            'color': (0, 0, 255),  # Red
            'size': 20
        },
        {
            'type': 'rectangle',
            'position': [width*3/4, height/4],
            'velocity': [-2, 2],
            'color': (0, 255, 0),  # Green
            'size': 30
        },
        {
            'type': 'triangle',
            'position': [width/2, height*3/4],
            'velocity': [3, -2],
            'color': (255, 0, 0),  # Blue
            'size': 25
        }
    ]
    
    # Generate frames
    for i in range(num_frames):
        # Create a blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Update and draw shapes
        for shape in shapes:
            # Update position
            shape['position'][0] += shape['velocity'][0]
            shape['position'][1] += shape['velocity'][1]
            
            # Bounce off walls
            if shape['position'][0] < 0 or shape['position'][0] > width:
                shape['velocity'][0] *= -1
            if shape['position'][1] < 0 or shape['position'][1] > height:
                shape['velocity'][1] *= -1
            
            # Draw shape
            if shape['type'] == 'circle':
                cv2.circle(frame, 
                           (int(shape['position'][0]), int(shape['position'][1])), 
                           shape['size'], 
                           shape['color'], 
                           -1)
            elif shape['type'] == 'rectangle':
                x = int(shape['position'][0] - shape['size']/2)
                y = int(shape['position'][1] - shape['size']/2)
                cv2.rectangle(frame, 
                              (x, y), 
                              (x + shape['size'], y + shape['size']), 
                              shape['color'], 
                              -1)
            elif shape['type'] == 'triangle':
                pts = np.array([
                    [int(shape['position'][0]), int(shape['position'][1] - shape['size'])],
                    [int(shape['position'][0] - shape['size']), int(shape['position'][1] + shape['size'])],
                    [int(shape['position'][0] + shape['size']), int(shape['position'][1] + shape['size'])]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(frame, [pts], shape['color'])
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()

if __name__ == "__main__":
    generate_sample_videos()
