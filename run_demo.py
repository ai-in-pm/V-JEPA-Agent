import os
import sys
import subprocess
import argparse
import traceback

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and script directory to the Python path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, script_dir)

# Print current working directory and Python path for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Python path: {sys.path}")

try:
    from utils.video_generator import generate_sample_videos
except ImportError as e:
    print(f"Error importing video_generator: {e}")
    print("Attempting to import with absolute path...")
    sys.path.insert(0, os.path.join(script_dir, 'utils'))
    try:
        from video_generator import generate_sample_videos
    except ImportError as e2:
        print(f"Still failed to import: {e2}")
        print("Available files in utils directory:")
        utils_dir = os.path.join(script_dir, 'utils')
        if os.path.exists(utils_dir):
            print(os.listdir(utils_dir))
        else:
            print(f"Utils directory not found at {utils_dir}")
        sys.exit(1)

def main():
    try:
        parser = argparse.ArgumentParser(description="Run V-JEPA Demonstration")
        parser.add_argument("--generate-videos", action="store_true", help="Generate sample videos")
        parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
        parser.add_argument("--debug", action="store_true", help="Run in debug mode with extra logging")
        args = parser.parse_args()

        # Set up paths relative to the script directory
        static_dir = os.path.join(script_dir, "static")
        app_path = os.path.join(script_dir, "app.py")

        if args.debug:
            print(f"Static directory: {static_dir}")
            print(f"App path: {app_path}")

        # Create static directory if it doesn't exist
        os.makedirs(static_dir, exist_ok=True)

        # Generate sample videos if requested
        if args.generate_videos:
            print("Generating sample videos...")
            generate_sample_videos(output_dir=static_dir)

        # Check if sample videos exist, generate them if not
        sample_video_1 = os.path.join(static_dir, "sample_video_1.mp4")
        sample_video_2 = os.path.join(static_dir, "sample_video_2.mp4")

        if not os.path.exists(sample_video_1) or not os.path.exists(sample_video_2):
            print("Sample videos not found. Generating them...")
            generate_sample_videos(output_dir=static_dir)

        # Verify app.py exists
        if not os.path.exists(app_path):
            print(f"Error: App file not found at {app_path}")
            print("Files in script directory:")
            print(os.listdir(script_dir))
            sys.exit(1)

        # Run the Streamlit app
        print(f"Starting V-JEPA demonstration on port {args.port}...")
        print(f"Running: streamlit run {app_path} --server.port {args.port}")

        # Use full path to streamlit if needed
        streamlit_cmd = "streamlit"

        subprocess.run([
            streamlit_cmd, "run",
            app_path,
            "--server.port", str(args.port)
        ])
    except Exception as e:
        print(f"Error running demonstration: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
