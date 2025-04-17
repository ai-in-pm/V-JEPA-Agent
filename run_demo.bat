@echo off
echo ======================================================
echo V-JEPA Demonstration - PhD-level AI Implementation
echo ======================================================
echo.
echo This demonstration will show how V-JEPA works for learning
echo visual representations from video data.
echo.
echo Setting up the environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import torch, streamlit, cv2, numpy, matplotlib, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Some required packages are missing. Installing them now...
    pip install torch torchvision streamlit opencv-python numpy matplotlib scikit-learn
    if %errorlevel% neq 0 (
        echo Error installing packages. Please try manually:
        echo pip install torch torchvision streamlit opencv-python numpy matplotlib scikit-learn
        pause
        exit /b 1
    )
)

echo Starting V-JEPA Demonstration...
echo (This may take a moment to initialize)
echo.
python run_demo.py --debug

if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running the demonstration.
    echo Please check the error message above.
    pause
)

