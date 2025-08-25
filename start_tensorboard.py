#!/usr/bin/env python3
"""
Simple script to start TensorBoard for your EEG meta-learning project.
Run this after you've started training to visualize your results.
"""

import os
import subprocess
import sys
from pathlib import Path

def start_tensorboard(log_dir="results/tensorboard_logs", port=6006):
    """
    Start TensorBoard pointing to the specified log directory.
    
    Parameters:
    -----------
    log_dir : str
        Path to the directory containing TensorBoard logs
    port : int
        Port number for TensorBoard to run on
    """
    
    # Convert to absolute path
    log_dir = Path(log_dir).resolve()
    
    if not log_dir.exists():
        print(f"Warning: Log directory {log_dir} does not exist yet.")
        print("Make sure to run your training script first to generate logs.")
        print(f"Creating directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting TensorBoard on port {port}")
    print(f"Log directory: {log_dir}")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        # Start TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(log_dir),
            "--port", str(port),
            "--host", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
        print("Make sure tensorboard is installed: pip install tensorboard")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start TensorBoard for EEG meta-learning project")
    parser.add_argument("--log_dir", default="results/tensorboard_logs", 
                       help="Directory containing TensorBoard logs")
    parser.add_argument("--port", type=int, default=6006,
                       help="Port number for TensorBoard")
    
    args = parser.parse_args()
    start_tensorboard(args.log_dir, args.port)
