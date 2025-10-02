import torch
from train import main
import os

if __name__ == "__main__":
    print("🔄 Resuming training from checkpoint...")
    
    # Check if checkpoint exists
    if not os.path.exists("models/checkpoint.pth"):
        print("❌ No checkpoint found. Starting fresh training...")
    else:
        print("✅ Checkpoint found. Resuming...")
    
    main()