#!/usr/bin/env python3
"""
Script to download CosyVoice models using modelscope
"""

import os
from modelscope import snapshot_download

def main():
    print("Starting model download...")
    
    # Create pretrained_models directory if it doesn't exist
    os.makedirs('pretrained_models', exist_ok=True)
    
    # Download CosyVoice2 0.5B model
    print("Downloading CosyVoice2 0.5B model...")
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
    
    # Download text frontend (optional but recommended)
    print("Downloading text frontend...")
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
    
    print("Model download completed!")

if __name__ == "__main__":
    main() 