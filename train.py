#!/usr/bin/env python3
"""
train.py - Training Script
Train the voice description encoder with voice descriptions and audio samples
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add TTS to path
sys.path.append(str(Path(__file__).parent / "TTS"))

from voice_encoder import VoiceDescriptionEncoder
from custom_xtts import CustomXTTS
from utils import save_audio, load_config

# Patch PyTorch for XTTS checkpoint compatibility
import torch_fix

import os
import json
from pathlib import Path
from custom_xtts import CustomXTTS
from utils import load_config

def main():
    """Train voice description encoder"""
    print("🎓 VoiceForge Training")
    print("="*30)
    
    config = load_config("config.json")
    
    # Initialize custom XTTS
    custom_tts = CustomXTTS(
        tts_repo_path=config["tts_repo_path"],
        sentence_model_name=config["sentence_model_name"]
    )
    
    # Load training data
    data_path = Path(config["training_data_path"])
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("Please create training_data.json with voice descriptions and audio paths")
        return
    
    with open(data_path, 'r') as f:
        training_data_config = json.load(f)
    
    descriptions = training_data_config["voice_descriptions"]
    audio_files = training_data_config["audio_files"]
    
    print(f"Training samples: {len(descriptions)}")
    
    # Create training dataset
    training_data = custom_tts.create_training_dataset(descriptions, audio_files)
    
    # Train
    custom_tts.train_voice_encoder(
        training_data, 
        epochs=config["training_epochs"],
        batch_size=config["batch_size"]
    )
    
    # Save trained model
    custom_tts.save_voice_encoder(config["voice_encoder_path"])
    print("Training complete!")

if __name__ == "__main__":
    main()