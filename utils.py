#!/usr/bin/env python3
"""
utils.py - Utility Functions
Common utilities for VoiceForge
"""

import json
import os
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        return json.load(f)

def create_default_config(config_path: str):
    """Create default configuration file"""
    default_config = {
        "tts_repo_path": "./TTS",
        "sentence_model_name": "all-MiniLM-L6-v2",
        "voice_encoder_path": "voice_encoder.pth",
        "training_data_path": "training_data.json",
        "training_epochs": 100,
        "batch_size": 8,
        "output_dir": "outputs"
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Created default config: {config_path}")

def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 24000):
    """Save audio array to WAV file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Ensure audio is in correct format
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio)

def create_sample_training_data():
    """Create sample training data file"""
    sample_data = {
        "voice_descriptions": [
            "Deep male voice with authority and confidence",
            "Elderly male voice, wise and gentle",
            "Young female voice, cheerful and energetic",
        ],
        "audio_files": [
            "training_audio/1.wav",
            "training_audio/2.wav", 
            "training_audio/3.wav",
        ]
    }
    
    with open("training_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created sample training_data.json")