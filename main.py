#!/usr/bin/env python3
"""
main.py - VoiceForge Main Application
Entry point for the VoiceForge TTS system with sentence transformer integration
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

def main():
    """Main application entry point"""
    print("🎙️  VoiceForge - Character Voice Generation from Text")
    print("="*50)
    
    # Configuration
    config = load_config("config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Initialize custom XTTS system
    custom_tts = CustomXTTS(
        tts_repo_path=config["tts_repo_path"],
        sentence_model_name=config["sentence_model_name"]
    )
    
    # Check if voice encoder is trained
    encoder_path = config["voice_encoder_path"]
    if os.path.exists(encoder_path):
        print("Loading trained voice encoder...")
        custom_tts.load_voice_encoder(encoder_path)
    else:
        print("Voice encoder not found. Please run training first.")
        print("Use: python train.py")
        return
    
    # Interactive mode
    while True:
        print("\n" + "="*30)
        text = input("Enter text to synthesize (or 'quit'): ")
        if text.lower() == 'quit':
            break
            
        voice_desc = input("Enter voice description: ")
        language = input("Language (default: en): ").strip() or "en"
        
        print("Generating speech...")
        try:
            audio = custom_tts.text_to_speech(
                text=text,
                voice_description=voice_desc,
                language=language
            )
            
            # Save audio
            output_path = f"output_{len(text)[:10]}_{voice_desc[:20]}.wav"
            save_audio(audio, output_path)
            print(f"Audio saved: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()