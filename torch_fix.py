#!/usr/bin/env python3
"""
torch_fix.py - PyTorch Compatibility Fix
"""

import torch
import torch.serialization

def setup_torch_compatibility():
    """Fix PyTorch weights loading for TTS models"""
    try:
        from TTS.TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
        print("PyTorch compatibility fixed")
    except ImportError as e:
        print(f"Warning: {e}")

setup_torch_compatibility()