#!/usr/bin/env python3
"""torch_fix.py - PyTorch Fix"""

import torch
import torch.serialization

# Global fix for TTS loading
from TTS.TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

print("PyTorch TTS compatibility fixed")