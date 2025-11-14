"""
VoiceForge Wrapper
A comprehensive wrapper for CosyVoice TTS with advanced features

Main components:
- VoiceSynthesizer: Complete TTS synthesis with all modes
- SpeakerEncoder: Speaker embedding extraction and management
- VoiceForgeConfig: Configuration and output management
"""

from wrapper.synthesizer import VoiceSynthesizer
from wrapper.encoders.speaker_encoder import SpeakerEncoder
from wrapper.config import VoiceForgeConfig

__version__ = '1.0.0'
__all__ = ['VoiceSynthesizer', 'SpeakerEncoder', 'VoiceForgeConfig']
