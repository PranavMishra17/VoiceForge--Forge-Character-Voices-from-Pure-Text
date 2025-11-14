"""
Voice Synthesizer Wrapper for VoiceForge
Handles all TTS synthesis operations with comprehensive parameter support
"""

import sys
import os
import logging
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime

# Add CosyVoice to path
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from wrapper.config import VoiceForgeConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
    """
    Postprocess audio like official CosyVoice webui
    Trims silence and normalizes audio

    Args:
        speech: Audio tensor [1, samples]
        top_db: Threshold for silence removal
        hop_length: Hop length for librosa
        win_length: Window length for librosa
        max_val: Maximum value for normalization

    Returns:
        Processed audio tensor
    """
    # Convert to numpy for librosa
    speech_np = speech.squeeze().cpu().numpy()

    # Trim silence
    speech_np, _ = librosa.effects.trim(
        speech_np,
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    # Convert back to tensor
    speech = torch.from_numpy(speech_np).unsqueeze(0)

    # Normalize
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    return speech


class VoiceSynthesizer:
    """
    Comprehensive Voice Synthesizer wrapper for CosyVoice2
    Supports all synthesis modes with full parameter control
    """

    def __init__(
        self,
        model_dir: str = 'pretrained_models/CosyVoice2-0.5B',
        output_dir: Optional[str] = None,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False
    ):
        """
        Initialize Voice Synthesizer

        Args:
            model_dir: Path to CosyVoice2 model directory
            output_dir: Root directory for outputs (default: src/outputs)
            load_jit: Whether to load JIT optimized model
            load_trt: Whether to load TensorRT optimized model
            load_vllm: Whether to load VLLM optimized model
            fp16: Whether to use FP16 precision
        """
        try:
            logger.info("Initializing VoiceSynthesizer...")

            # Initialize configuration
            self.config = VoiceForgeConfig(output_root=output_dir)

            # Load CosyVoice2 model
            logger.info(f"Loading CosyVoice2 model from {model_dir}...")
            self.model = CosyVoice2(
                model_dir=model_dir,
                load_jit=load_jit,
                load_trt=load_trt,
                load_vllm=load_vllm,
                fp16=fp16
            )

            self.sample_rate = self.model.sample_rate
            logger.info(f"VoiceSynthesizer initialized successfully. Sample rate: {self.sample_rate}")

        except Exception as e:
            logger.error(f"Failed to initialize VoiceSynthesizer: {str(e)}")
            raise

    def synthesize_simple(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False,
        text_frontend: bool = True
    ) -> Dict[str, Any]:
        """
        Simple TTS synthesis using built-in speakers (SFT mode)

        Note: This requires the model to have pre-trained speakers.
        Use synthesize_zero_shot() with audio samples if no speakers available.

        Args:
            text: Text to synthesize
            speaker_id: Speaker ID from model's built-in speakers
            language: Language code ('en' or 'zh'), defaults to English
            speed: Speech speed (0.5 to 2.0, default 1.0)
            output_filename: Optional output filename
            stream: Whether to use streaming synthesis
            text_frontend: Whether to use text normalization

        Returns:
            Dictionary with synthesis results
        """
        try:
            logger.info("Starting simple TTS synthesis...")

            # Validate parameters
            language = self.config.validate_language(language)
            speed = self.config.validate_speed(speed)

            # Check if we have built-in speakers
            available_speakers = self.list_available_speakers()

            if not available_speakers:
                raise ValueError(
                    "No built-in speakers available in the model. "
                    "Use synthesize_zero_shot() with an audio sample instead, or "
                    "use synthesize_with_speaker_embedding() with a pre-saved speaker profile."
                )

            # Use provided speaker_id or first available speaker
            if speaker_id is None:
                speaker_id = available_speakers[0]
                logger.info(f"No speaker_id provided, using default: {speaker_id}")
            elif speaker_id not in available_speakers:
                raise ValueError(
                    f"Speaker '{speaker_id}' not found. "
                    f"Available speakers: {available_speakers}"
                )

            # Perform synthesis using SFT mode
            logger.info(f"Synthesizing with speaker: {speaker_id}")
            audio_chunks = []

            for i, output in enumerate(self.model.inference_sft(
                tts_text=text,
                spk_id=speaker_id,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate all audio chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"simple_{speaker_id}_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('tts', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Synthesis completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'speaker_id': speaker_id,
                'parameters': {
                    'text': text,
                    'language': language,
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Simple synthesis failed: {str(e)}")
            raise

    def synthesize_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio: Optional[Union[str, torch.Tensor]] = None,
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False,
        text_frontend: bool = True
    ) -> Dict[str, Any]:
        """
        Zero-shot TTS synthesis with audio prompt

        Args:
            text: Text to synthesize
            prompt_text: Transcription of prompt audio
            prompt_audio: Path to prompt audio file or audio tensor (16kHz), can be None
            speaker_id: Optional pre-saved speaker ID to use instead of prompt_audio
            language: Language code ('en' or 'zh'), defaults to English
            speed: Speech speed (0.5 to 2.0, default 1.0)
            output_filename: Optional output filename
            stream: Whether to use streaming synthesis
            text_frontend: Whether to use text normalization

        Returns:
            Dictionary with synthesis results including output path
        """
        try:
            logger.info("Starting zero-shot TTS synthesis...")

            # Validate parameters
            language = self.config.validate_language(language)
            speed = self.config.validate_speed(speed)

            # Load prompt audio if needed
            prompt_audio_16k = None
            if prompt_audio is not None:
                if isinstance(prompt_audio, str):
                    prompt_audio_16k = self._load_audio(prompt_audio, target_sr=16000)
                elif isinstance(prompt_audio, torch.Tensor):
                    prompt_audio_16k = prompt_audio
                else:
                    raise TypeError("prompt_audio must be str (path) or torch.Tensor")

            # Use speaker_id if provided, otherwise use prompt audio
            zero_shot_spk_id = speaker_id if speaker_id else ''

            # Perform synthesis
            logger.info(f"Synthesizing: '{text[:50]}...'")
            audio_chunks = []

            for i, output in enumerate(self.model.inference_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_audio_16k,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate all audio chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"zero_shot_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('tts', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Synthesis completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'parameters': {
                    'text': text,
                    'language': language,
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Zero-shot synthesis failed: {str(e)}")
            raise

    def synthesize_with_speaker_embedding(
        self,
        text: str,
        speaker_embedding: Union[str, torch.Tensor, Dict[str, Any]],
        language: Optional[str] = None,
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False,
        text_frontend: bool = True
    ) -> Dict[str, Any]:
        """
        TTS synthesis using a pre-extracted speaker embedding vector

        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding (path to .pt file, tensor, or speaker_id string)
            language: Language code ('en' or 'zh'), defaults to English
            speed: Speech speed (0.5 to 2.0, default 1.0)
            output_filename: Optional output filename
            stream: Whether to use streaming synthesis
            text_frontend: Whether to use text normalization

        Returns:
            Dictionary with synthesis results
        """
        try:
            logger.info("Starting TTS synthesis with speaker embedding...")

            # Validate parameters
            language = self.config.validate_language(language)
            speed = self.config.validate_speed(speed)

            # Handle different speaker_embedding input types
            if isinstance(speaker_embedding, str):
                # Check if it's a speaker_id in the model's spk2info
                if speaker_embedding in self.model.frontend.spk2info:
                    speaker_id = speaker_embedding
                    logger.info(f"Using existing speaker profile: {speaker_id}")
                # Otherwise, assume it's a path to a saved embedding/profile
                elif os.path.exists(speaker_embedding):
                    speaker_id = self._load_speaker_profile(speaker_embedding)
                else:
                    raise ValueError(f"Speaker embedding not found: {speaker_embedding}")
            elif isinstance(speaker_embedding, dict):
                # Direct profile dictionary
                speaker_id = self._add_temp_speaker_profile(speaker_embedding)
            elif isinstance(speaker_embedding, torch.Tensor):
                # Direct embedding tensor - need to create minimal profile
                speaker_id = self._create_speaker_profile_from_embedding(speaker_embedding)
            else:
                raise TypeError("speaker_embedding must be str, dict, or torch.Tensor")

            # Perform synthesis using the speaker profile
            logger.info(f"Synthesizing with speaker: {speaker_id}")
            audio_chunks = []

            # Use zero-shot synthesis with the speaker_id
            for i, output in enumerate(self.model.inference_zero_shot(
                tts_text=text,
                prompt_text='',
                prompt_speech_16k=None,
                zero_shot_spk_id=speaker_id,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate all audio chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"embedding_{speaker_id}_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('tts', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Synthesis completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'speaker_id': speaker_id,
                'parameters': {
                    'text': text,
                    'language': language,
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Embedding-based synthesis failed: {str(e)}")
            raise

    def synthesize_cross_lingual(
        self,
        text: str,
        prompt_audio: Union[str, torch.Tensor],
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False,
        text_frontend: bool = True
    ) -> Dict[str, Any]:
        """
        Cross-lingual TTS synthesis (e.g., English text with Chinese voice)

        Args:
            text: Text to synthesize
            prompt_audio: Path to prompt audio or audio tensor (16kHz)
            speaker_id: Optional pre-saved speaker ID
            language: Language code for the text
            speed: Speech speed
            output_filename: Optional output filename
            stream: Whether to use streaming
            text_frontend: Whether to use text normalization

        Returns:
            Dictionary with synthesis results
        """
        try:
            logger.info("Starting cross-lingual TTS synthesis...")

            # Validate parameters
            language = self.config.validate_language(language)
            speed = self.config.validate_speed(speed)

            # Load prompt audio
            if isinstance(prompt_audio, str):
                prompt_audio_16k = self._load_audio(prompt_audio, target_sr=16000)
            elif isinstance(prompt_audio, torch.Tensor):
                prompt_audio_16k = prompt_audio
            else:
                raise TypeError("prompt_audio must be str (path) or torch.Tensor")

            zero_shot_spk_id = speaker_id if speaker_id else ''

            # Perform synthesis
            logger.info(f"Synthesizing (cross-lingual): '{text[:50]}...'")
            audio_chunks = []

            for i, output in enumerate(self.model.inference_cross_lingual(
                tts_text=text,
                prompt_speech_16k=prompt_audio_16k,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate audio
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"cross_lingual_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('tts', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Cross-lingual synthesis completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'parameters': {
                    'text': text,
                    'language': language,
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Cross-lingual synthesis failed: {str(e)}")
            raise

    def synthesize_instruct(
        self,
        text: str,
        instruct_text: str,
        prompt_audio: Union[str, torch.Tensor],
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False,
        text_frontend: bool = True
    ) -> Dict[str, Any]:
        """
        Instruction-based TTS synthesis (control emotion, style, etc.)

        Args:
            text: Text to synthesize
            instruct_text: Instruction text (e.g., "speak with excitement", "calm voice")
            prompt_audio: Path to prompt audio or audio tensor (16kHz)
            speaker_id: Optional pre-saved speaker ID
            language: Language code
            speed: Speech speed
            output_filename: Optional output filename
            stream: Whether to use streaming
            text_frontend: Whether to use text normalization

        Returns:
            Dictionary with synthesis results
        """
        try:
            logger.info("Starting instruction-based TTS synthesis...")

            # Validate parameters
            language = self.config.validate_language(language)
            speed = self.config.validate_speed(speed)

            # Load prompt audio
            if isinstance(prompt_audio, str):
                prompt_audio_16k = self._load_audio(prompt_audio, target_sr=16000)
            elif isinstance(prompt_audio, torch.Tensor):
                prompt_audio_16k = prompt_audio
            else:
                raise TypeError("prompt_audio must be str (path) or torch.Tensor")

            zero_shot_spk_id = speaker_id if speaker_id else ''

            # Perform synthesis
            logger.info(f"Synthesizing with instruction: '{instruct_text}'")
            logger.info(f"Text: '{text[:50]}...'")
            audio_chunks = []

            for i, output in enumerate(self.model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_text,
                prompt_speech_16k=prompt_audio_16k,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate audio
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"instruct_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('tts', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Instruction-based synthesis completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'parameters': {
                    'text': text,
                    'instruct_text': instruct_text,
                    'language': language,
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Instruction-based synthesis failed: {str(e)}")
            raise

    def voice_conversion(
        self,
        source_audio: Union[str, torch.Tensor],
        target_audio: Union[str, torch.Tensor],
        speed: float = 1.0,
        output_filename: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Voice conversion - convert source audio to target voice

        Args:
            source_audio: Source audio path or tensor (16kHz)
            target_audio: Target voice audio path or tensor (16kHz)
            speed: Speech speed
            output_filename: Optional output filename
            stream: Whether to use streaming

        Returns:
            Dictionary with conversion results
        """
        try:
            logger.info("Starting voice conversion...")

            # Validate speed
            speed = self.config.validate_speed(speed)

            # Load source audio
            if isinstance(source_audio, str):
                source_audio_16k = self._load_audio(source_audio, target_sr=16000)
            elif isinstance(source_audio, torch.Tensor):
                source_audio_16k = source_audio
            else:
                raise TypeError("source_audio must be str (path) or torch.Tensor")

            # Load target audio
            if isinstance(target_audio, str):
                target_audio_16k = self._load_audio(target_audio, target_sr=16000)
            elif isinstance(target_audio, torch.Tensor):
                target_audio_16k = target_audio
            else:
                raise TypeError("target_audio must be str (path) or torch.Tensor")

            # Perform voice conversion
            logger.info("Converting voice...")
            audio_chunks = []

            for i, output in enumerate(self.model.inference_vc(
                source_speech_16k=source_audio_16k,
                prompt_speech_16k=target_audio_16k,
                stream=stream,
                speed=speed
            )):
                audio_chunks.append(output['tts_speech'])
                logger.info(f"Generated chunk {i + 1}")

            # Concatenate audio
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"voice_conversion_{timestamp}.wav"

            # Save audio
            output_path = self.config.get_output_path('voice_conversion', output_filename)
            torchaudio.save(str(output_path), final_audio, self.sample_rate)

            logger.info(f"Voice conversion completed. Saved to: {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'audio': final_audio,
                'sample_rate': self.sample_rate,
                'duration': final_audio.shape[1] / self.sample_rate,
                'parameters': {
                    'speed': speed,
                    'stream': stream
                }
            }

        except Exception as e:
            logger.error(f"Voice conversion failed: {str(e)}")
            raise

    # Helper methods

    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load and preprocess audio file using official CosyVoice methods

        This matches the official webui.py behavior:
        1. Load with load_wav() (uses soundfile backend)
        2. Apply postprocess() to trim and normalize
        """
        try:
            logger.info(f"Loading audio from {audio_path}...")

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Use official CosyVoice load_wav function
            # This handles: mono conversion, resampling, proper backend
            waveform = load_wav(audio_path, target_sr)

            # Apply postprocessing: trim silence and normalize
            # This is CRITICAL - without this, output is gibberish!
            waveform = postprocess(waveform)

            logger.info(f"Audio loaded and preprocessed: shape={waveform.shape}")
            return waveform

        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise

    def _load_speaker_profile(self, profile_path: str) -> str:
        """Load speaker profile and add to model"""
        try:
            data = torch.load(profile_path)
            speaker_id = data.get('speaker_id', f'loaded_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            profile = data.get('profile', data)  # Handle both formats

            self.model.frontend.spk2info[speaker_id] = profile
            logger.info(f"Loaded speaker profile: {speaker_id}")

            return speaker_id

        except Exception as e:
            logger.error(f"Failed to load speaker profile: {str(e)}")
            raise

    def _add_temp_speaker_profile(self, profile: Dict[str, Any]) -> str:
        """Add temporary speaker profile to model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            speaker_id = f"temp_{timestamp}"
            self.model.frontend.spk2info[speaker_id] = profile
            logger.info(f"Added temporary speaker profile: {speaker_id}")
            return speaker_id

        except Exception as e:
            logger.error(f"Failed to add speaker profile: {str(e)}")
            raise

    def _create_speaker_profile_from_embedding(self, embedding: torch.Tensor) -> str:
        """Create minimal speaker profile from embedding tensor"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            speaker_id = f"embedding_{timestamp}"

            # Create minimal profile with just the embedding
            # Note: This may not work for all synthesis modes
            profile = {
                'llm_embedding': embedding,
                'flow_embedding': embedding
            }

            self.model.frontend.spk2info[speaker_id] = profile
            logger.info(f"Created speaker profile from embedding: {speaker_id}")

            return speaker_id

        except Exception as e:
            logger.error(f"Failed to create speaker profile: {str(e)}")
            raise

    def list_available_speakers(self) -> List[str]:
        """List all available speaker IDs"""
        try:
            speakers = list(self.model.frontend.spk2info.keys())
            logger.info(f"Available speakers: {len(speakers)}")
            return speakers

        except Exception as e:
            logger.error(f"Failed to list speakers: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'sample_rate': self.sample_rate,
            'model_dir': self.model.model_dir,
            'available_speakers': len(self.model.frontend.spk2info),
            'output_dirs': {
                'tts': str(self.config.get_tts_output_dir()),
                'embeddings': str(self.config.get_embeddings_output_dir()),
                'voice_conversion': str(self.config.get_vc_output_dir())
            }
        }
