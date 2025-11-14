"""
Speaker Encoder Wrapper for VoiceForge
Handles speaker embedding extraction and management
"""

import sys
import os
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime

# Add CosyVoice to path
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeakerEncoder:
    """
    Speaker Encoder wrapper for extracting and managing speaker embeddings
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
        Initialize Speaker Encoder

        Args:
            model_dir: Path to CosyVoice2 model directory
            output_dir: Directory to save embeddings (default: src/outputs/embeddings)
            load_jit: Whether to load JIT optimized model
            load_trt: Whether to load TensorRT optimized model
            load_vllm: Whether to load VLLM optimized model
            fp16: Whether to use FP16 precision
        """
        try:
            logger.info("Initializing SpeakerEncoder...")
            self.model_dir = model_dir
            self.output_dir = Path(output_dir) if output_dir else Path('src/outputs/embeddings')
            self.output_dir.mkdir(parents=True, exist_ok=True)

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
            logger.info(f"SpeakerEncoder initialized successfully. Sample rate: {self.sample_rate}")

        except Exception as e:
            logger.error(f"Failed to initialize SpeakerEncoder: {str(e)}")
            raise

    def load_audio(
        self,
        audio_path: str,
        target_sr: int = 16000
    ) -> torch.Tensor:
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default 16000 for speaker embedding)

        Returns:
            Audio tensor resampled to target_sr
        """
        try:
            logger.info(f"Loading audio from {audio_path}...")

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")

            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sr
                )
                waveform = resampler(waveform)
                logger.info(f"Resampled from {sample_rate}Hz to {target_sr}Hz")

            logger.info(f"Audio loaded successfully: shape={waveform.shape}")
            return waveform

        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise

    def extract_embedding(
        self,
        audio_input: Union[str, torch.Tensor],
        speaker_id: Optional[str] = None,
        save_embedding: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract speaker embedding from audio

        Args:
            audio_input: Path to audio file or audio tensor (16kHz)
            speaker_id: Optional speaker ID for saving
            save_embedding: Whether to save embedding to disk
            metadata: Optional metadata to save with embedding

        Returns:
            Dictionary containing:
                - embedding: Speaker embedding tensor
                - embedding_path: Path where embedding was saved (if save_embedding=True)
                - speaker_id: Speaker ID used
                - metadata: Metadata dictionary
        """
        try:
            logger.info("Extracting speaker embedding...")

            # Load audio if path provided
            if isinstance(audio_input, str):
                audio_16k = self.load_audio(audio_input, target_sr=16000)
            elif isinstance(audio_input, torch.Tensor):
                audio_16k = audio_input
            else:
                raise TypeError("audio_input must be str (path) or torch.Tensor")

            # Extract embedding using frontend
            embedding = self.model.frontend._extract_spk_embedding(audio_16k)

            logger.info(f"Embedding extracted: shape={embedding.shape}")

            # Generate speaker ID if not provided
            if speaker_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                speaker_id = f"speaker_{timestamp}"

            result = {
                'embedding': embedding,
                'speaker_id': speaker_id,
                'shape': list(embedding.shape),
                'metadata': metadata or {}
            }

            # Save embedding if requested
            if save_embedding:
                embedding_path = self._save_embedding(
                    embedding=embedding,
                    speaker_id=speaker_id,
                    metadata=metadata
                )
                result['embedding_path'] = str(embedding_path)
                logger.info(f"Embedding saved to {embedding_path}")

            return result

        except Exception as e:
            logger.error(f"Failed to extract embedding: {str(e)}")
            raise

    def extract_embedding_with_features(
        self,
        audio_input: Union[str, torch.Tensor],
        prompt_text: str,
        speaker_id: Optional[str] = None,
        save_embedding: bool = True
    ) -> Dict[str, Any]:
        """
        Extract complete speaker features (embedding + speech tokens + features)
        This creates a complete speaker profile that can be used for zero-shot TTS

        Args:
            audio_input: Path to audio file or audio tensor (16kHz)
            prompt_text: Transcription of the audio (for zero-shot TTS)
            speaker_id: Optional speaker ID
            save_embedding: Whether to save the complete profile

        Returns:
            Dictionary with complete speaker profile
        """
        try:
            logger.info("Extracting complete speaker features...")

            # Load audio if path provided
            if isinstance(audio_input, str):
                audio_16k = self.load_audio(audio_input, target_sr=16000)
            elif isinstance(audio_input, torch.Tensor):
                audio_16k = audio_input
            else:
                raise TypeError("audio_input must be str (path) or torch.Tensor")

            # Generate speaker ID if not provided
            if speaker_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                speaker_id = f"speaker_{timestamp}"

            # Add speaker to model's spk2info using add_zero_shot_spk
            # This extracts all necessary features
            success = self.model.add_zero_shot_spk(
                prompt_text=prompt_text,
                prompt_speech_16k=audio_16k,
                zero_shot_spk_id=speaker_id
            )

            if not success:
                raise RuntimeError("Failed to add zero-shot speaker")

            # Retrieve the stored speaker info
            speaker_profile = self.model.frontend.spk2info[speaker_id]

            logger.info(f"Complete speaker features extracted for {speaker_id}")

            result = {
                'speaker_id': speaker_id,
                'profile': speaker_profile,
                'prompt_text': prompt_text,
            }

            # Save if requested
            if save_embedding:
                profile_path = self._save_speaker_profile(
                    speaker_id=speaker_id,
                    profile=speaker_profile,
                    prompt_text=prompt_text
                )
                result['profile_path'] = str(profile_path)
                logger.info(f"Speaker profile saved to {profile_path}")

            return result

        except Exception as e:
            logger.error(f"Failed to extract speaker features: {str(e)}")
            raise

    def _save_embedding(
        self,
        embedding: torch.Tensor,
        speaker_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save speaker embedding to disk

        Args:
            embedding: Embedding tensor to save
            speaker_id: Speaker ID
            metadata: Optional metadata

        Returns:
            Path to saved embedding file
        """
        try:
            # Create filename
            filename = f"{speaker_id}_embedding.pt"
            embedding_path = self.output_dir / filename

            # Prepare data to save
            save_data = {
                'embedding': embedding,
                'speaker_id': speaker_id,
                'timestamp': datetime.now().isoformat(),
                'shape': list(embedding.shape),
                'metadata': metadata or {}
            }

            # Save embedding
            torch.save(save_data, embedding_path)

            # Also save metadata as JSON for easy inspection
            metadata_path = self.output_dir / f"{speaker_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'speaker_id': speaker_id,
                    'timestamp': save_data['timestamp'],
                    'shape': save_data['shape'],
                    'metadata': save_data['metadata']
                }, f, indent=2)

            return embedding_path

        except Exception as e:
            logger.error(f"Failed to save embedding: {str(e)}")
            raise

    def _save_speaker_profile(
        self,
        speaker_id: str,
        profile: Dict[str, torch.Tensor],
        prompt_text: str
    ) -> Path:
        """
        Save complete speaker profile

        Args:
            speaker_id: Speaker ID
            profile: Speaker profile dictionary
            prompt_text: Prompt text used

        Returns:
            Path to saved profile
        """
        try:
            filename = f"{speaker_id}_profile.pt"
            profile_path = self.output_dir / filename

            save_data = {
                'speaker_id': speaker_id,
                'profile': profile,
                'prompt_text': prompt_text,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(save_data, profile_path)

            # Save metadata
            metadata_path = self.output_dir / f"{speaker_id}_profile_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'speaker_id': speaker_id,
                    'prompt_text': prompt_text,
                    'timestamp': save_data['timestamp'],
                    'profile_keys': list(profile.keys())
                }, f, indent=2)

            return profile_path

        except Exception as e:
            logger.error(f"Failed to save speaker profile: {str(e)}")
            raise

    def load_embedding(self, embedding_path: str) -> Dict[str, Any]:
        """
        Load a saved speaker embedding

        Args:
            embedding_path: Path to saved embedding file

        Returns:
            Dictionary with embedding data
        """
        try:
            logger.info(f"Loading embedding from {embedding_path}...")

            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

            data = torch.load(embedding_path)
            logger.info(f"Embedding loaded: speaker_id={data.get('speaker_id', 'unknown')}")

            return data

        except Exception as e:
            logger.error(f"Failed to load embedding: {str(e)}")
            raise

    def load_speaker_profile(self, profile_path: str) -> Dict[str, Any]:
        """
        Load a saved speaker profile

        Args:
            profile_path: Path to saved profile file

        Returns:
            Dictionary with profile data
        """
        try:
            logger.info(f"Loading speaker profile from {profile_path}...")

            if not os.path.exists(profile_path):
                raise FileNotFoundError(f"Profile file not found: {profile_path}")

            data = torch.load(profile_path)
            logger.info(f"Profile loaded: speaker_id={data.get('speaker_id', 'unknown')}")

            return data

        except Exception as e:
            logger.error(f"Failed to load speaker profile: {str(e)}")
            raise

    def batch_extract_embeddings(
        self,
        audio_files: list,
        speaker_ids: Optional[list] = None,
        save_embeddings: bool = True
    ) -> list:
        """
        Extract embeddings from multiple audio files

        Args:
            audio_files: List of paths to audio files
            speaker_ids: Optional list of speaker IDs (same length as audio_files)
            save_embeddings: Whether to save embeddings

        Returns:
            List of embedding dictionaries
        """
        try:
            logger.info(f"Batch extracting embeddings from {len(audio_files)} files...")

            results = []
            for i, audio_file in enumerate(audio_files):
                try:
                    speaker_id = speaker_ids[i] if speaker_ids and i < len(speaker_ids) else None
                    result = self.extract_embedding(
                        audio_input=audio_file,
                        speaker_id=speaker_id,
                        save_embedding=save_embeddings
                    )
                    results.append(result)
                    logger.info(f"Processed {i + 1}/{len(audio_files)}: {audio_file}")
                except Exception as e:
                    logger.error(f"Failed to process {audio_file}: {str(e)}")
                    results.append({'error': str(e), 'audio_file': audio_file})

            logger.info(f"Batch extraction completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch extraction failed: {str(e)}")
            raise
