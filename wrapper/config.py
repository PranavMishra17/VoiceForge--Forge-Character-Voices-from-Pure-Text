"""
Configuration module for VoiceForge wrapper
Handles output paths, default settings, and validation
"""

import os
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceForgeConfig:
    """Configuration class for VoiceForge wrapper"""

    # Default model paths
    DEFAULT_MODEL_PATH = 'pretrained_models/CosyVoice2-0.5B'

    # Default output directory (in src root)
    DEFAULT_OUTPUT_DIR = 'src/outputs'

    # Subdirectories for different output types
    TTS_OUTPUT_DIR = 'tts'
    EMBEDDINGS_OUTPUT_DIR = 'embeddings'
    VOICE_CONVERSION_DIR = 'voice_conversion'

    # Default synthesis parameters
    DEFAULT_LANGUAGE = 'en'  # English as default
    DEFAULT_SPEED = 1.0
    DEFAULT_STREAM = False
    DEFAULT_TEXT_FRONTEND = True

    # Model loading parameters
    DEFAULT_LOAD_JIT = False
    DEFAULT_LOAD_TRT = False
    DEFAULT_LOAD_VLLM = False
    DEFAULT_FP16 = False

    # Audio parameters
    SAMPLE_RATE_16K = 16000
    SAMPLE_RATE_24K = 24000

    # File naming
    TTS_PREFIX = 'tts'
    EMBEDDING_PREFIX = 'embedding'
    VC_PREFIX = 'vc'

    def __init__(self, output_root: Optional[str] = None):
        """
        Initialize configuration

        Args:
            output_root: Root directory for all outputs. If None, uses DEFAULT_OUTPUT_DIR
        """
        try:
            if output_root is None:
                self.output_root = Path(self.DEFAULT_OUTPUT_DIR)
            else:
                self.output_root = Path(output_root)

            # Create output directories
            self._setup_output_directories()
            logger.info(f"VoiceForge config initialized with output root: {self.output_root}")

        except Exception as e:
            logger.error(f"Failed to initialize VoiceForge config: {str(e)}")
            raise

    def _setup_output_directories(self):
        """Create all necessary output directories"""
        try:
            # Create main output directory
            self.output_root.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (self.output_root / self.TTS_OUTPUT_DIR).mkdir(exist_ok=True)
            (self.output_root / self.EMBEDDINGS_OUTPUT_DIR).mkdir(exist_ok=True)
            (self.output_root / self.VOICE_CONVERSION_DIR).mkdir(exist_ok=True)

            logger.info("Output directories created successfully")

        except Exception as e:
            logger.error(f"Failed to create output directories: {str(e)}")
            raise

    def get_tts_output_dir(self) -> Path:
        """Get TTS output directory path"""
        return self.output_root / self.TTS_OUTPUT_DIR

    def get_embeddings_output_dir(self) -> Path:
        """Get embeddings output directory path"""
        return self.output_root / self.EMBEDDINGS_OUTPUT_DIR

    def get_vc_output_dir(self) -> Path:
        """Get voice conversion output directory path"""
        return self.output_root / self.VOICE_CONVERSION_DIR

    def get_output_path(self, output_type: str, filename: str) -> Path:
        """
        Get full output path for a file

        Args:
            output_type: Type of output ('tts', 'embeddings', 'voice_conversion')
            filename: Name of the output file

        Returns:
            Full path to output file
        """
        try:
            if output_type == 'tts':
                return self.get_tts_output_dir() / filename
            elif output_type == 'embeddings':
                return self.get_embeddings_output_dir() / filename
            elif output_type == 'voice_conversion':
                return self.get_vc_output_dir() / filename
            else:
                raise ValueError(f"Unknown output type: {output_type}")
        except Exception as e:
            logger.error(f"Failed to get output path: {str(e)}")
            raise

    @staticmethod
    def validate_speed(speed: float) -> float:
        """
        Validate and clamp speed parameter

        Args:
            speed: Speed value to validate

        Returns:
            Validated speed value
        """
        if speed <= 0:
            logger.warning(f"Invalid speed {speed}, using default 1.0")
            return 1.0
        if speed > 2.0:
            logger.warning(f"Speed {speed} too high, clamping to 2.0")
            return 2.0
        if speed < 0.5:
            logger.warning(f"Speed {speed} too low, clamping to 0.5")
            return 0.5
        return speed

    @staticmethod
    def validate_language(language: Optional[str]) -> str:
        """
        Validate language parameter

        Args:
            language: Language code ('en', 'zh', or None)

        Returns:
            Validated language code
        """
        if language is None:
            return VoiceForgeConfig.DEFAULT_LANGUAGE

        language = language.lower()
        if language not in ['en', 'zh', 'english', 'chinese']:
            logger.warning(f"Unknown language {language}, defaulting to English")
            return 'en'

        # Normalize language codes
        if language in ['english', 'en']:
            return 'en'
        elif language in ['chinese', 'zh']:
            return 'zh'

        return language
