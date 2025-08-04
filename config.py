"""
Configuration management for VoiceForge
Handles all configuration settings and paths
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_path: str = 'CosyVoice/pretrained_models/CosyVoice2-0.5B'
    load_jit: bool = False
    load_trt: bool = False
    load_vllm: bool = False
    fp16: bool = False


@dataclass
class PathConfig:
    """Path configuration settings."""
    output_base_dir: str = 'voiceforge_output'
    speaker_db_path: Optional[str] = None
    audio_output_dir: str = 'audio'
    dialogue_output_dir: str = 'dialogue'
    speaker_output_dir: str = 'speakers'
    logs_output_dir: str = 'logs'


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds
    audio_formats: list = None
    
    def __post_init__(self):
        if self.audio_formats is None:
            self.audio_formats = ['.wav', '.mp3', '.flac']


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_rotation: bool = True
    max_log_files: int = 10


@dataclass
class VoiceForgeConfig:
    """Main configuration class."""
    model: ModelConfig
    paths: PathConfig
    audio: AudioConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VoiceForgeConfig':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            audio=AudioConfig(**config_dict.get('audio', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'paths': asdict(self.paths),
            'audio': asdict(self.audio),
            'logging': asdict(self.logging)
        }
    
    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, config_path: str) -> 'VoiceForgeConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Return default config if file doesn't exist
            return cls.default()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def default(cls) -> 'VoiceForgeConfig':
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            paths=PathConfig(),
            audio=AudioConfig(),
            logging=LoggingConfig()
        )
    
    def get_full_paths(self) -> Dict[str, Path]:
        """Get all full output paths."""
        base_dir = Path(self.paths.output_base_dir)
        
        return {
            'base': base_dir,
            'audio': base_dir / self.paths.audio_output_dir,
            'dialogue': base_dir / self.paths.dialogue_output_dir,
            'speakers': base_dir / self.paths.speaker_output_dir,
            'logs': base_dir / self.paths.logs_output_dir,
            'speaker_db': Path(self.paths.speaker_db_path) if self.paths.speaker_db_path 
                         else base_dir / self.paths.speaker_output_dir / 'speaker_database.json'
        }


class ConfigManager:
    """Configuration manager for VoiceForge."""
    
    def __init__(self, config_path: str = 'voiceforge_config.json'):
        self.config_path = Path(config_path)
        self.config = VoiceForgeConfig.load(str(self.config_path))
    
    def save_config(self):
        """Save current configuration."""
        self.config.save(str(self.config_path))
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        config_dict = self.config.to_dict()
        
        # Deep update
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        self.config = VoiceForgeConfig.from_dict(config_dict)
    
    def reset_to_default(self):
        """Reset configuration to default values."""
        self.config = VoiceForgeConfig.default()
    
    def validate_config(self) -> tuple[bool, list]:
        """Validate configuration settings."""
        errors = []
        
        # Validate model path
        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            errors.append(f"Model path does not exist: {model_path}")
        
        # Validate audio formats
        if not self.config.audio.audio_formats:
            errors.append("No audio formats specified")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level not in valid_levels:
            errors.append(f"Invalid logging level: {self.config.logging.level}")
        
        # Validate sample rate
        if self.config.audio.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        
        return len(errors) == 0, errors
    
    def create_directories(self):
        """Create all necessary output directories."""
        paths = self.config.get_full_paths()
        
        for path_name, path in paths.items():
            if path_name != 'speaker_db':  # Skip file path
                path.mkdir(parents=True, exist_ok=True)
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # Model configuration
        if os.getenv('VOICEFORGE_MODEL_PATH'):
            overrides.setdefault('model', {})['model_path'] = os.getenv('VOICEFORGE_MODEL_PATH')
        
        if os.getenv('VOICEFORGE_FP16'):
            overrides.setdefault('model', {})['fp16'] = os.getenv('VOICEFORGE_FP16').lower() == 'true'
        
        # Path configuration
        if os.getenv('VOICEFORGE_OUTPUT_DIR'):
            overrides.setdefault('paths', {})['output_base_dir'] = os.getenv('VOICEFORGE_OUTPUT_DIR')
        
        if os.getenv('VOICEFORGE_SPEAKER_DB'):
            overrides.setdefault('paths', {})['speaker_db_path'] = os.getenv('VOICEFORGE_SPEAKER_DB')
        
        # Logging configuration
        if os.getenv('VOICEFORGE_LOG_LEVEL'):
            overrides.setdefault('logging', {})['level'] = os.getenv('VOICEFORGE_LOG_LEVEL')
        
        return overrides
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()
        if overrides:
            self.update_config(overrides)


# Global configuration instance
_config_manager = None


def get_config_manager(config_path: str = 'voiceforge_config.json') -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
        _config_manager.apply_environment_overrides()
    return _config_manager


def get_config() -> VoiceForgeConfig:
    """Get current configuration."""
    return get_config_manager().config


# Example configuration file content
EXAMPLE_CONFIG = {
    "model": {
        "model_path": "CosyVoice/pretrained_models/CosyVoice2-0.5B",
        "load_jit": False,
        "load_trt": False,
        "load_vllm": False,
        "fp16": False
    },
    "paths": {
        "output_base_dir": "voiceforge_output",
        "speaker_db_path": None,
        "audio_output_dir": "audio",
        "dialogue_output_dir": "dialogue",
        "speaker_output_dir": "speakers",
        "logs_output_dir": "logs"
    },
    "audio": {
        "sample_rate": 16000,
        "max_audio_length": 30,
        "audio_formats": [".wav", ".mp3", ".flac"]
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_rotation": True,
        "max_log_files": 10
    }
}


if __name__ == "__main__":
    # Create example configuration file
    config = VoiceForgeConfig.default()
    config.save('voiceforge_config.json')
    print("Created example configuration file: voiceforge_config.json")