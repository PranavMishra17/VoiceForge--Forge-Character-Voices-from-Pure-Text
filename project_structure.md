# VoiceForge Project Structure

## Directory Layout

```
VoiceForge/                          # Root project directory
├── main.py                          # External interface entry point
├── config.py                        # Configuration management
├── voiceforge_config.json           # Configuration file (auto-generated)
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── project_structure.md             # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
│
├── CosyVoice/                       # CosyVoice repository (git submodule)
│   ├── cosyvoice_interface.py       # Main interface class (NEW)
│   ├── cosyvoice_tts.py            # Basic TTS wrapper
│   ├── dialogue_processor.py       # Dialogue processing
│   ├── speaker_embedding_extractor.py # Speaker embedding extraction
│   ├── download_models.py          # Model download script
│   ├── test_basic.py               # Basic functionality test
│   ├── webui.py                    # Web interface
│   ├── vllm_example.py             # VLLM usage example
│   ├── README.md                   # CosyVoice documentation
│   ├── third_party/                # Third-party dependencies
│   └── pretrained_models/          # Downloaded models
│       ├── CosyVoice2-0.5B/        # Main model
│       └── CosyVoice-ttsfrd/       # Text processing
│
├── voiceforge_output/               # Output directory (outside CosyVoice)
│   ├── audio/                      # Generated audio files
│   │   ├── character_voices/       # Character-specific audio
│   │   └── temp/                   # Temporary audio files
│   ├── speakers/                   # Speaker embeddings and database
│   │   ├── speaker_database.json   # Speaker metadata
│   │   └── embeddings/             # Individual speaker data
│   ├── dialogue/                   # Dialogue processing results
│   │   └── [dialogue_name]/        # Named dialogue sessions
│   │       ├── line_001_output_0.wav
│   │       ├── line_002_output_0.wav
│   │       └── processing_report.txt
│   └── logs/                       # Application logs
│       ├── cosyvoice_[timestamp].log
│       └── error.log
│
├── samples/                        # Sample audio and scripts
│   ├── audio/                     # Sample speaker audio files
│   │   ├── wizard_voice.wav
│   │   └── elf_voice.wav
│   └── scripts/                   # Sample dialogue scripts
│       ├── fantasy_story.txt
│       └── character_intro.txt
│
└── docs/                          # Documentation
    ├── api_reference.md           # API documentation
    ├── examples.md               # Usage examples
    └── troubleshooting.md        # Common issues and solutions
```

## File Descriptions

### Core Files

| File | Location | Purpose | Key Features |
|------|----------|---------|--------------|
| `main.py` | Root | External interface entry point | CLI interface, mode selection, argument parsing |
| `config.py` | Root | Configuration management | Environment overrides, validation, path management |
| `CosyVoice/cosyvoice_interface.py` | CosyVoice/ | Main interface class | Speaker management, synthesis, dialogue processing |

### CosyVoice Integration Files

| File | Location | Purpose | Modifications |
|------|----------|---------|---------------|
| `cosyvoice_tts.py` | CosyVoice/ | Basic TTS wrapper | ✅ Original functionality |
| `dialogue_processor.py` | CosyVoice/ | Dialogue processing | ✅ Original functionality |
| `speaker_embedding_extractor.py` | CosyVoice/ | Speaker extraction | ✅ Original functionality |
| `download_models.py` | CosyVoice/ | Model download | ✅ Original functionality |
| `test_basic.py` | CosyVoice/ | Basic tests | ✅ Original functionality |
| `webui.py` | CosyVoice/ | Web interface | ✅ Original functionality |

### Output Structure

#### Audio Output (`voiceforge_output/audio/`)
- Individual synthesis: `[output_name]_output_0.wav`
- Character voices: `character_voices/[character_id]/[filename].wav`
- Temporary files: `temp/temp_[timestamp].wav`

#### Speaker Database (`voiceforge_output/speakers/`)
- Database file: `speaker_database.json`
- Format:
```json
{
  "speaker_id": {
    "audio_path": "path/to/original/audio.wav",
    "transcript": "Original audio transcript",
    "created": "2024-01-01T12:00:00"
  }
}
```

#### Dialogue Output (`voiceforge_output/dialogue/[dialogue_name]/`)
- Audio files: `line_001_output_0.wav`, `line_002_output_0.wav`, ...
- Report: `processing_report.txt`

#### Logs (`voiceforge_output/logs/`)
- Main log: `cosyvoice_[timestamp].log`
- Error log: `error.log`
- Format: `timestamp - logger - level - message`

## Configuration Management

### Configuration File (`voiceforge_config.json`)
```json
{
  "model": {
    "model_path": "CosyVoice/pretrained_models/CosyVoice2-0.5B",
    "load_jit": false,
    "load_trt": false,
    "load_vllm": false,
    "fp16": false
  },
  "paths": {
    "output_base_dir": "voiceforge_output",
    "speaker_db_path": null,
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
    "file_rotation": true,
    "max_log_files": 10
  }
}
```

### Environment Variables
- `VOICEFORGE_MODEL_PATH`: Override model path
- `VOICEFORGE_OUTPUT_DIR`: Override output directory
- `VOICEFORGE_SPEAKER_DB`: Override speaker database path
- `VOICEFORGE_LOG_LEVEL`: Override logging level
- `VOICEFORGE_FP16`: Enable FP16 mode

## Software Architecture

### Design Principles
1. **Separation of Concerns**: Core CosyVoice functionality separate from VoiceForge interface
2. **Configuration Management**: Centralized, environment-variable-friendly configuration
3. **Error Handling**: Comprehensive logging and graceful error recovery
4. **Extensibility**: Modular design for easy feature addition
5. **Data Persistence**: Speaker embeddings and configuration persist across sessions

### Key Classes

#### `CosyVoiceInterface`
- **Purpose**: Main interface for all TTS functionality
- **Key Methods**:
  - `extract_speaker_embedding()`: Extract and save speaker embeddings
  - `synthesize_speech()`: Generate speech with various options
  - `process_dialogue_script()`: Process dialogue scripts
  - `list_speakers()`: Manage speaker database

#### `ConfigManager`
- **Purpose**: Handle all configuration management
- **Key Methods**:
  - `load()`: Load configuration from file
  - `save_config()`: Save current configuration
  - `validate_config()`: Validate settings
  - `apply_environment_overrides()`: Apply env var overrides

### Error Handling Strategy
1. **Logging**: Comprehensive logging at all levels
2. **Graceful Degradation**: Fallback options when features fail
3. **User Feedback**: Clear error messages and suggestions
4. **Recovery**: Automatic retry for transient failures
5. **Validation**: Input validation before processing

### External Dependencies
- **CosyVoice**: Core TTS functionality
- **PyTorch**: Deep learning framework
- **torchaudio**: Audio processing
- **pathlib**: Path management
- **json**: Configuration and data storage
- **logging**: Application logging

## Installation Structure

### Prerequisites
```bash
# System requirements
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM)
- Git LFS for model downloads
```

### Installation Steps
```bash
# 1. Clone repository
git clone --recursive https://github.com/PranavMishra17/VoiceForge.git
cd VoiceForge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
cd CosyVoice
python download_models.py
cd ..

# 4. Test installation
python main.py --mode stats
```

### Directory Permissions
- `voiceforge_output/`: Read/write access required
- `CosyVoice/pretrained_models/`: Read access required
- Log files: Write access required

## Development Guidelines

### Adding New Features
1. Update `CosyVoiceInterface` class if core functionality
2. Add configuration options to `config.py` if needed
3. Update CLI interface in `main.py`
4. Add comprehensive logging
5. Update documentation

### Testing Strategy
1. Unit tests for core functionality
2. Integration tests for full workflow
3. Performance benchmarks
4. Error handling validation

### Code Quality
- Follow PEP 8 style guidelines
- Comprehensive docstrings
- Type hints where appropriate
- Error handling for all external operations
- Logging for debugging and monitoring

## Deployment Considerations

### Production Setup
- Use environment variables for configuration
- Implement proper logging rotation
- Monitor GPU memory usage
- Set up health checks
- Consider containerization

### Scaling Options
- Model optimization (TRT, ONNX)
- Batch processing for dialogue
- Caching for repeated synthesis
- Load balancing for multiple instances