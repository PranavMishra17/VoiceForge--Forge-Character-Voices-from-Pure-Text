# VoiceForge 🎭

**Forge Character Voices from Pure Text**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

![VoiceForge Architecture](https://github.com/PranavMishra17/VoiceForge--Forge-Character-Voices-from-Pure-Text/blob/ec67f1f9dc9a2779a9839109084cfe4a6c031f5a/head.png)

## Overview

VoiceForge is an AI-powered text-to-voice architecture built on CosyVoice that generates consistent character voices from natural language descriptions and audio samples. Create unlimited unique character voices for games, stories, and interactive applications.

## ✨ Key Features

- **Speaker Embedding Extraction**: Create persistent voice profiles from audio samples
- **Zero-Shot Voice Cloning**: Clone any voice with just a short audio sample
- **Emotion & Style Control**: Use natural language instructions for voice modulation
- **Dialogue Processing**: Batch process entire dialogue scripts with multiple characters
- **External Output Management**: All outputs saved outside CosyVoice folder structure
- **Comprehensive Logging**: Detailed logging and error handling
- **Configurable Interface**: Environment variables and configuration file support

## 🚀 Quick Start

```bash
# Clone repository with CosyVoice submodule
git clone --recursive https://github.com/PranavMishra17/VoiceForge.git
cd VoiceForge

# Install dependencies
pip install -r requirements.txt

# Download CosyVoice models
cd CosyVoice
python download_models.py
cd ..

# Test installation
python main.py --mode stats
```

## 📁 Project Structure

```
VoiceForge/
├── main.py                          # External interface entry point
├── config.py                        # Configuration management
├── CosyVoice/                       # CosyVoice repository
│   ├── cosyvoice_interface.py       # Main interface class
│   └── pretrained_models/          # Downloaded models
└── voiceforge_output/               # All outputs (auto-created)
    ├── speakers/                    # Speaker database & embeddings
    ├── audio/                       # Generated audio files
    ├── dialogue/                    # Dialogue processing results
    └── logs/                        # Application logs
```

## 💫 Usage Examples

**Extract Speaker Embedding:**
```bash
python main.py --mode extract --audio samples/wizard_voice.wav --transcript "Greetings, traveler. Welcome to the magical realm." --speaker_id wizard_voice
```

```bash
python main.py --mode extract --audio samples/character.wav --transcript "Hello, this is my character voice sample" --speaker_id my_character
```

**Simple Synthesis:**
```bash
python main.py --mode synthesize --text "Welcome to the magical realm" --output_name wizard_greeting --speaker_id wizard_voice
```

```bash
python main.py --mode synthesize --text "The ancient magic flows through these halls." --output_name magic_speech --speaker_id wizard_voice
```

**With Emotion Control:**
```bash
python main.py --mode synthesize --text "I'm so excited to see you!" --output_name excited_greeting --speaker_id wizard_voice --instruction "speak with excitement and joy"
```

```bash
python main.py --mode synthesize --text "Something dark approaches..." --output_name dark_warning --speaker_id wizard_voice --instruction "speak mysteriously and ominously"
```

**Real-time Voice Cloning:**
```bash
python main.py --mode synthesize --text "This is a test of voice cloning" --output_name cloned_test --prompt_audio samples/reference.wav --prompt_text "Original audio content"
```

**Dialogue Processing:**
```bash
python main.py --mode dialogue --script dialogue_scripts/fantasy_story.txt --dialogue_name fantasy_story --default_speaker wizard_voice
```

```bash
python main.py --mode dialogue --script sample_dialogue.txt --dialogue_name presentation --default_speaker test_speaker
```

**Speaker Management:**
```bash
python main.py --mode list
```

```bash
python main.py --mode stats
```

```bash
python main.py --mode delete --speaker_id old_speaker
```

**With Custom Output Directory:**
```bash
python main.py --mode synthesize --text "Custom output test" --output_name test --speaker_id wizard_voice --output_dir my_project_output
```


**Dialogue Script Format:**
```text
[speaker:wizard_voice] Greetings, brave adventurer!
[speaker:elf_voice,instruction:speak gently] The forest whispers your name.
[instruction:speak mysteriously] Ancient secrets lie hidden here.
Basic text without tags uses the default speaker.
```

### 6. Speaker Management

```bash
# List all available speakers
python main.py --mode list

# Delete a speaker
python main.py --mode delete --speaker_id old_speaker

# Show system statistics
python main.py --mode stats
```

## 🎛️ Configuration

### Configuration File (`voiceforge_config.json`)

Automatically generated on first run:

```json
{
  "model": {
    "model_path": "CosyVoice/pretrained_models/CosyVoice2-0.5B",
    "fp16": false
  },
  "paths": {
    "output_base_dir": "voiceforge_output",
    "speaker_db_path": null
  },
  "audio": {
    "sample_rate": 16000,
    "max_audio_length": 30
  },
  "logging": {
    "level": "INFO"
  }
}
```

### Environment Variables

Override configuration with environment variables:

```bash
export VOICEFORGE_MODEL_PATH="custom/model/path"
export VOICEFORGE_OUTPUT_DIR="custom/output/dir"
export VOICEFORGE_SPEAKER_DB="custom/speakers.json"
export VOICEFORGE_LOG_LEVEL="DEBUG"
export VOICEFORGE_FP16="true"
```

## 🎮 Programming Interface

Use VoiceForge programmatically:

```python
from CosyVoice.cosyvoice_interface import CosyVoiceInterface

# Initialize interface
voice_forge = CosyVoiceInterface(
    output_base_dir='my_project_output',
    log_level='INFO'
)

# Extract speaker embedding
success = voice_forge.extract_speaker_embedding(
    audio_path='samples/character.wav',
    transcript='Hello, this is my character voice.',
    speaker_id='my_character'
)

# Synthesize speech
audio_path = voice_forge.synthesize_speech(
    text='Welcome to the adventure!',
    output_filename='character_greeting',
    speaker_id='my_character',
    instruction='speak with enthusiasm'
)

# Process dialogue
results = voice_forge.process_dialogue_script(
    script_path='story.txt',
    dialogue_name='episode_1',
    default_speaker_id='my_character'
)
```

## 🔧 Advanced Features

### Instruction Examples

Control voice characteristics with natural language:

- `"speak with excitement and joy"`
- `"use a deep, mysterious voice"`
- `"speak slowly and thoughtfully"`
- `"add a slight accent"`
- `"speak like an elderly wizard"`

### Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC

### Hardware Requirements

- **Minimum**: RTX 3060 (12GB VRAM)
- **Recommended**: RTX 4070+ (16GB+ VRAM)
- **Storage**: 5GB for models
- **RAM**: 16GB system memory

## 📊 Output Management

All outputs are organized outside the CosyVoice folder:

```
voiceforge_output/
├── speakers/
│   ├── speaker_database.json         # Speaker metadata
│   └── embeddings/                   # Individual speaker data
├── audio/
│   ├── [output_name]_output_0.wav   # Generated audio files
│   └── character_voices/            # Character-specific audio
├── dialogue/
│   └── [dialogue_name]/             # Dialogue session results
│       ├── line_001_output_0.wav
│       ├── line_002_output_0.wav
│       └── processing_report.txt
└── logs/
    └── cosyvoice_[timestamp].log    # Application logs
```

## 🛠️ Troubleshooting

### Common Issues

**Model not found:**
```bash
cd CosyVoice
python download_models.py
```

**CUDA out of memory:**
```bash
export VOICEFORGE_FP16="true"
```

**Permission errors:**
Ensure write permissions for the output directory.

**Import errors:**
Check that all dependencies are installed and CosyVoice submodule is properly initialized.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow software architecture in `project_structure.md`
4. Add comprehensive logging and error handling
5. Update documentation
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the excellent TTS framework
- [FunAudioLLM](https://funaudiollm.github.io/) for the underlying models

## 📞 Contact

**Pranav Mishra**

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=github&logoColor=white)](https://portfolio-pranav-mishra-paranoid.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavgamedev/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:pmishr23@uic.edu)

---

*"Where Text Becomes Voice, Characters Come Alive"* ✨