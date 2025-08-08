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

### 🎭 Basic Operations

**Extract Speaker Embedding from Audio:**
```bash
python main.py --mode extract --audio audio1/test.wav --transcript "Greetings, traveler. Welcome to the magical realm." --speaker_id wizard_voice
```

```bash
python main.py --mode extract --audio samples/character.wav --transcript "Hello, this is my character voice sample" --speaker_id my_character
```

**Add Speaker from Embedding File (No Audio Needed!):**
```bash
python main.py --mode add_embedding --speaker_id new_speaker --embedding_file embeddings/speaker.npy --transcript "Speaker description"
```

```bash
python main.py --mode add_embedding --speaker_id custom_voice --embedding_file voice_embeddings/character_voice.json --transcript "Custom voice description"
```

**Create Embedding Files:**
```bash
# Using the utility script (recommended)
python create_embedding.py embeddings/test_voice.npy --random --seed 42

python create_embedding.py embeddings/custom_voice.json --random --seed 123

# Manual creation (alternative)
python -c "import numpy as np; np.save('embeddings/my_voice.npy', np.random.randn(1, 192))"

python -c "import json; import numpy as np; embedding = np.random.randn(1, 192).tolist(); json.dump({'spk_emb': embedding, 'metadata': 'voice description'}, open('embeddings/my_voice.json', 'w'))"
```

### 🎤 Speech Synthesis

**Basic Synthesis:**
```bash
python main.py --mode synthesize --text "Welcome to the magical realm" --output_name wizard_greeting --speaker_id wizard_voice
```

```bash
python main.py --mode synthesize --text "The ancient magic flows through these halls." --output_name magic_speech --speaker_id wizard_voice
```

**Synthesis with Emotion Control:**
```bash
python main.py --mode synthesize --text "I'm absolutely thrilled to meet you!" --output_name excited_greeting --speaker_id wizard_voice --emotion excited --speed 1.1
```

```bash
python main.py --mode synthesize --text "Something dark approaches from the shadows..." --output_name dark_warning --speaker_id wizard_voice --emotion mysterious --tone whispering
```

**Synthesis with Tone Control:**
```bash
python main.py --mode synthesize --text "Please consider this proposal carefully." --output_name formal_speech --speaker_id business_voice --tone professional --emotion confident
```

```bash
python main.py --mode synthesize --text "Hey there, how's it going?" --output_name casual_greeting --speaker_id friend_voice --tone casual --emotion friendly
```

**Advanced Multi-Parameter Synthesis:**
```bash
python main.py --mode synthesize --text "The ancient prophecy speaks of a chosen one who will restore balance to the realm." --output_name prophecy_narration --speaker_id mystic_voice --emotion mysterious --tone dramatic --instruction "speak as an ancient oracle" --speed 0.9 --language english
```

**Multilingual Synthesis:**
```bash
python main.py --mode synthesize --text "今天天气真好，我们去公园吧" --output_name chinese_speech --speaker_id multilingual_voice --language chinese --emotion happy
```

```bash
python main.py --mode synthesize --text "こんにちは、元気ですか？" --output_name japanese_greeting --speaker_id japanese_voice --language japanese --emotion friendly
```

**Real-time Voice Cloning:**
```bash
python main.py --mode synthesize --text "This is a test of voice cloning technology" --output_name cloned_test --prompt_audio samples/reference.wav --prompt_text "Original audio content" --speed 0.9
```

### 📜 Dialogue Processing

**Process Dialogue Script:**
```bash
python main.py --mode dialogue --script dialogue_scripts/fantasy_story.txt --dialogue_name fantasy_story --default_speaker wizard_voice
```

```bash
python main.py --mode dialogue --script sample_dialogue.txt --dialogue_name presentation --default_speaker test_speaker
```

### 🗄️ Data Management

**Export All Speaker Embeddings:**
```bash
python main.py --mode export --export_path backup/all_speakers.json
```

**Import Speaker Embeddings:**
```bash
python main.py --mode import --import_path backup/all_speakers.json
```

### 📊 System Management

**List All Speakers:**
```bash
python main.py --mode list
```

**Show Enhanced Statistics:**
```bash
python main.py --mode stats
```

**Delete a Speaker:**
```bash
python main.py --mode delete --speaker_id old_speaker
```

### 🎛️ Advanced Configuration

**Custom Output Directory:**
```bash
python main.py --mode synthesize --text "Custom output test" --output_name test --speaker_id wizard_voice --output_dir my_project_output
```

**Debug Mode:**
```bash
python main.py --mode synthesize --text "Debug test" --output_name debug_test --speaker_id wizard_voice --log_level DEBUG
```

### 📝 Enhanced Dialogue Script Format

**Basic Format:**
```text
Hello, how are you doing today?

[speaker:wizard_voice] Greetings, brave adventurer!

[emotion:excited] I can't wait to begin our quest!

[tone:whispering] The guards are coming this way...

[speaker:elf_voice,emotion:gentle,tone:formal] Your Majesty, the realm is at peace.

[speaker:narrator,instruction:speak dramatically] The dragon awakened from its slumber.
```

### 🎭 Available Emotions
- `happy` - Joy and happiness
- `sad` - Sadness and melancholy  
- `angry` - Anger and intensity
- `excited` - Excitement and enthusiasm
- `calm` - Calm and peaceful
- `nervous` - Nervous with hesitation
- `confident` - Confidence and authority
- `mysterious` - Mysterious and enigmatic
- `dramatic` - Dramatic with emphasis
- `gentle` - Gentle and soft
- `energetic` - High energy and vigor
- `romantic` - Romantic and loving

### 🎯 Available Tones
- `formal` - Formal tone
- `casual` - Casual and relaxed
- `professional` - Professional business tone
- `friendly` - Warm and friendly
- `serious` - Serious and solemn
- `playful` - Playful and lighthearted
- `authoritative` - Authoritative and commanding
- `whispering` - Soft whisper
- `shouting` - Loud and forceful

### 🌍 Supported Languages
- `chinese` - Chinese (Mandarin)
- `english` - English
- `japanese` - Japanese
- `korean` - Korean
- `cantonese` - Cantonese
- `sichuanese` - Sichuanese dialect
- `shanghainese` - Shanghainese dialect

### ⚡ Speed Control
- `0.5` - Very slow
- `0.8` - Slow
- `1.0` - Normal (default)
- `1.2` - Fast
- `1.5` - Very fast

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

## 🛠️ Utilities

### Embedding File Creator

Use the included utility script to create embedding files for testing:

```bash
# Create random embedding for testing
python create_embedding.py embeddings/test_voice.npy --random --seed 42

# Create embedding from vector string (shows first 5 values)
python create_embedding.py embeddings/custom_voice.json --vector "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0,13.1,13.2,13.3,13.4,13.5,13.6,13.7,13.8,13.9,14.0,14.1,14.2,14.3,14.4,14.5,14.6,14.7,14.8,14.9,15.0,15.1,15.2,15.3,15.4,15.5,15.6,15.7,15.8,15.9,16.0,16.1,16.2,16.3,16.4,16.5,16.6,16.7,16.8,16.9,17.0,17.1,17.2,17.3,17.4,17.5,17.6,17.7,17.8,17.9,18.0,18.1,18.2,18.3,18.4,18.5,18.6,18.7,18.8,18.9,19.0,19.1,19.2,19.3,19.4,19.5,19.6,19.7,19.8,19.9,20.0"

# Show help
python create_embedding.py --help
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