# VoiceForge Wrapper

A comprehensive Python wrapper for CosyVoice2 TTS engine with advanced features including zero-shot voice cloning, speaker embedding management, cross-lingual synthesis, and instruction-based voice control.

## Features

### 1. **Voice Synthesis** (`VoiceSynthesizer`)
- ✅ Simple TTS with default English voice
- ✅ Zero-shot voice cloning from audio samples
- ✅ Cross-lingual synthesis (speak text in different language than sample)
- ✅ Instruction-based synthesis (control emotion, style, tone)
- ✅ Voice conversion (convert one voice to another)
- ✅ Synthesis with pre-saved speaker embeddings
- ✅ Full parameter control (speed, language, streaming, etc.)

### 2. **Speaker Embedding** (`SpeakerEncoder`)
- ✅ Extract speaker embeddings from audio
- ✅ Extract complete speaker profiles (embeddings + features)
- ✅ Save and load speaker profiles
- ✅ Batch embedding extraction
- ✅ Metadata management

### 3. **Configuration** (`VoiceForgeConfig`)
- ✅ Automatic output directory management
- ✅ Separate folders for TTS, embeddings, voice conversion
- ✅ Parameter validation (speed, language, etc.)
- ✅ Default English language setting

## Installation

```bash
# All dependencies should already be installed for CosyVoice
# The wrapper uses the same dependencies
```

## Quick Start

### Basic TTS Synthesis

```python
from wrapper import VoiceSynthesizer

# Initialize
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs'
)

# Simple synthesis
result = synthesizer.synthesize_simple(
    text="Hello, this is a test of the VoiceForge system.",
    language='en',  # English (default)
    speed=1.0,
    output_filename='test.wav'
)

print(f"Audio saved to: {result['output_path']}")
```

### Zero-Shot Voice Cloning

```python
# Clone a voice from a 3-10 second audio sample
result = synthesizer.synthesize_zero_shot(
    text="This is the text I want to say in the cloned voice.",
    prompt_text="This is what the speaker says in the sample.",
    prompt_audio="path/to/speaker_sample.wav",
    language='en',
    speed=1.0,
    output_filename='cloned_voice.wav'
)
```

### Extract Speaker Embedding

```python
from wrapper import SpeakerEncoder

# Initialize encoder
encoder = SpeakerEncoder(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs/embeddings'
)

# Extract and save speaker profile
result = encoder.extract_embedding_with_features(
    audio_input='path/to/speaker.wav',
    prompt_text='Sample text from the speaker',
    speaker_id='my_custom_voice',
    save_embedding=True
)

print(f"Profile saved to: {result['profile_path']}")
```

### Use Saved Speaker Embedding

```python
# Synthesize using saved embedding
result = synthesizer.synthesize_with_speaker_embedding(
    text="New text to synthesize in the saved voice.",
    speaker_embedding='src/outputs/embeddings/my_custom_voice_profile.pt',
    language='en',
    output_filename='from_embedding.wav'
)

# Or use speaker_id directly (after loading)
result = synthesizer.synthesize_with_speaker_embedding(
    text="Another sentence with the same voice.",
    speaker_embedding='my_custom_voice',  # speaker_id
    language='en'
)
```

### Cross-Lingual Synthesis

```python
# English text with Chinese voice (or vice versa)
result = synthesizer.synthesize_cross_lingual(
    text="Hello, this is English text spoken in a Chinese accent.",
    prompt_audio='path/to/chinese_speaker.wav',
    language='en',
    output_filename='cross_lingual.wav'
)
```

### Instruction-Based Synthesis

```python
# Control emotion and style
result = synthesizer.synthesize_instruct(
    text="This is amazing news!",
    instruct_text="Speak with excitement and enthusiasm",
    prompt_audio='path/to/speaker.wav',
    language='en',
    output_filename='excited_speech.wav'
)
```

### Voice Conversion

```python
# Convert source audio to target voice
result = synthesizer.voice_conversion(
    source_audio='path/to/source.wav',
    target_audio='path/to/target_voice.wav',
    speed=1.0,
    output_filename='converted.wav'
)
```

## API Reference

### VoiceSynthesizer

#### Methods

**`synthesize_simple(text, language='en', speed=1.0, output_filename=None, stream=False)`**
- Simple TTS with default voice
- **Parameters:**
  - `text` (str): Text to synthesize
  - `language` (str): 'en' or 'zh' (default: 'en')
  - `speed` (float): Speech speed 0.5-2.0 (default: 1.0)
  - `output_filename` (str): Optional filename
  - `stream` (bool): Use streaming synthesis
- **Returns:** Dict with output_path, audio, duration, etc.

**`synthesize_zero_shot(text, prompt_text, prompt_audio, speaker_id=None, ...)`**
- Zero-shot voice cloning
- **Parameters:**
  - `text` (str): Text to synthesize
  - `prompt_text` (str): Transcription of prompt audio
  - `prompt_audio` (str|Tensor): Audio file path or tensor
  - `speaker_id` (str): Optional pre-saved speaker ID
  - `language` (str): Language code
  - `speed` (float): Speech speed
  - `output_filename` (str): Optional filename
  - `stream` (bool): Use streaming
  - `text_frontend` (bool): Use text normalization

**`synthesize_with_speaker_embedding(text, speaker_embedding, ...)`**
- Synthesize using speaker embedding
- **Parameters:**
  - `text` (str): Text to synthesize
  - `speaker_embedding` (str|Tensor|dict): Embedding file path, tensor, or speaker_id
  - Other parameters same as above

**`synthesize_cross_lingual(text, prompt_audio, ...)`**
- Cross-lingual synthesis

**`synthesize_instruct(text, instruct_text, prompt_audio, ...)`**
- Instruction-based synthesis with emotion/style control
- **Additional Parameter:**
  - `instruct_text` (str): Instruction (e.g., "speak excitedly")

**`voice_conversion(source_audio, target_audio, speed=1.0, ...)`**
- Convert source voice to target voice
- **Parameters:**
  - `source_audio` (str|Tensor): Source audio
  - `target_audio` (str|Tensor): Target voice

**`list_available_speakers()`**
- Returns list of available speaker IDs

**`get_model_info()`**
- Returns model information and output directories

### SpeakerEncoder

#### Methods

**`extract_embedding(audio_input, speaker_id=None, save_embedding=True, metadata=None)`**
- Extract speaker embedding from audio
- **Parameters:**
  - `audio_input` (str|Tensor): Audio file or tensor
  - `speaker_id` (str): Optional speaker ID
  - `save_embedding` (bool): Save to disk
  - `metadata` (dict): Optional metadata
- **Returns:** Dict with embedding, speaker_id, path

**`extract_embedding_with_features(audio_input, prompt_text, speaker_id=None, save_embedding=True)`**
- Extract complete speaker profile (recommended)
- **Parameters:**
  - `audio_input` (str|Tensor): Audio file or tensor
  - `prompt_text` (str): Transcription of audio
  - `speaker_id` (str): Optional speaker ID
  - `save_embedding` (bool): Save to disk
- **Returns:** Dict with profile, speaker_id, path

**`load_embedding(embedding_path)`**
- Load saved embedding file

**`load_speaker_profile(profile_path)`**
- Load saved speaker profile

**`batch_extract_embeddings(audio_files, speaker_ids=None, save_embeddings=True)`**
- Extract embeddings from multiple files
- **Parameters:**
  - `audio_files` (list): List of audio file paths
  - `speaker_ids` (list): Optional list of speaker IDs
  - `save_embeddings` (bool): Save embeddings

## Output Structure

```
src/outputs/
├── tts/                      # TTS synthesis outputs
│   ├── zero_shot_*.wav
│   ├── embedding_*.wav
│   └── ...
├── embeddings/               # Speaker embeddings
│   ├── speaker_001_embedding.pt
│   ├── speaker_001_metadata.json
│   ├── speaker_001_profile.pt
│   └── ...
└── voice_conversion/         # Voice conversion outputs
    └── converted_*.wav
```

## Examples

See the `examples/` directory for comprehensive demos:

- `demo_basic_synthesis.py` - Basic TTS synthesis examples
- `demo_speaker_embedding.py` - Speaker embedding extraction and usage
- `demo_advanced_synthesis.py` - Advanced features (zero-shot, cross-lingual, etc.)

Run examples:
```bash
python examples/demo_basic_synthesis.py
python examples/demo_speaker_embedding.py
python examples/demo_advanced_synthesis.py
```

## Error Handling

All methods include comprehensive error handling with try-catch blocks:

```python
try:
    result = synthesizer.synthesize_simple(
        text="Test",
        language='en'
    )
    print(f"Success: {result['output_path']}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Parameters

### Language Codes
- `'en'` or `'english'` - English (default)
- `'zh'` or `'chinese'` - Chinese

### Speed Range
- Minimum: `0.5` (half speed)
- Default: `1.0` (normal speed)
- Maximum: `2.0` (double speed)

### Audio Requirements
- **Sample Rate:** 16kHz for input audio
- **Format:** WAV, MP3, or any format supported by torchaudio
- **Length:** 3-10 seconds recommended for speaker samples
- **Quality:** Clean audio with minimal background noise

## Advanced Usage

### Custom Output Directory

```python
# Use custom output directory
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='/path/to/custom/outputs'
)
```

### Streaming Synthesis

```python
# Use streaming for real-time applications
result = synthesizer.synthesize_simple(
    text="Long text...",
    stream=True  # Enable streaming
)
```

### Disable Text Normalization

```python
# Disable automatic text normalization
result = synthesizer.synthesize_zero_shot(
    text="Text with special formatting",
    prompt_text="...",
    prompt_audio="...",
    text_frontend=False  # Disable normalization
)
```

### Model Optimization

```python
# Load with optimizations
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    load_jit=True,   # JIT compilation
    load_trt=True,   # TensorRT (GPU only)
    fp16=True        # Half precision
)
```

## Troubleshooting

### Common Issues

1. **"Audio file not found"**
   - Check file path is correct
   - Use absolute paths or verify current directory

2. **"Failed to load model"**
   - Verify model directory exists
   - Check model files are complete

3. **"Invalid speed parameter"**
   - Speed must be between 0.5 and 2.0
   - Will auto-clamp if out of range

4. **"No audio generated"**
   - Check input text is not empty
   - Verify audio sample is valid
   - Check model loaded successfully

## Performance Tips

1. **Reuse synthesizer instance** - Don't recreate for each synthesis
2. **Use speaker profiles** - Save and reuse instead of processing audio each time
3. **Batch operations** - Use batch methods when processing multiple files
4. **Enable optimizations** - Use JIT/TRT for faster inference
5. **Streaming** - Use streaming mode for long texts

## License

Same as CosyVoice (Apache 2.0)

## Credits

Built on top of [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by Alibaba Inc.
