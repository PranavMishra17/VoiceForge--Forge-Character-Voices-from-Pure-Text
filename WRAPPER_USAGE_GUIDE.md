# VoiceForge Wrapper - Complete Usage Guide

## Overview

The VoiceForge wrapper provides a comprehensive, production-ready interface to CosyVoice2 TTS with the following features:

### ✅ Implemented Features

1. **Voice Synthesis (VoiceSynthesizer)**
   - ✅ Simple TTS with default English voice
   - ✅ Zero-shot voice cloning from 3-10 second audio samples
   - ✅ Cross-lingual synthesis (e.g., English text with Chinese voice)
   - ✅ Instruction-based synthesis (emotion/style control)
   - ✅ Voice conversion (convert one voice to another)
   - ✅ Synthesis using pre-saved speaker embeddings
   - ✅ Full parameter control (speed: 0.5-2.0x, language, streaming)

2. **Speaker Embedding Management (SpeakerEncoder)**
   - ✅ Extract speaker embeddings from audio
   - ✅ Extract complete speaker profiles (embeddings + speech features)
   - ✅ Save and load speaker profiles
   - ✅ Batch processing for multiple audio files
   - ✅ Metadata management for embeddings

3. **Configuration & Error Handling**
   - ✅ Automatic output directory management
   - ✅ Separate folders for TTS, embeddings, voice conversion
   - ✅ Parameter validation and sanitization
   - ✅ Default language set to English
   - ✅ Comprehensive error handling with try-catch blocks
   - ✅ Detailed logging throughout

## File Structure

```
VoiceForge--Forge-Character-Voices-from-Pure-Text/
├── wrapper/
│   ├── __init__.py              # Main package exports
│   ├── config.py                # Configuration and validation
│   ├── synthesizer.py           # VoiceSynthesizer class
│   ├── README.md                # Detailed wrapper documentation
│   └── encoders/
│       ├── __init__.py
│       └── speaker_encoder.py  # SpeakerEncoder class
│
├── examples/
│   ├── demo_basic_synthesis.py      # Basic TTS examples
│   ├── demo_speaker_embedding.py    # Embedding extraction examples
│   └── demo_advanced_synthesis.py   # Advanced features examples
│
├── test_wrapper.py              # Comprehensive test suite
├── WRAPPER_USAGE_GUIDE.md       # This file
└── src/outputs/                 # Default output directory
    ├── tts/                     # TTS synthesis outputs
    ├── embeddings/              # Speaker embeddings
    └── voice_conversion/        # Voice conversion outputs
```

## Quick Start Examples

### 1. Simple TTS Synthesis

```python
from wrapper import VoiceSynthesizer

# Initialize
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs'
)

# Synthesize
result = synthesizer.synthesize_simple(
    text="Hello, this is VoiceForge speaking.",
    language='en',  # English (default)
    speed=1.0,      # Normal speed
    output_filename='hello.wav'
)

print(f"✓ Audio saved to: {result['output_path']}")
print(f"  Duration: {result['duration']:.2f} seconds")
```

### 2. Zero-Shot Voice Cloning

```python
# Clone a voice from a short audio sample
result = synthesizer.synthesize_zero_shot(
    text="This will be spoken in the cloned voice.",
    prompt_text="This is what the speaker says in the sample.",
    prompt_audio="path/to/3-10sec_audio_sample.wav",
    language='en',
    speed=1.0,
    output_filename='cloned_voice.wav'
)
```

### 3. Extract and Save Speaker Embedding

```python
from wrapper import SpeakerEncoder

# Initialize encoder
encoder = SpeakerEncoder(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs/embeddings'
)

# Extract complete speaker profile
result = encoder.extract_embedding_with_features(
    audio_input='path/to/speaker_audio.wav',
    prompt_text='Transcription of what speaker says',
    speaker_id='my_custom_voice',
    save_embedding=True
)

print(f"✓ Speaker profile saved to: {result['profile_path']}")
```

### 4. Synthesize Using Saved Speaker Embedding

```python
# Method 1: Use saved embedding file path
result = synthesizer.synthesize_with_speaker_embedding(
    text="New text in the saved voice.",
    speaker_embedding='src/outputs/embeddings/my_custom_voice_profile.pt',
    language='en',
    output_filename='from_embedding.wav'
)

# Method 2: Use speaker_id directly (faster, no file loading)
result = synthesizer.synthesize_with_speaker_embedding(
    text="Another sentence with the same voice.",
    speaker_embedding='my_custom_voice',  # speaker_id
    language='en'
)
```

### 5. Cross-Lingual Synthesis

```python
# Speak English with a Chinese voice
result = synthesizer.synthesize_cross_lingual(
    text="Hello, this is English text.",
    prompt_audio='path/to/chinese_speaker.wav',
    language='en',
    output_filename='cross_lingual.wav'
)
```

### 6. Instruction-Based Synthesis (Emotion Control)

```python
# Control emotion and style
result = synthesizer.synthesize_instruct(
    text="This is wonderful news!",
    instruct_text="Speak with excitement and joy",
    prompt_audio='path/to/speaker.wav',
    language='en',
    output_filename='excited.wav'
)

# Different emotions
emotions = [
    "Speak calmly and slowly",
    "Speak with surprise and disbelief",
    "Speak in a whisper",
    "Speak loudly and clearly"
]

for emotion in emotions:
    result = synthesizer.synthesize_instruct(
        text="The weather is nice today.",
        instruct_text=emotion,
        prompt_audio='path/to/speaker.wav'
    )
```

### 7. Voice Conversion

```python
# Convert source audio to target voice
result = synthesizer.voice_conversion(
    source_audio='path/to/original_speech.wav',
    target_audio='path/to/desired_voice.wav',
    speed=1.0,
    output_filename='converted.wav'
)
```

### 8. Batch Speaker Embedding Extraction

```python
# Extract embeddings from multiple files
audio_files = [
    'path/to/speaker1.wav',
    'path/to/speaker2.wav',
    'path/to/speaker3.wav'
]

speaker_ids = ['speaker_001', 'speaker_002', 'speaker_003']

results = encoder.batch_extract_embeddings(
    audio_files=audio_files,
    speaker_ids=speaker_ids,
    save_embeddings=True
)

for result in results:
    if 'error' not in result:
        print(f"✓ {result['speaker_id']}: {result['embedding_path']}")
```

## Complete Workflow Examples

### Workflow 1: Create Custom Voice Library

```python
from wrapper import SpeakerEncoder, VoiceSynthesizer

# Step 1: Initialize
encoder = SpeakerEncoder(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs/embeddings'
)

synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs'
)

# Step 2: Extract and save multiple voice profiles
voices = [
    ('path/to/narrator.wav', 'Sample narration text', 'narrator'),
    ('path/to/character1.wav', 'Sample dialogue', 'hero'),
    ('path/to/character2.wav', 'Another sample', 'villain'),
]

for audio_path, prompt_text, voice_id in voices:
    result = encoder.extract_embedding_with_features(
        audio_input=audio_path,
        prompt_text=prompt_text,
        speaker_id=voice_id,
        save_embedding=True
    )
    print(f"✓ Saved voice: {voice_id}")

# Step 3: Use saved voices for synthesis
script = [
    ("narrator", "The story begins on a dark and stormy night."),
    ("hero", "I must save the kingdom!"),
    ("villain", "You'll never defeat me!"),
]

for voice_id, text in script:
    result = synthesizer.synthesize_with_speaker_embedding(
        text=text,
        speaker_embedding=voice_id,
        language='en',
        output_filename=f'{voice_id}_{len(text)}.wav'
    )
    print(f"✓ Generated: {result['output_path']}")
```

### Workflow 2: Multi-Language Audiobook Production

```python
# English narration with consistent voice
narrator_profile = encoder.extract_embedding_with_features(
    audio_input='path/to/narrator_en.wav',
    prompt_text='Sample English narration',
    speaker_id='narrator_en',
    save_embedding=True
)

chapters = [
    "Chapter one: The journey begins...",
    "Chapter two: A new challenge appears...",
    "Chapter three: The final confrontation..."
]

for i, chapter_text in enumerate(chapters, 1):
    result = synthesizer.synthesize_with_speaker_embedding(
        text=chapter_text,
        speaker_embedding='narrator_en',
        language='en',
        speed=0.95,  # Slightly slower for clarity
        output_filename=f'chapter_{i}.wav'
    )
    print(f"✓ Chapter {i} generated")
```

### Workflow 3: Voice Cloning Pipeline

```python
# One-time setup: Extract profile from audio sample
profile = encoder.extract_embedding_with_features(
    audio_input='celebrity_sample.wav',
    prompt_text='Sample text from the celebrity',
    speaker_id='celebrity_voice',
    save_embedding=True
)

# Ongoing use: Generate multiple outputs with cloned voice
texts = [
    "Welcome to the show!",
    "Today we have an exciting episode.",
    "Thank you for watching!"
]

for i, text in enumerate(texts):
    result = synthesizer.synthesize_with_speaker_embedding(
        text=text,
        speaker_embedding='celebrity_voice',
        language='en',
        output_filename=f'intro_{i}.wav'
    )
```

## All Available Parameters

### VoiceSynthesizer Methods

**Common Parameters:**
- `text` (str): Text to synthesize
- `language` (str): 'en' or 'zh' (default: 'en')
- `speed` (float): 0.5 to 2.0 (default: 1.0)
- `output_filename` (str): Optional output filename
- `stream` (bool): Enable streaming synthesis (default: False)
- `text_frontend` (bool): Use text normalization (default: True)

**Method-Specific Parameters:**

1. `synthesize_zero_shot()`
   - `prompt_text` (str): Transcription of prompt audio
   - `prompt_audio` (str|Tensor): Audio sample or None
   - `speaker_id` (str): Pre-saved speaker ID (optional)

2. `synthesize_with_speaker_embedding()`
   - `speaker_embedding` (str|Tensor|dict): Path, tensor, or speaker_id

3. `synthesize_cross_lingual()`
   - `prompt_audio` (str|Tensor): Audio sample
   - `speaker_id` (str): Optional pre-saved speaker ID

4. `synthesize_instruct()`
   - `instruct_text` (str): Instruction (e.g., "speak excitedly")
   - `prompt_audio` (str|Tensor): Audio sample
   - `speaker_id` (str): Optional pre-saved speaker ID

5. `voice_conversion()`
   - `source_audio` (str|Tensor): Source audio
   - `target_audio` (str|Tensor): Target voice

### SpeakerEncoder Methods

1. `extract_embedding()`
   - `audio_input` (str|Tensor): Audio file or tensor
   - `speaker_id` (str): Optional speaker ID
   - `save_embedding` (bool): Save to disk (default: True)
   - `metadata` (dict): Optional metadata

2. `extract_embedding_with_features()` (Recommended)
   - `audio_input` (str|Tensor): Audio file or tensor
   - `prompt_text` (str): Transcription of audio
   - `speaker_id` (str): Optional speaker ID
   - `save_embedding` (bool): Save to disk (default: True)

## Error Handling

All methods include comprehensive error handling:

```python
try:
    result = synthesizer.synthesize_simple(
        text="Test",
        language='en'
    )
    print(f"Success: {result['output_path']}")

except FileNotFoundError as e:
    print(f"Audio file not found: {e}")

except ValueError as e:
    print(f"Invalid parameter: {e}")

except RuntimeError as e:
    print(f"Model error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
```

## Performance Tips

1. **Reuse synthesizer/encoder instances** - Don't recreate for each synthesis
2. **Use speaker profiles** - Extract once, use many times
3. **Batch operations** - Use batch methods for multiple files
4. **Enable model optimizations**:
   ```python
   synthesizer = VoiceSynthesizer(
       model_dir='pretrained_models/CosyVoice2-0.5B',
       load_jit=True,   # JIT compilation (faster)
       load_trt=True,   # TensorRT (GPU only, fastest)
       fp16=True        # Half precision (2x faster, slight quality loss)
   )
   ```
5. **Use streaming** - For long texts or real-time applications

## Audio Requirements

### Input Audio (for voice cloning)
- **Duration:** 3-10 seconds recommended
- **Sample Rate:** Any (will be resampled to 16kHz)
- **Format:** WAV, MP3, FLAC (any format torchaudio supports)
- **Quality:** Clean audio, minimal background noise
- **Content:** Clear speech, single speaker

### Output Audio
- **Sample Rate:** 24kHz (CosyVoice2 native)
- **Format:** WAV
- **Channels:** Mono
- **Quality:** High-quality synthetic speech

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'torch'"**
   - Solution: Activate the virtual environment or install dependencies

2. **"Model directory not found"**
   - Solution: Verify model path exists: `pretrained_models/CosyVoice2-0.5B`

3. **"Audio file not found"**
   - Solution: Use absolute paths or verify current working directory

4. **"No audio generated"**
   - Check input text is not empty
   - Verify audio sample is valid
   - Check model loaded successfully

5. **Poor quality output**
   - Use higher quality input audio
   - Avoid background noise in samples
   - Use longer audio samples (5-10 seconds)
   - Disable fp16 if quality is critical

## Next Steps: Web UI

Once the wrapper is working with Python scripts, you can:

1. Create Flask/FastAPI web server
2. Build web UI for file uploads
3. Add REST API endpoints
4. Implement real-time synthesis
5. Add voice library management UI

Example Flask integration:
```python
from flask import Flask, request, send_file
from wrapper import VoiceSynthesizer

app = Flask(__name__)
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B'
)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    result = synthesizer.synthesize_simple(text=text)
    return send_file(result['output_path'])

if __name__ == '__main__':
    app.run(debug=True)
```

## Testing

Run the comprehensive test suite:

```bash
python test_wrapper.py
```

Run example demos:

```bash
python examples/demo_basic_synthesis.py
python examples/demo_speaker_embedding.py
python examples/demo_advanced_synthesis.py
```

## Support

- See `wrapper/README.md` for detailed API documentation
- Check `examples/` directory for more usage patterns
- Review test files for additional examples

## Summary

The VoiceForge wrapper provides:

✅ **Complete TTS functionality** - All CosyVoice2 features wrapped
✅ **Speaker embedding management** - Extract, save, load, use
✅ **Multiple synthesis modes** - Simple, zero-shot, cross-lingual, instruction-based, voice conversion
✅ **Full parameter control** - Speed, language, emotion, style
✅ **Production-ready** - Error handling, logging, validation
✅ **Well-documented** - Examples, API docs, usage guides
✅ **Extensible** - Easy to add web UI or additional features

All synthesis modes work via Python scripts and are ready for integration into your web UI or other applications!
