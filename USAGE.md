# VoiceForge Usage Guide

## Quick Start - Choose Your Method

### Method 1: Interactive Menu (`main.py`) ⭐ RECOMMENDED
Run the interactive script and choose from a menu:

```bash
python main.py
```

This gives you 8 options:
1. Zero-shot voice cloning
2. Extract speaker embedding
3. Synthesize using saved embedding
4. Cross-lingual synthesis
5. Instruction-based synthesis (emotion control)
6. Voice conversion
7. Batch extract embeddings
8. List available speakers

### Method 2: Quick Scripts (Edit & Run)
Edit the configuration at the top of these files, then run:

**Clone a voice:**
```bash
python quick_zero_shot.py
```
Edit these variables in the file:
- `TEXT_TO_SYNTHESIZE` - What you want to say
- `PROMPT_AUDIO_PATH` - Path to 3-10 second audio sample
- `PROMPT_TEXT` - Transcription of the audio

**Save a voice profile:**
```bash
python quick_extract_embedding.py
```
Edit these variables:
- `AUDIO_FILE` - Path to speaker audio
- `PROMPT_TEXT` - Transcription
- `SPEAKER_ID` - Name for this voice

**Use saved voice profile:**
```bash
python quick_use_embedding.py
```
Edit these variables:
- `TEXT_TO_SYNTHESIZE` - What to say
- `SPEAKER_ID` - Name from quick_extract_embedding.py

### Method 3: Python API (For Integration)
Use in your own Python code:

```python
from wrapper import VoiceSynthesizer, SpeakerEncoder

# Initialize
synthesizer = VoiceSynthesizer(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    output_dir='src/outputs'
)

# Zero-shot cloning
result = synthesizer.synthesize_zero_shot(
    text="Text to speak",
    prompt_text="What's in the audio",
    prompt_audio="path/to/audio.wav",
    language='en',
    speed=1.0
)

print(f"Saved to: {result['output_path']}")
```

## Complete Workflows

### Workflow 1: Voice Cloning (No Saving)
**When to use:** One-time voice cloning

1. Have a 3-10 second audio sample of the target voice
2. Run `quick_zero_shot.py` (or use `main.py` option 1)
3. Audio is generated immediately

### Workflow 2: Save & Reuse Voice (Recommended)
**When to use:** You want to reuse the same voice multiple times

**Step 1 - Extract voice (do once):**
```bash
python quick_extract_embedding.py
```
Configure:
- `AUDIO_FILE = "path/to/speaker.wav"`
- `PROMPT_TEXT = "What speaker says"`
- `SPEAKER_ID = "narrator_voice"`

**Step 2 - Use saved voice (do many times):**
```bash
python quick_use_embedding.py
```
Configure:
- `TEXT_TO_SYNTHESIZE = "Any text you want"`
- `SPEAKER_ID = "narrator_voice"`  # Same as Step 1

Benefits:
- ✅ Much faster (no audio processing)
- ✅ Consistent voice across multiple files
- ✅ No need to load audio each time

### Workflow 3: Multiple Characters
**When to use:** Creating dialogue with different voices

```bash
# Extract each character's voice
python quick_extract_embedding.py
# Set SPEAKER_ID = "hero"

python quick_extract_embedding.py
# Set SPEAKER_ID = "villain"

python quick_extract_embedding.py
# Set SPEAKER_ID = "narrator"

# Generate dialogue
python quick_use_embedding.py
# Set SPEAKER_ID = "hero", TEXT = "I will save the day!"

python quick_use_embedding.py
# Set SPEAKER_ID = "villain", TEXT = "You cannot defeat me!"

python quick_use_embedding.py
# Set SPEAKER_ID = "narrator", TEXT = "And so the battle began..."
```

## All Synthesis Modes

### 1. Zero-Shot Voice Cloning
Clone any voice from a short audio sample.

**Requirements:**
- 3-10 second audio of target voice
- Transcription of what's said
- Text you want to synthesize

**Use:**
- `main.py` → Option 1
- Or `quick_zero_shot.py`
- Or API: `synthesizer.synthesize_zero_shot()`

### 2. Saved Speaker Embeddings
Use a previously extracted voice profile.

**Requirements:**
- Previously extracted speaker profile (Task 2)
- Text to synthesize

**Use:**
- `main.py` → Option 3
- Or `quick_use_embedding.py`
- Or API: `synthesizer.synthesize_with_speaker_embedding()`

### 3. Cross-Lingual Synthesis
Speak one language with another language's accent.

**Example:** English text with Chinese voice
**Requirements:**
- Audio sample (any language)
- Text in your desired language

**Use:**
- `main.py` → Option 4
- Or API: `synthesizer.synthesize_cross_lingual()`

### 4. Instruction-Based (Emotion Control)
Control emotion, style, and tone.

**Examples:**
- "Speak with excitement"
- "Whisper softly"
- "Speak angrily"
- "Calm and soothing voice"

**Requirements:**
- Audio sample
- Instruction text
- Text to synthesize

**Use:**
- `main.py` → Option 5
- Or API: `synthesizer.synthesize_instruct()`

### 5. Voice Conversion
Convert one voice to sound like another.

**Requirements:**
- Source audio (what to convert)
- Target audio (what it should sound like)

**Use:**
- `main.py` → Option 6
- Or API: `synthesizer.voice_conversion()`

## Parameters

### Speed
- Range: `0.5` to `2.0`
- Default: `1.0`
- `0.5` = half speed (slower)
- `2.0` = double speed (faster)

### Language
- `'en'` = English (default)
- `'zh'` = Chinese

### Audio Requirements
**For voice cloning samples:**
- Duration: 3-10 seconds (optimal)
- Format: WAV, MP3, FLAC (any torchaudio supports)
- Quality: Clear speech, minimal background noise
- Content: Single speaker

## Output Files

All outputs go to `src/outputs/`:

```
src/outputs/
├── tts/                      # Generated speech files
│   ├── cloned_voice.wav
│   ├── from_embedding.wav
│   └── ...
│
├── embeddings/              # Saved speaker profiles
│   ├── narrator_voice_profile.pt
│   ├── narrator_voice_profile_metadata.json
│   ├── hero_profile.pt
│   └── ...
│
└── voice_conversion/        # Voice conversion outputs
    └── converted.wav
```

## Common Issues & Solutions

### Error: "No built-in speakers available"
**Solution:** CosyVoice2-0.5B doesn't have built-in speakers.
You must:
1. Extract speaker embedding (Task 2 or `quick_extract_embedding.py`)
2. Use that embedding for synthesis (Task 3 or `quick_use_embedding.py`)

### Error: "Audio file not found"
**Solution:** Check the file path.
- Use absolute paths: `C:/Users/you/audio.wav`
- Or relative from project root: `audio_samples/voice.wav`

### Error: "Speaker 'X' not found"
**Solution:** Make sure you extracted the speaker first.
- Run `quick_extract_embedding.py` with `SPEAKER_ID = "X"`
- Or check `src/outputs/embeddings/` for available speakers

### Error: "'NoneType' object has no attribute..."
**Solution:** You're trying to use simple synthesis without speakers.
- Use zero-shot with audio instead: `quick_zero_shot.py`
- Or extract a speaker profile first

### Poor Quality Output
**Solutions:**
- Use higher quality input audio (clean, no noise)
- Use longer audio samples (5-10 seconds optimal)
- Make sure transcription matches the audio exactly
- Disable fp16 for better quality (slower):
  ```python
  synthesizer = VoiceSynthesizer(..., fp16=False)
  ```

## File Overview

**Main Scripts:**
- `main.py` - Interactive menu (EASIEST)
- `quick_zero_shot.py` - Quick voice cloning
- `quick_extract_embedding.py` - Save a voice profile
- `quick_use_embedding.py` - Use saved voice

**Wrapper Code:**
- `wrapper/synthesizer.py` - VoiceSynthesizer class
- `wrapper/encoders/speaker_encoder.py` - SpeakerEncoder class
- `wrapper/config.py` - Configuration

**Documentation:**
- `USAGE.md` - This file (quick start guide)
- `WRAPPER_USAGE_GUIDE.md` - Complete guide with all features
- `wrapper/README.md` - API documentation

**Testing:**
- `test_wrapper.py` - Test suite

## Examples

See `examples/` directory for more detailed examples (these are code documentation, not runnable demos).

## Need Help?

1. **Start simple:** Use `main.py` and follow the prompts
2. **Read errors:** Error messages tell you what's wrong
3. **Check paths:** Most errors are file path issues
4. **Try quick scripts:** Edit configuration at top of `quick_*.py` files
5. **Read full guide:** See `WRAPPER_USAGE_GUIDE.md`

## Next: Web UI

Once you're comfortable with the Python scripts, you can:
1. Create a Flask/FastAPI web server
2. Build a web UI for file uploads
3. Add REST API endpoints
4. The wrapper is ready for integration!
