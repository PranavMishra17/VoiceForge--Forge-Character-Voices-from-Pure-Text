# CosyVoice Implementation Guide: Custom Speaker Embeddings & Dialogue Generation

## Table of Contents
1. [Project Requirements Verification](#project-requirements-verification)
2. [Model Architecture Deep Dive](#model-architecture-deep-dive)
3. [Installation Guide](#installation-guide)
4. [Basic Model Usage](#basic-model-usage)
5. [Custom Implementation](#custom-implementation)
6. [Speaker Embedding Extraction](#speaker-embedding-extraction)
7. [Dialogue Generation with Custom Speakers](#dialogue-generation-with-custom-speakers)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)

---

## Project Requirements Verification ✅

Your requirements are **fully supported** by CosyVoice:

1. **Extract Speaker Embeddings from Audio** ✅
   - CosyVoice uses 3D-Speaker model for speaker embedding extraction
   - Embeddings capture voice identity independent of content
   - Can be saved and reused for consistent voice generation

2. **Generate Speech with Custom Speaker Identity** ✅
   - Input: speaker embedding + text + emotion/style
   - Output: TTS audio maintaining speaker identity
   - Supports emotion control, speed, and fine-grained instructions

---

## Model Architecture Deep Dive

### Three-Stage Pipeline

```
Audio Input → Speaker Embedding (3D-Speaker)
                      ↓
Text Input → [Speech Tokenizer] → [LM] → [Flow Matching] → Audio Output
                                            ↑
                                    Speaker Embedding
```

### Key Components for Your Use Case

1. **Speaker Embedding Extractor**
   - Model: 3D-Speaker (CAM++ architecture)
   - Output: 192-dimensional speaker embedding vector
   - Captures: timbre, voice characteristics, speaker identity

2. **Text-Speech Language Model**
   - Handles semantic content (what to say)
   - Speaker-independent (no voice info here)
   - Supports emotion/style instructions

3. **Flow Matching Model**
   - Combines semantic tokens with speaker embedding
   - Generates acoustic features (how to say it)
   - Maintains voice consistency

---

## Installation Guide

### 1. System Requirements
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install sox libsox-dev git git-lfs

# CentOS/RHEL
sudo yum install sox sox-devel git git-lfs
```

### 2. Clone Repository
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
```

### 3. Create Conda Environment
```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 4. Download Models
```python
# Run this Python script to download models
from modelscope import snapshot_download

# Download CosyVoice2 0.5B model
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# Download text frontend (optional but recommended)
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

### 5. Install Optional Components
```bash
# For better text normalization (optional)
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

---

## Basic Model Usage

### Test Basic Functionality First
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Initialize model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# Test zero-shot synthesis
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_text = '希望你以后能够做的比我还好呦。'
target_text = 'Hello, this is a test of voice cloning.'

# Generate speech
for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_speech_16k, stream=False)):
    torchaudio.save('test_output_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

---

## Custom Implementation

### Phase 1: Speaker Embedding Extraction Pipeline

```python
import os
import json
import torch
import torchaudio
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class SpeakerEmbeddingExtractor:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B'):
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.embeddings_db = {}
        
    def extract_embedding_from_audio(self, audio_path, transcript, speaker_id):
        """
        Extract speaker embedding from audio file
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript of the audio
            speaker_id: Unique identifier for the speaker
        
        Returns:
            success: Boolean indicating if extraction was successful
        """
        try:
            # Load audio at 16kHz
            audio_16k = load_wav(audio_path, 16000)
            
            # Add speaker to CosyVoice's internal database
            success = self.cosyvoice.add_zero_shot_spk(
                transcript, 
                audio_16k, 
                speaker_id
            )
            
            if success:
                self.embeddings_db[speaker_id] = {
                    'audio_path': audio_path,
                    'transcript': transcript,
                    'extracted': True
                }
                
            return success
            
        except Exception as e:
            print(f"Error extracting embedding for {speaker_id}: {e}")
            return False
    
    def process_dataset(self, dataset_path, metadata_file):
        """
        Process entire dataset to extract embeddings
        
        Args:
            dataset_path: Path to dataset directory
            metadata_file: Path to JSON file with metadata
        """
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Process each audio file
        for entry in tqdm(metadata, desc="Extracting embeddings"):
            audio_path = os.path.join(dataset_path, entry['audio_file'])
            transcript = entry['transcript']
            speaker_id = entry['speaker_id']
            
            # Store additional metadata
            self.embeddings_db[speaker_id] = {
                'age': entry.get('age'),
                'gender': entry.get('gender'),
                'metadata': entry
            }
            
            # Extract embedding
            self.extract_embedding_from_audio(audio_path, transcript, speaker_id)
        
        # Save all embeddings
        self.cosyvoice.save_spkinfo()
        
        # Save metadata mapping
        with open('speaker_embeddings_metadata.json', 'w') as f:
            json.dump(self.embeddings_db, f, indent=2)

# Usage
extractor = SpeakerEmbeddingExtractor()
extractor.process_dataset('path/to/your/dataset', 'metadata.json')
```

### Phase 2: Dialogue Generation with Custom Speakers

```python
class DialogueGenerator:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B'):
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        # Load saved speaker embeddings
        self.load_speaker_embeddings()
        
    def load_speaker_embeddings(self):
        """Load previously saved speaker embeddings"""
        # CosyVoice automatically loads saved speakers from spk_info.json
        pass
    
    def generate_dialogue(self, text, speaker_id, emotion=None, style=None):
        """
        Generate speech with specified speaker identity and style
        
        Args:
            text: Text to synthesize
            speaker_id: ID of the speaker (must be previously extracted)
            emotion: Emotion instruction (e.g., 'happy', 'sad', 'angry')
            style: Style parameters (e.g., 'fast', 'slow', 'sarcastic')
        
        Returns:
            audio: Generated audio tensor
        """
        # Construct instruction if emotion/style specified
        if emotion or style:
            instruction = self._build_instruction(emotion, style)
            # Use instruct mode with speaker embedding
            audio_generator = self.cosyvoice.inference_instruct2(
                text, 
                instruction,
                None,  # No prompt speech needed, using saved embedding
                zero_shot_spk_id=speaker_id,
                stream=False
            )
        else:
            # Use zero-shot mode with saved speaker
            audio_generator = self.cosyvoice.inference_zero_shot(
                text,
                '',  # Empty prompt text
                '',  # Empty prompt speech
                zero_shot_spk_id=speaker_id,
                stream=False
            )
        
        # Get generated audio
        audio_chunks = list(audio_generator)
        if audio_chunks:
            return audio_chunks[0]['tts_speech']
        return None
    
    def _build_instruction(self, emotion=None, style=None):
        """Build natural language instruction for synthesis"""
        instructions = []
        
        # Emotion mapping
        emotion_map = {
            'happy': '用高兴的情感说',
            'sad': '用悲伤的情感说',
            'angry': '用愤怒的情感说',
            'calm': '用冷静的语气说',
            'excited': '用兴奋的语气说'
        }
        
        # Style mapping
        style_map = {
            'fast': '请说得快一些',
            'slow': '请说得慢一些',
            'sarcastic': '用讽刺的语气说',
            'formal': '用正式的语气说',
            'casual': '用随意的语气说'
        }
        
        if emotion and emotion in emotion_map:
            instructions.append(emotion_map[emotion])
        
        if style and style in style_map:
            instructions.append(style_map[style])
        
        return '，'.join(instructions) + '。' if instructions else ''
    
    def generate_dialogue_batch(self, dialogue_data):
        """
        Generate multiple dialogues
        
        Args:
            dialogue_data: List of dicts with keys: text, speaker_id, emotion, style
        
        Returns:
            List of generated audio files
        """
        results = []
        
        for idx, dialogue in enumerate(dialogue_data):
            audio = self.generate_dialogue(
                dialogue['text'],
                dialogue['speaker_id'],
                dialogue.get('emotion'),
                dialogue.get('style')
            )
            
            if audio is not None:
                # Save audio
                output_path = f"dialogue_{idx}_{dialogue['speaker_id']}.wav"
                torchaudio.save(output_path, audio, self.cosyvoice.sample_rate)
                results.append(output_path)
            else:
                results.append(None)
        
        return results

# Usage Example
generator = DialogueGenerator()

# Single dialogue generation
audio = generator.generate_dialogue(
    text="Hello, how are you today?",
    speaker_id="speaker_001",
    emotion="happy",
    style="casual"
)

# Batch generation
dialogues = [
    {
        "text": "I can't believe you did that!",
        "speaker_id": "speaker_001",
        "emotion": "angry",
        "style": "fast"
    },
    {
        "text": "Oh really? That's interesting...",
        "speaker_id": "speaker_002",
        "emotion": "calm",
        "style": "sarcastic"
    }
]

output_files = generator.generate_dialogue_batch(dialogues)
```

### Phase 3: Complete Pipeline Script

```python
# complete_pipeline.py

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='CosyVoice Custom Pipeline')
    parser.add_argument('--mode', choices=['extract', 'generate'], required=True)
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--metadata', type=str, help='Path to metadata JSON')
    parser.add_argument('--dialogue_file', type=str, help='Path to dialogue JSON for generation')
    parser.add_argument('--output_dir', type=str, default='output')
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        # Extract speaker embeddings
        extractor = SpeakerEmbeddingExtractor()
        extractor.process_dataset(args.dataset_path, args.metadata)
        print("Speaker embeddings extracted successfully!")
        
    elif args.mode == 'generate':
        # Generate dialogues
        generator = DialogueGenerator()
        
        with open(args.dialogue_file, 'r') as f:
            dialogues = json.load(f)
        
        Path(args.output_dir).mkdir(exist_ok=True)
        results = generator.generate_dialogue_batch(dialogues)
        
        print(f"Generated {len(results)} dialogue files in {args.output_dir}")

if __name__ == "__main__":
    main()
```

---

## Speaker Embedding Extraction

### Understanding Speaker Embeddings

1. **What are they?**
   - 192-dimensional vectors capturing voice characteristics
   - Independent of spoken content
   - Consistent across different utterances

2. **Quality Requirements**
   - Audio should be 3-20 seconds long
   - Clear speech without background noise
   - Single speaker only
   - 16kHz sampling rate (auto-converted)

3. **Storage Format**
   - Saved in `spk_info.json` in model directory
   - Can be exported and imported between systems
   - Lightweight (few KB per speaker)

---

## Dialogue Generation with Custom Speakers

### Input Format Example

```json
{
  "dialogues": [
    {
      "text": "Welcome to our presentation today.",
      "speaker_id": "john_doe_001",
      "emotion": "happy",
      "style": "formal"
    },
    {
      "text": "I'm excited to share our results!",
      "speaker_id": "jane_smith_002",
      "emotion": "excited",
      "style": "fast"
    }
  ]
}
```

### Fine-Grained Control

```python
# Add vocal bursts and emphasis
text_with_control = "This is [laughter] absolutely <strong>amazing</strong>!"

# Generate with fine control
audio = generator.generate_dialogue(
    text=text_with_control,
    speaker_id="speaker_001"
)
```

---

## Advanced Features

### 1. Streaming Generation
```python
# Enable streaming for real-time applications
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# Stream generation
for chunk in cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id='speaker_001', stream=True):
    # Process chunk immediately
    process_audio_chunk(chunk['tts_speech'])
```

### 2. Multi-Speaker Fine-Tuning
```python
# If you want to improve quality for specific speakers
# Prepare fine-tuning data (400+ samples per speaker)
# Use the mSFT approach with speaker tags
```

### 3. Cross-Lingual Synthesis
```python
# Maintain speaker identity across languages
audio = generator.generate_dialogue(
    text="Hello, how are you? 你好吗？",
    speaker_id="bilingual_speaker_001"
)
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use fp16 mode
   cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', fp16=True)
   ```

2. **Poor Voice Quality**
   - Ensure reference audio is high quality (16kHz, clear speech)
   - Use longer reference audio (10-20 seconds)
   - Check if transcript matches audio exactly

3. **Speaker Identity Not Consistent**
   - Verify speaker embedding was saved correctly
   - Use same speaker_id consistently
   - Consider fine-tuning for critical speakers

4. **Emotion/Style Not Working**
   - Use instructed generation mode (inference_instruct2)
   - Ensure instructions are in supported format
   - Try Chinese instructions for better results

---

## Best Practices

1. **Data Preparation**
   - Clean audio (remove noise, normalize volume)
   - Accurate transcripts (critical for embedding quality)
   - Consistent speaker labeling

2. **Embedding Extraction**
   - Use representative audio samples
   - Extract multiple embeddings per speaker and average
   - Validate embeddings with test synthesis

3. **Production Deployment**
   - Use VLLM for faster inference
   - Implement caching for frequently used speakers
   - Monitor synthesis quality with automated metrics

---

## Next Steps

1. Start with basic model testing
2. Extract embeddings from your dataset
3. Test dialogue generation with a few speakers
4. Scale up to full dataset
5. Fine-tune if needed for specific speakers
6. Optimize for production (streaming, caching, etc.)

---

## Resources

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Model Weights](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)
- [Technical Paper](https://arxiv.org/abs/2412.10117)
- [Demo Site](https://funaudiollm.github.io/cosyvoice2/)

---

## Conclusion

CosyVoice provides all the components needed for your dialogue generation system:
- ✅ Speaker embedding extraction from audio
- ✅ Consistent voice identity preservation  
- ✅ Emotion and style control
- ✅ Batch processing capabilities
- ✅ Production-ready architecture

The modular design allows you to customize each component while maintaining the overall pipeline integrity.