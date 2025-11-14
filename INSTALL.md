# VoiceForge Installation Guide

Complete setup guide for VoiceForge - a custom wrapper over CosyVoice 2.0 for TTS with custom speaker embeddings.

## Prerequisites

- Python 3.10
- Conda (Miniconda or Anaconda)
- Git with submodules support
- 10GB+ free disk space for models
- Windows/Linux/Mac OS

## Installation Steps

### 1. Clone Repository

```bash
git clone <your-repo-url> VoiceForge2
cd VoiceForge2
```

### 2. Initialize CosyVoice Submodule

```bash
git submodule update --init --recursive
```

This will clone CosyVoice and its nested submodules (Matcha-TTS).

### 3. Create Conda Environment

**Environment Name:** VoiceForge

```bash
conda create -n VoiceForge python=3.10 -y
conda activate VoiceForge
```

### 4. Install Python Dependencies

**Windows:**
```bash
pip install -r cosyvoice\requirements.txt
```

**Linux/Mac:**
```bash
pip install -r cosyvoice/requirements.txt
```

### 5. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install sox libsox-dev
```

**CentOS/RHEL:**
```bash
sudo yum install sox sox-devel
```

**Windows:**
- Download sox from https://sourceforge.net/projects/sox/files/sox/
- Extract and add to system PATH
- Or skip if not using audio processing features

### 6. Download Pretrained Models

**Create models directory:**
```bash
mkdir pretrained_models
```

**Download CosyVoice2-0.5B (Recommended):**

Using Python SDK:
```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
```

Using git (requires git-lfs):
```bash
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
```

**Download ttsfrd Resource (Optional but recommended):**
```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')"
```

### 7. Install ttsfrd (Optional - Better Text Normalization)

**Linux Only:**
```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
cd ../..
```

**Note:** Windows users will use WeText by default (no action needed).

### 8. Verify Installation

Test CosyVoice2 basic functionality:

```python
import sys
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
print("CosyVoice2 loaded successfully!")
print(f"Sample rate: {cosyvoice.sample_rate}")
```

## Project Structure

```
VoiceForge2/
├── cosyvoice/                  # CosyVoice submodule (DO NOT MODIFY)
│   ├── third_party/
│   │   └── Matcha-TTS/        # Nested submodule
│   ├── cosyvoice/             # Core CosyVoice code
│   └── requirements.txt
├── wrapper/                    # Your custom wrapper code
│   ├── ui/                    # UI components (Gradio/Streamlit)
│   ├── encoders/              # Speaker embedding management
│   ├── api/                   # API wrappers
│   └── __init__.py
├── pretrained_models/         # Downloaded models (gitignored)
│   ├── CosyVoice2-0.5B/
│   └── CosyVoice-ttsfrd/
├── .gitignore
├── requirements.txt           # Your additional dependencies
├── config.py                  # Your configurations
├── INSTALL.md                 # This file
└── README.md
```

## Additional Dependencies for Wrapper

Install after base setup:

```bash
pip install gradio flask fastapi uvicorn streamlit
```

For speaker encoder storage:
```bash
pip install numpy scipy librosa
```

## Troubleshooting

### Issue: Submodule failed to clone
**Solution:**
```bash
cd cosyvoice
git submodule update --init --recursive
cd ..
```

### Issue: Model download fails
**Solution:**
- Check internet connection
- Try git clone method instead of Python SDK
- Ensure sufficient disk space

### Issue: Sox not found (Linux)
**Solution:**
```bash
sudo apt-get install sox libsox-dev
# or
sudo yum install sox sox-devel
```

### Issue: Import errors
**Solution:**
Ensure paths are added:
```python
import sys
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')
```

### Issue: CUDA/GPU errors
**Solution:**
- Install PyTorch with CUDA support
- Check CUDA compatibility with your GPU
- Use CPU mode by setting appropriate flags

## Environment Management

**Activate environment:**
```bash
conda activate VoiceForge
```

**Deactivate environment:**
```bash
conda deactivate
```

**Delete environment (if needed):**
```bash
conda env remove -n VoiceForge
```

## Updating CosyVoice

To update the CosyVoice submodule to latest version:

```bash
cd cosyvoice
git pull origin main
git submodule update --init --recursive
cd ..
git add cosyvoice
git commit -m "Update CosyVoice submodule"
```

## Notes

- **Never modify files inside `cosyvoice/` directory** - it's a submodule
- Models are large (5-10GB) - ensure adequate storage
- First run will be slower due to model initialization
- GPU recommended for real-time inference
- `pretrained_models/` should be in `.gitignore`

## Quick Start After Installation

```python
import sys
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# Initialize model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# Zero-shot voice cloning
prompt_speech = load_wav('path/to/voice_sample.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    'Your text here', 
    'Prompt text matching the voice sample', 
    prompt_speech, 
    stream=False
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## Support

- CosyVoice GitHub: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice Issues: https://github.com/FunAudioLLM/CosyVoice/issues
- VoiceForge Issues: <your-repo-issues-url>