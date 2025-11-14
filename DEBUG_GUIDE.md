# Debug Guide - Gibberish Speech Issue

## Current Status
Speech output is still gibberish despite implementing official audio preprocessing.

## Debug Script Created
`debug_exact_webui.py` - Replicates EXACTLY what webui.py does with detailed logging at every step

### Run it:
```bash
python debug_exact_webui.py
```

This will show:
- How audio is loaded and preprocessed
- Exact shapes and values at each step
- What goes into the model
- What comes out of the model
- Audio quality metrics

## About ttsfrd Warning

**The ttsfrd import failure is NORMAL and NOT the problem:**

```
failed to import ttsfrd, use wetext instead
```

**Why it fails:**
- The `.whl` files in `pretrained_models/CosyVoice-ttsfrd` are Linux-only (`linux_x86_64`)
- You're on Windows
- There's no Windows build of ttsfrd

**Why it's okay:**
- The code automatically falls back to `wetext` for text normalization
- `wetext` works perfectly fine
- Official CosyVoice supports both ttsfrd and wetext
- The webui.py code has this exact fallback (lines 27-34 in cosyvoice/cosyvoice/cli/frontend.py)

**If you really want ttsfrd:**
- You'd need to compile it for Windows yourself
- Or use WSL (Windows Subsystem for Linux)
- But it's NOT necessary - wetext is fine!

## Requirements Check

### Python Version
CosyVoice2 officially supports Python 3.8-3.11. You don't need exactly 3.10.

Check your version:
```bash
python --version
```

### FFmpeg
**NOT required** for basic synthesis. FFmpeg is only needed if:
- You want to use certain audio codecs
- You're doing video processing
- You need to load non-standard audio formats

For WAV files (which we're using), FFmpeg is NOT needed.

## What to Look For in Debug Output

Run `debug_exact_webui.py` and check:

### 1. Audio Loading
```
POSTPROCESS - Input
  Input shape: torch.Size([1, XXXXX])
  Input min/max: should be reasonable values (not all zeros)
  Input mean/std: should have variance (not flat)
```

### 2. After Postprocess
```
POSTPROCESS - Output
  Output shape: torch.Size([1, XXXXX])  (might be smaller due to trim)
  Output min/max: should be within [-0.8, 0.8] due to normalization
```

### 3. Model Output
```
Chunk 1:
  tts_speech min/max: should have variance
  tts_speech mean/std: should not be near zero
```

### 4. Audio Quality Check
```
STEP 7: Audio quality check
  Variance: should be > 0.001 (if lower, audio is too flat)
  Abs mean: should be > 0.001 (if lower, audio is too quiet/silent)
```

## Common Issues to Check

### Issue 1: Flat Audio (All Zeros)
**Symptoms:**
- Variance < 0.001
- Abs mean < 0.001

**Possible causes:**
- Audio not loading correctly
- Postprocessing removing all signal
- Model not generating properly

### Issue 2: Clipped Audio
**Symptoms:**
- Min/max at exactly -1.0 or 1.0
- Distorted sound

**Possible causes:**
- Normalization not working
- Amplitude too high

### Issue 3: Wrong Sample Rate
**Symptoms:**
- Audio plays too fast or too slow
- Pitch is wrong

**Possible causes:**
- Sample rate mismatch
- Resampling error

## Next Steps After Running Debug Script

1. **Share the debug output** - Especially:
   - POSTPROCESS sections
   - Model output sections
   - Audio quality check section

2. **Check the generated `debug_output.wav`**
   - Is it gibberish?
   - Is it silent?
   - Is it distorted?

3. **Compare with working example**
   - Try running official webui.py:
     ```bash
     python cosyvoice/webui.py --model_dir pretrained_models/CosyVoice2-0.5B
     ```
   - Go to http://localhost:8000
   - Try same audio and text
   - Does that work?

## If Official WebUI Works But Wrapper Doesn't

This would tell us the difference is in HOW we're calling the model, not the model itself.

We'd need to check:
- Parameter order
- Parameter types (tensor vs numpy, device placement)
- Text preprocessing differences
- Any hidden state or caching issues

## If Official WebUI Also Produces Gibberish

This would indicate:
- Model file issue
- Missing dependencies
- Installation problem
- Hardware compatibility issue

Let me know the output from `debug_exact_webui.py` and whether the official webui works!
