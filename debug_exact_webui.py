"""
DEBUG SCRIPT - Exact webui.py replication with detailed logging
This script mimics EXACTLY what webui.py does for zero-shot synthesis
"""

import sys
import os
import numpy as np
import torch
import torchaudio
import librosa

# Setup paths
sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
MODEL_DIR = 'pretrained_models/CosyVoice2-0.5B'
TTS_TEXT = "Hello, this is a test of voice cloning technology."
PROMPT_TEXT = "I have seen many beautiful places in my life."
PROMPT_AUDIO = "examples/i have seen.wav"
SEED = 0
SPEED = 1.0
STREAM = False
PROMPT_SR = 16000
MAX_VAL = 0.8
# =========================


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """Exact copy from webui.py lines 46-55"""
    logger.info("="*80)
    logger.info("POSTPROCESS - Input")
    logger.info(f"  Input shape: {speech.shape}")
    logger.info(f"  Input dtype: {speech.dtype}")
    logger.info(f"  Input device: {speech.device}")
    logger.info(f"  Input min/max: {speech.min():.6f} / {speech.max():.6f}")
    logger.info(f"  Input mean/std: {speech.mean():.6f} / {speech.std():.6f}")

    # Trim silence
    speech_np = speech.squeeze().cpu().numpy()
    logger.info(f"  After squeeze numpy: shape={speech_np.shape}")

    speech_trimmed, intervals = librosa.effects.trim(
        speech_np,
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    logger.info(f"  After trim: shape={speech_trimmed.shape}")
    logger.info(f"  Trim intervals: {intervals}")

    speech = torch.from_numpy(speech_trimmed).unsqueeze(0)
    logger.info(f"  After torch convert: shape={speech.shape}")

    # Normalize
    abs_max = speech.abs().max()
    logger.info(f"  Abs max before normalize: {abs_max:.6f}")
    if abs_max > MAX_VAL:
        speech = speech / abs_max * MAX_VAL
        logger.info(f"  Normalized to {MAX_VAL}")

    # Add silence padding (as in webui.py line 54)
    # Note: webui.py uses cosyvoice.sample_rate which is 24000 for CosyVoice2-0.5B
    # But for prompt audio, we're at 16000 Hz, so no padding needed here
    # The padding is added to OUTPUT, not input

    logger.info("POSTPROCESS - Output")
    logger.info(f"  Output shape: {speech.shape}")
    logger.info(f"  Output min/max: {speech.min():.6f} / {speech.max():.6f}")
    logger.info(f"  Output mean/std: {speech.mean():.6f} / {speech.std():.6f}")
    logger.info("="*80 + "\n")

    return speech


def main():
    print("\n" + "="*80)
    print("DEBUG: Exact webui.py Zero-Shot Synthesis Replication")
    print("="*80 + "\n")

    # Step 1: Load model (webui.py lines 187-193)
    logger.info("STEP 1: Loading model")
    logger.info(f"  Model dir: {MODEL_DIR}")
    try:
        cosyvoice = CosyVoice2(MODEL_DIR)
        logger.info(f"  Model loaded successfully")
        logger.info(f"  Sample rate: {cosyvoice.sample_rate}")
        logger.info(f"  Instruct mode: {cosyvoice.instruct}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Step 2: Load and preprocess prompt audio (webui.py line 121)
    logger.info("\nSTEP 2: Load prompt audio")
    logger.info(f"  Prompt audio path: {PROMPT_AUDIO}")
    logger.info(f"  Target sample rate: {PROMPT_SR}")

    logger.info("\n--- Calling load_wav ---")
    prompt_speech_16k = load_wav(PROMPT_AUDIO, PROMPT_SR)
    logger.info(f"  After load_wav:")
    logger.info(f"    Shape: {prompt_speech_16k.shape}")
    logger.info(f"    Dtype: {prompt_speech_16k.dtype}")
    logger.info(f"    Device: {prompt_speech_16k.device}")
    logger.info(f"    Min/max: {prompt_speech_16k.min():.6f} / {prompt_speech_16k.max():.6f}")
    logger.info(f"    Mean/std: {prompt_speech_16k.mean():.6f} / {prompt_speech_16k.std():.6f}")

    logger.info("\n--- Calling postprocess ---")
    prompt_speech_16k = postprocess(prompt_speech_16k)

    # Step 3: Set random seed (webui.py line 122)
    logger.info("\nSTEP 3: Set random seed")
    logger.info(f"  Seed: {SEED}")
    set_all_random_seed(SEED)
    logger.info("  Random seed set for: random, numpy, torch, torch.cuda")

    # Step 4: Text normalization check
    logger.info("\nSTEP 4: Text inputs")
    logger.info(f"  TTS text: '{TTS_TEXT}'")
    logger.info(f"  TTS text length: {len(TTS_TEXT)}")
    logger.info(f"  Prompt text: '{PROMPT_TEXT}'")
    logger.info(f"  Prompt text length: {len(PROMPT_TEXT)}")

    # Step 5: Call inference_zero_shot (webui.py line 123)
    logger.info("\nSTEP 5: Calling inference_zero_shot")
    logger.info(f"  Parameters:")
    logger.info(f"    tts_text: '{TTS_TEXT}'")
    logger.info(f"    prompt_text: '{PROMPT_TEXT}'")
    logger.info(f"    prompt_speech_16k: shape={prompt_speech_16k.shape}")
    logger.info(f"    stream: {STREAM}")
    logger.info(f"    speed: {SPEED}")

    logger.info("\n" + "="*80)
    logger.info("Starting inference...")
    logger.info("="*80 + "\n")

    outputs = []
    try:
        for idx, output in enumerate(cosyvoice.inference_zero_shot(
            TTS_TEXT,
            PROMPT_TEXT,
            prompt_speech_16k,
            stream=STREAM,
            speed=SPEED
        )):
            logger.info(f"\nChunk {idx + 1}:")
            logger.info(f"  Keys in output: {output.keys()}")
            tts_speech = output['tts_speech']
            logger.info(f"  tts_speech shape: {tts_speech.shape}")
            logger.info(f"  tts_speech dtype: {tts_speech.dtype}")
            logger.info(f"  tts_speech device: {tts_speech.device}")
            logger.info(f"  tts_speech min/max: {tts_speech.min():.6f} / {tts_speech.max():.6f}")
            logger.info(f"  tts_speech mean/std: {tts_speech.mean():.6f} / {tts_speech.std():.6f}")
            logger.info(f"  Duration: {tts_speech.shape[1] / cosyvoice.sample_rate:.2f}s")

            outputs.append(tts_speech)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 6: Save output
    logger.info("\nSTEP 6: Saving output")
    if outputs:
        final_audio = torch.cat(outputs, dim=1)
        logger.info(f"  Final concatenated audio:")
        logger.info(f"    Shape: {final_audio.shape}")
        logger.info(f"    Duration: {final_audio.shape[1] / cosyvoice.sample_rate:.2f}s")
        logger.info(f"    Sample rate: {cosyvoice.sample_rate}")
        logger.info(f"    Min/max: {final_audio.min():.6f} / {final_audio.max():.6f}")
        logger.info(f"    Mean/std: {final_audio.mean():.6f} / {final_audio.std():.6f}")

        output_path = 'debug_output.wav'
        torchaudio.save(output_path, final_audio, cosyvoice.sample_rate)
        logger.info(f"  Saved to: {output_path}")

        print("\n" + "="*80)
        print(f"SUCCESS! Audio saved to: {output_path}")
        print(f"Duration: {final_audio.shape[1] / cosyvoice.sample_rate:.2f}s")
        print("="*80 + "\n")
    else:
        logger.error("No audio generated!")

    # Step 7: Check if output contains actual speech or gibberish
    logger.info("\nSTEP 7: Audio quality check")
    if outputs:
        # Check for signs of gibberish:
        # 1. Very low variance (flat audio)
        # 2. Values all near zero
        # 3. Repetitive patterns

        variance = final_audio.var().item()
        abs_mean = final_audio.abs().mean().item()

        logger.info(f"  Variance: {variance:.6f}")
        logger.info(f"  Abs mean: {abs_mean:.6f}")

        if variance < 0.001:
            logger.warning("  WARNING: Very low variance - might be flat/silent")
        if abs_mean < 0.001:
            logger.warning("  WARNING: Very low amplitude - might be silent")

        # Sample check
        sample_segment = final_audio[0, :min(1000, final_audio.shape[1])]
        logger.info(f"  First 1000 samples stats:")
        logger.info(f"    Min/max: {sample_segment.min():.6f} / {sample_segment.max():.6f}")
        logger.info(f"    Unique values: {len(torch.unique(sample_segment))}")


if __name__ == "__main__":
    main()
