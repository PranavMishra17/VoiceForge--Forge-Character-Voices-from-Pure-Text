"""
Quick Zero-Shot Voice Cloning Script
Clone a voice from an audio sample and synthesize text

Usage:
    python quick_zero_shot.py
"""

import sys
sys.path.insert(0, '.')

from wrapper import VoiceSynthesizer

# ===== CONFIGURE THESE =====
TEXT_TO_SYNTHESIZE = "Hello, this is a test of voice cloning technology. I have seen many beautiful places in my life. Beaches and mountains alike."  # Text to synthesize
PROMPT_AUDIO_PATH = "examples\\i have seen.wav"  # 3-10 second audio sample
PROMPT_TEXT = "I have seen snowy mountains, stretched dizzingly to the end of sight."  # Transcription
SPEED = 1.0  # 0.5 to 2.0
OUTPUT_FILENAME = "have_seen_voice.wav"  # Optional
# ===========================

def main():
    print("\n" + "="*80)
    print("Quick Zero-Shot Voice Cloning")
    print("="*80 + "\n")

    # Initialize synthesizer
    print("Initializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir='pretrained_models/CosyVoice2-0.5B',
        output_dir='src/outputs'
    )

    # Perform synthesis
    print(f"\nCloning voice from: {PROMPT_AUDIO_PATH}")
    print(f"Synthesizing: '{TEXT_TO_SYNTHESIZE[:50]}...'")

    try:
        result = synthesizer.synthesize_zero_shot(
            text=TEXT_TO_SYNTHESIZE,
            prompt_text=PROMPT_TEXT,
            prompt_audio=PROMPT_AUDIO_PATH,
            language='en',
            speed=SPEED,
            output_filename=OUTPUT_FILENAME
        )

        print(f"\n[SUCCESS]")
        print(f"Output saved to: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"\nPlay the audio file to hear the cloned voice!")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nMake sure:")
        print("1. PROMPT_AUDIO_PATH points to a valid audio file")
        print("2. The audio is 3-10 seconds long")
        print("3. PROMPT_TEXT matches what's said in the audio")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
