"""
Quick Synthesis with Saved Embedding Script
Synthesize text using a previously saved speaker embedding

Usage:
    1. First run quick_extract_embedding.py to create a speaker profile
    2. Then run this script with the same SPEAKER_ID
"""

import sys
sys.path.insert(0, '.')

from wrapper import VoiceSynthesizer

# ===== CONFIGURE THESE =====
TEXT_TO_SYNTHESIZE = "This text will be spoken in the saved custom voice."
SPEAKER_ID = "my_custom_voice"  # Must match the ID from quick_extract_embedding.py
SPEED = 1.0  # 0.5 to 2.0
OUTPUT_FILENAME = "from_embedding.wav"  # Optional
# ===========================

def main():
    print("\n" + "="*80)
    print("Quick Synthesis with Saved Embedding")
    print("="*80 + "\n")

    # Initialize synthesizer
    print("Initializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir='pretrained_models/CosyVoice2-0.5B',
        output_dir='src/outputs'
    )

    # Perform synthesis
    print(f"\nUsing speaker: {SPEAKER_ID}")
    print(f"Synthesizing: '{TEXT_TO_SYNTHESIZE[:50]}...'")

    try:
        result = synthesizer.synthesize_with_speaker_embedding(
            text=TEXT_TO_SYNTHESIZE,
            speaker_embedding=SPEAKER_ID,  # Use the saved speaker ID
            language='en',
            speed=SPEED,
            output_filename=OUTPUT_FILENAME
        )

        print(f"\n✅ Success!")
        print(f"Output saved to: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Speaker used: {result['speaker_id']}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"\nMake sure:")
        print(f"1. You have extracted a speaker embedding with ID '{SPEAKER_ID}'")
        print(f"2. Run quick_extract_embedding.py first if you haven't")
        print(f"3. Or check src/outputs/embeddings/ for available speaker IDs")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
