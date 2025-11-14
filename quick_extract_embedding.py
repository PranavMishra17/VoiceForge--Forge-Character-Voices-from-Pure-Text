"""
Quick Speaker Embedding Extraction Script
Extract and save a speaker's voice profile

Usage:
    python quick_extract_embedding.py
"""

import sys
sys.path.insert(0, '.')

from wrapper import SpeakerEncoder

# ===== CONFIGURE THESE =====
AUDIO_FILE = "path/to/your/speaker_audio.wav"  # Audio of the speaker
PROMPT_TEXT = "This is what the speaker is saying in the audio."  # Transcription
SPEAKER_ID = "my_custom_voice"  # Give it a name you'll remember
# ===========================

def main():
    print("\n" + "="*80)
    print("Quick Speaker Embedding Extraction")
    print("="*80 + "\n")

    # Initialize encoder
    print("Initializing encoder...")
    encoder = SpeakerEncoder(
        model_dir='pretrained_models/CosyVoice2-0.5B',
        output_dir='src/outputs/embeddings'
    )

    # Extract embedding
    print(f"\nExtracting speaker profile from: {AUDIO_FILE}")
    print(f"Speaker ID: {SPEAKER_ID}")

    try:
        result = encoder.extract_embedding_with_features(
            audio_input=AUDIO_FILE,
            prompt_text=PROMPT_TEXT,
            speaker_id=SPEAKER_ID,
            save_embedding=True
        )

        print(f"\n✅ Success!")
        print(f"Speaker profile saved to: {result['profile_path']}")
        print(f"Speaker ID: {result['speaker_id']}")
        print(f"\nYou can now use speaker_id '{SPEAKER_ID}' in synthesis!")
        print(f"Run quick_use_embedding.py and set SPEAKER_ID = '{SPEAKER_ID}'")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. AUDIO_FILE points to a valid audio file")
        print("2. PROMPT_TEXT matches what's said in the audio")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
