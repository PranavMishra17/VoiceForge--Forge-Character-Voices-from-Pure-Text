"""
VoiceForge Main Script
Easy-to-use interface for all TTS synthesis tasks
"""

import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, os.path.abspath('.'))

from wrapper import VoiceSynthesizer, SpeakerEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION =====
# Change these parameters for your needs

MODEL_DIR = 'pretrained_models/CosyVoice2-0.5B'
OUTPUT_DIR = 'src/outputs'


def main():
    """
    Main function - choose which task to run
    """
    print("\n" + "="*80)
    print("VoiceForge - TTS Synthesis System")
    print("="*80 + "\n")

    print("Choose a task:")
    print("1. Zero-shot voice cloning (requires audio sample)")
    print("2. Extract speaker embedding from audio")
    print("3. Synthesize using saved speaker embedding")
    print("4. Cross-lingual synthesis")
    print("5. Instruction-based synthesis (emotion control)")
    print("6. Voice conversion")
    print("7. Batch extract speaker embeddings")
    print("8. List available speakers")
    print("\n0. Exit")

    choice = input("\nEnter choice (0-8): ").strip()

    if choice == '1':
        task_zero_shot_cloning()
    elif choice == '2':
        task_extract_embedding()
    elif choice == '3':
        task_synthesize_with_embedding()
    elif choice == '4':
        task_cross_lingual()
    elif choice == '5':
        task_instruction_based()
    elif choice == '6':
        task_voice_conversion()
    elif choice == '7':
        task_batch_extract()
    elif choice == '8':
        task_list_speakers()
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice!")


def task_zero_shot_cloning():
    """Task 1: Zero-shot voice cloning"""
    print("\n" + "="*80)
    print("Zero-Shot Voice Cloning")
    print("="*80)

    # Get parameters
    text = input("\nEnter text to synthesize: ").strip()
    if not text:
        print("Error: Text cannot be empty")
        return

    prompt_audio = input("Enter path to audio sample (3-10 seconds): ").strip()
    if not os.path.exists(prompt_audio):
        print(f"Error: Audio file not found: {prompt_audio}")
        return

    prompt_text = input("Enter transcription of the audio: ").strip()
    if not prompt_text:
        print("Error: Transcription cannot be empty")
        return

    speed = input("Enter speed (0.5-2.0, default 1.0): ").strip()
    speed = float(speed) if speed else 1.0

    output_filename = input("Enter output filename (optional): ").strip()
    output_filename = output_filename if output_filename else None

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # Perform synthesis
    print("\nSynthesizing...")
    try:
        result = synthesizer.synthesize_zero_shot(
            text=text,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            language='en',
            speed=speed,
            output_filename=output_filename
        )

        print(f"\n✅ Success!")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_extract_embedding():
    """Task 2: Extract speaker embedding"""
    print("\n" + "="*80)
    print("Extract Speaker Embedding")
    print("="*80)

    audio_path = input("\nEnter path to audio file: ").strip()
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    prompt_text = input("Enter transcription of the audio: ").strip()
    if not prompt_text:
        print("Error: Transcription cannot be empty")
        return

    speaker_id = input("Enter speaker ID (e.g., my_voice): ").strip()
    if not speaker_id:
        print("Error: Speaker ID cannot be empty")
        return

    # Initialize encoder
    print("\nInitializing encoder...")
    encoder = SpeakerEncoder(
        model_dir=MODEL_DIR,
        output_dir=os.path.join(OUTPUT_DIR, 'embeddings')
    )

    # Extract embedding
    print("\nExtracting embedding...")
    try:
        result = encoder.extract_embedding_with_features(
            audio_input=audio_path,
            prompt_text=prompt_text,
            speaker_id=speaker_id,
            save_embedding=True
        )

        print(f"\n✅ Success!")
        print(f"Speaker profile saved to: {result['profile_path']}")
        print(f"Speaker ID: {result['speaker_id']}")
        print(f"\nYou can now use speaker_id '{speaker_id}' for synthesis!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_synthesize_with_embedding():
    """Task 3: Synthesize using saved embedding"""
    print("\n" + "="*80)
    print("Synthesize with Saved Speaker Embedding")
    print("="*80)

    text = input("\nEnter text to synthesize: ").strip()
    if not text:
        print("Error: Text cannot be empty")
        return

    print("\nChoose input method:")
    print("1. Speaker ID (from previously extracted embedding)")
    print("2. Path to .pt embedding file")

    method = input("Choose (1 or 2): ").strip()

    if method == '1':
        speaker_embedding = input("Enter speaker ID: ").strip()
    elif method == '2':
        speaker_embedding = input("Enter path to .pt file: ").strip()
        if not os.path.exists(speaker_embedding):
            print(f"Error: File not found: {speaker_embedding}")
            return
    else:
        print("Invalid choice")
        return

    speed = input("Enter speed (0.5-2.0, default 1.0): ").strip()
    speed = float(speed) if speed else 1.0

    output_filename = input("Enter output filename (optional): ").strip()
    output_filename = output_filename if output_filename else None

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # Perform synthesis
    print("\nSynthesizing...")
    try:
        result = synthesizer.synthesize_with_speaker_embedding(
            text=text,
            speaker_embedding=speaker_embedding,
            language='en',
            speed=speed,
            output_filename=output_filename
        )

        print(f"\n✅ Success!")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_cross_lingual():
    """Task 4: Cross-lingual synthesis"""
    print("\n" + "="*80)
    print("Cross-Lingual Synthesis")
    print("="*80)

    text = input("\nEnter text to synthesize: ").strip()
    if not text:
        print("Error: Text cannot be empty")
        return

    prompt_audio = input("Enter path to audio sample: ").strip()
    if not os.path.exists(prompt_audio):
        print(f"Error: Audio file not found: {prompt_audio}")
        return

    language = input("Enter text language (en/zh, default en): ").strip()
    language = language if language else 'en'

    speed = input("Enter speed (0.5-2.0, default 1.0): ").strip()
    speed = float(speed) if speed else 1.0

    output_filename = input("Enter output filename (optional): ").strip()
    output_filename = output_filename if output_filename else None

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # Perform synthesis
    print("\nSynthesizing...")
    try:
        result = synthesizer.synthesize_cross_lingual(
            text=text,
            prompt_audio=prompt_audio,
            language=language,
            speed=speed,
            output_filename=output_filename
        )

        print(f"\n✅ Success!")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_instruction_based():
    """Task 5: Instruction-based synthesis"""
    print("\n" + "="*80)
    print("Instruction-Based Synthesis (Emotion Control)")
    print("="*80)

    text = input("\nEnter text to synthesize: ").strip()
    if not text:
        print("Error: Text cannot be empty")
        return

    instruct_text = input("Enter instruction (e.g., 'speak excitedly'): ").strip()
    if not instruct_text:
        print("Error: Instruction cannot be empty")
        return

    prompt_audio = input("Enter path to audio sample: ").strip()
    if not os.path.exists(prompt_audio):
        print(f"Error: Audio file not found: {prompt_audio}")
        return

    speed = input("Enter speed (0.5-2.0, default 1.0): ").strip()
    speed = float(speed) if speed else 1.0

    output_filename = input("Enter output filename (optional): ").strip()
    output_filename = output_filename if output_filename else None

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # Perform synthesis
    print("\nSynthesizing...")
    try:
        result = synthesizer.synthesize_instruct(
            text=text,
            instruct_text=instruct_text,
            prompt_audio=prompt_audio,
            language='en',
            speed=speed,
            output_filename=output_filename
        )

        print(f"\n✅ Success!")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_voice_conversion():
    """Task 6: Voice conversion"""
    print("\n" + "="*80)
    print("Voice Conversion")
    print("="*80)

    source_audio = input("\nEnter path to source audio: ").strip()
    if not os.path.exists(source_audio):
        print(f"Error: Source audio not found: {source_audio}")
        return

    target_audio = input("Enter path to target voice audio: ").strip()
    if not os.path.exists(target_audio):
        print(f"Error: Target audio not found: {target_audio}")
        return

    speed = input("Enter speed (0.5-2.0, default 1.0): ").strip()
    speed = float(speed) if speed else 1.0

    output_filename = input("Enter output filename (optional): ").strip()
    output_filename = output_filename if output_filename else None

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # Perform conversion
    print("\nConverting...")
    try:
        result = synthesizer.voice_conversion(
            source_audio=source_audio,
            target_audio=target_audio,
            speed=speed,
            output_filename=output_filename
        )

        print(f"\n✅ Success!")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_batch_extract():
    """Task 7: Batch extract embeddings"""
    print("\n" + "="*80)
    print("Batch Extract Speaker Embeddings")
    print("="*80)

    print("\nEnter audio file paths (one per line, empty line to finish):")
    audio_files = []
    while True:
        path = input("> ").strip()
        if not path:
            break
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
        audio_files.append(path)

    if not audio_files:
        print("No audio files provided")
        return

    print(f"\nFound {len(audio_files)} audio files")

    # Initialize encoder
    print("\nInitializing encoder...")
    encoder = SpeakerEncoder(
        model_dir=MODEL_DIR,
        output_dir=os.path.join(OUTPUT_DIR, 'embeddings')
    )

    # Extract embeddings
    print("\nExtracting embeddings...")
    try:
        results = encoder.batch_extract_embeddings(
            audio_files=audio_files,
            speaker_ids=None,  # Auto-generate IDs
            save_embeddings=True
        )

        print(f"\n✅ Batch extraction complete!")
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                print(f"{i}. {result['speaker_id']}: {result['embedding_path']}")
            else:
                print(f"{i}. Failed: {result['error']}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def task_list_speakers():
    """Task 8: List available speakers"""
    print("\n" + "="*80)
    print("List Available Speakers")
    print("="*80)

    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = VoiceSynthesizer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

    speakers = synthesizer.list_available_speakers()

    if speakers:
        print(f"\n✅ Found {len(speakers)} speakers:")
        for i, speaker_id in enumerate(speakers, 1):
            print(f"{i}. {speaker_id}")
    else:
        print("\n⚠ No built-in speakers found in the model.")
        print("You need to:")
        print("1. Extract speaker embeddings from audio (Task 2)")
        print("2. Then use those embeddings for synthesis (Task 3)")


if __name__ == "__main__":
    try:
        while True:
            main()
            print("\n")
            again = input("Do another task? (y/n): ").strip().lower()
            if again != 'y':
                break
        print("\nGoodbye!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
