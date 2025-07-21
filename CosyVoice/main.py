import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse

sys.path.append('third_party/Matcha-TTS')
sys.path.append('CosyVoice')

from speaker_embedding_extractor import SpeakerEmbeddingExtractor
from cosyvoice_tts import CosyVoiceTTS

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='CosyVoice Main Entrypoint')
    parser.add_argument('--mode', choices=['tts', 'embedding'], required=True, help='Mode to run: tts (text-to-speech), embedding (extractor only)')
    parser.add_argument('--audio_folder', type=str, help='Path to folder containing audio files (wav/mp3)')
    parser.add_argument('--text', type=str, help='Text to synthesize (for tts mode)')
    parser.add_argument('--output_path', type=str, help='Output path for synthesized audio (for tts mode)')
    parser.add_argument('--prompt_text', type=str, help='Prompt text (for tts mode)')
    parser.add_argument('--prompt_audio', type=str, help='Prompt audio file (for tts mode)')
    parser.add_argument('--speaker_id', type=str, help='Speaker ID (for tts mode)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("[DEBUG] Logging enabled")

    try:
        if args.mode == 'embedding':
            if not args.audio_folder:
                print("[ERROR] --audio_folder is required for embedding mode.")
                sys.exit(1)
            extractor = SpeakerEmbeddingExtractor()
            extractor.extract_from_folder(args.audio_folder)
            print(f"[INFO] Embeddings and metadata saved to output/embeddings/")
        elif args.mode == 'tts':
            if not args.text or not args.output_path:
                print("[ERROR] --text and --output_path are required for tts mode.")
                sys.exit(1)
            tts = CosyVoiceTTS()
            success = tts.synthesize(
                text=args.text,
                output_path=args.output_path,
                prompt_text=args.prompt_text,
                prompt_audio_path=args.prompt_audio,
                speaker_id=args.speaker_id
            )
            if success:
                print(f"[INFO] TTS audio generated at {args.output_path}_output_0.wav")
            else:
                print(f"[ERROR] TTS synthesis failed.")
        else:
            print(f"[ERROR] Unknown mode: {args.mode}")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"[ERROR] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
