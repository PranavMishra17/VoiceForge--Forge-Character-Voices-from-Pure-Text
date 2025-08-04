"""
VoiceForge - Main application entry point
External usage of CosyVoice interface
"""

import argparse
import sys
from pathlib import Path

# Import our custom interface
try:
    from CosyVoice.cosyvoice_interface import CosyVoiceInterface
except ImportError as e:
    print(f"❌ Failed to import CosyVoiceInterface: {e}")
    print("Ensure CosyVoice is properly installed and paths are correct")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='VoiceForge - Forge Character Voices from Pure Text')
    parser.add_argument('--mode', 
                       choices=['extract', 'synthesize', 'dialogue', 'list', 'delete', 'stats'], 
                       required=True, 
                       help='Operation mode')
    
    # Model and output configuration
    parser.add_argument('--model_path', 
                       type=str, 
                       default='CosyVoice/pretrained_models/CosyVoice2-0.5B',
                       help='Path to CosyVoice model')
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='voiceforge_output',
                       help='Base output directory (outside CosyVoice folder)')
    parser.add_argument('--speaker_db', 
                       type=str, 
                       help='Custom speaker database path')
    
    # Speaker extraction
    parser.add_argument('--audio', 
                       type=str, 
                       help='Audio file path for speaker extraction')
    parser.add_argument('--transcript', 
                       type=str, 
                       help='Transcript of the audio')
    parser.add_argument('--speaker_id', 
                       type=str, 
                       help='Speaker ID')
    
    # Text synthesis
    parser.add_argument('--text', 
                       type=str, 
                       help='Text to synthesize')
    parser.add_argument('--output_name', 
                       type=str, 
                       help='Output filename (without extension)')
    
    # Voice cloning options
    parser.add_argument('--prompt_audio', 
                       type=str, 
                       help='Audio prompt for real-time cloning')
    parser.add_argument('--prompt_text', 
                       type=str, 
                       default="", 
                       help='Text for audio prompt')
    
    # Emotion/style control
    parser.add_argument('--instruction', 
                       type=str, 
                       help='Natural language instruction (emotion/style)')
    
    # Dialogue processing
    parser.add_argument('--script', 
                       type=str, 
                       help='Path to dialogue script file')
    parser.add_argument('--dialogue_name', 
                       type=str, 
                       help='Name for dialogue session')
    parser.add_argument('--default_speaker', 
                       type=str, 
                       help='Default speaker for dialogue')
    
    # Logging
    parser.add_argument('--log_level', 
                       type=str, 
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()

    try:
        # Initialize VoiceForge interface
        print("🚀 Initializing VoiceForge...")
        
        voice_forge = CosyVoiceInterface(
            model_path=args.model_path,
            output_base_dir=args.output_dir,
            speaker_db_path=args.speaker_db,
            log_level=args.log_level
        )
        
        print("✅ VoiceForge initialized successfully!")
        
        # Execute based on mode
        if args.mode == 'extract':
            extract_speaker(voice_forge, args)
            
        elif args.mode == 'synthesize':
            synthesize_speech(voice_forge, args)
            
        elif args.mode == 'dialogue':
            process_dialogue(voice_forge, args)
            
        elif args.mode == 'list':
            list_speakers(voice_forge)
            
        elif args.mode == 'delete':
            delete_speaker(voice_forge, args)
            
        elif args.mode == 'stats':
            show_stats(voice_forge)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def extract_speaker(voice_forge: CosyVoiceInterface, args):
    """Extract speaker embedding from audio."""
    if not args.audio or not args.transcript or not args.speaker_id:
        print("❌ --audio, --transcript, and --speaker_id required for extract mode")
        sys.exit(1)
    
    print(f"🎙️ Extracting speaker embedding...")
    print(f"   Audio: {args.audio}")
    print(f"   Transcript: {args.transcript}")
    print(f"   Speaker ID: {args.speaker_id}")
    
    success = voice_forge.extract_speaker_embedding(
        args.audio, 
        args.transcript, 
        args.speaker_id
    )
    
    if success:
        print("✅ Speaker embedding extracted successfully!")
        print(f"📁 Speaker saved to: {voice_forge.speaker_db_path}")
    else:
        print("❌ Speaker embedding extraction failed")
        sys.exit(1)


def synthesize_speech(voice_forge: CosyVoiceInterface, args):
    """Synthesize speech with various options."""
    if not args.text or not args.output_name:
        print("❌ --text and --output_name required for synthesize mode")
        sys.exit(1)
    
    print(f"🎤 Synthesizing speech...")
    print(f"   Text: {args.text}")
    print(f"   Output: {args.output_name}")
    
    if args.speaker_id:
        print(f"   Speaker: {args.speaker_id}")
    if args.prompt_audio:
        print(f"   Prompt Audio: {args.prompt_audio}")
    if args.instruction:
        print(f"   Instruction: {args.instruction}")
    
    result_path = voice_forge.synthesize_speech(
        text=args.text,
        output_filename=args.output_name,
        speaker_id=args.speaker_id,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        instruction=args.instruction
    )
    
    if result_path:
        print(f"✅ Audio generated successfully!")
        print(f"📁 Saved to: {result_path}")
    else:
        print("❌ Speech synthesis failed")
        sys.exit(1)


def process_dialogue(voice_forge: CosyVoiceInterface, args):
    """Process dialogue script."""
    if not args.script or not args.dialogue_name:
        print("❌ --script and --dialogue_name required for dialogue mode")
        sys.exit(1)
    
    print(f"📜 Processing dialogue script...")
    print(f"   Script: {args.script}")
    print(f"   Dialogue Name: {args.dialogue_name}")
    if args.default_speaker:
        print(f"   Default Speaker: {args.default_speaker}")
    
    results = voice_forge.process_dialogue_script(
        script_path=args.script,
        dialogue_name=args.dialogue_name,
        default_speaker_id=args.default_speaker
    )
    
    if results:
        successful = sum(1 for r in results if r['output_path'])
        print(f"✅ Dialogue processing completed!")
        print(f"   Processed: {len(results)} lines")
        print(f"   Successful: {successful} lines")
        print(f"📁 Results saved to: {voice_forge.output_base_dir}/dialogue/{args.dialogue_name}")
    else:
        print("❌ Dialogue processing failed")
        sys.exit(1)


def list_speakers(voice_forge: CosyVoiceInterface):
    """List available speakers."""
    speakers = voice_forge.list_speakers()
    
    if speakers:
        print(f"👥 Available speakers ({len(speakers)}):")
        for speaker_id in speakers:
            info = voice_forge.get_speaker_info(speaker_id)
            if info:
                print(f"   🎭 {speaker_id}")
                print(f"      Audio: {info.get('audio_path', 'N/A')}")
                print(f"      Created: {info.get('created', 'N/A')}")
                print(f"      Transcript: {info.get('transcript', 'N/A')[:50]}...")
    else:
        print("📭 No speakers found")
        print("💡 Use --mode extract to add speakers")


def delete_speaker(voice_forge: CosyVoiceInterface, args):
    """Delete a speaker."""
    if not args.speaker_id:
        print("❌ --speaker_id required for delete mode")
        sys.exit(1)
    
    print(f"🗑️ Deleting speaker: {args.speaker_id}")
    
    success = voice_forge.delete_speaker(args.speaker_id)
    
    if success:
        print("✅ Speaker deleted successfully!")
    else:
        print("❌ Speaker deletion failed (speaker may not exist)")


def show_stats(voice_forge: CosyVoiceInterface):
    """Show interface statistics."""
    stats = voice_forge.get_stats()
    
    print("📊 VoiceForge Statistics:")
    print(f"   Model Path: {stats.get('model_path', 'N/A')}")
    print(f"   Output Directory: {stats.get('output_base_dir', 'N/A')}")
    print(f"   Speaker Database: {stats.get('speaker_database_path', 'N/A')}")
    print(f"   Speaker Count: {stats.get('speaker_count', 0)}")
    print(f"   Sample Rate: {stats.get('sample_rate', 'N/A')} Hz")


def show_examples():
    """Show usage examples."""
    examples = """
🔥 VoiceForge Usage Examples:

1. Extract Speaker Embedding:
   python main.py --mode extract \\
     --audio samples/speaker1.wav \\
     --transcript "Hello, this is my voice sample" \\
     --speaker_id wizard_voice

2. Synthesize with Speaker:
   python main.py --mode synthesize \\
     --text "Welcome to the magical realm" \\
     --output_name wizard_greeting \\
     --speaker_id wizard_voice

3. Synthesize with Emotion:
   python main.py --mode synthesize \\
     --text "I'm so excited to see you!" \\
     --output_name excited_greeting \\
     --speaker_id wizard_voice \\
     --instruction "speak with excitement and joy"

4. Real-time Voice Cloning:
   python main.py --mode synthesize \\
     --text "This is a test of voice cloning" \\
     --output_name cloned_voice \\
     --prompt_audio samples/reference.wav \\
     --prompt_text "Original audio content"

5. Process Dialogue Script:
   python main.py --mode dialogue \\
     --script dialogue_scripts/fantasy_story.txt \\
     --dialogue_name fantasy_story \\
     --default_speaker wizard_voice

6. List Available Speakers:
   python main.py --mode list

7. Delete Speaker:
   python main.py --mode delete --speaker_id old_speaker

8. Show Statistics:
   python main.py --mode stats

📝 Dialogue Script Format:
   Basic: Hello, how are you?
   With speaker: [speaker:wizard_voice] Greetings, traveler!
   With emotion: [instruction:speak mysteriously] The ancient secrets await...
   Combined: [speaker:elf_voice,instruction:speak gently] Take care on your journey.
"""
    print(examples)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎭 VoiceForge - Forge Character Voices from Pure Text")
        print("Run with --help for usage information")
        print("Run with --examples for usage examples")
        show_examples()
    elif '--examples' in sys.argv:
        show_examples()
    else:
        main()