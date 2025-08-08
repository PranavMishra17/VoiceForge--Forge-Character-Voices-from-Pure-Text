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
                       choices=['extract', 'synthesize', 'dialogue', 'list', 'delete', 'stats', 'add_embedding', 'export', 'import'], 
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
    parser.add_argument('--emotion', 
                       type=str, 
                       help='Emotion keyword (happy, sad, angry, excited, etc.)')
    parser.add_argument('--tone', 
                       type=str, 
                       help='Tone specification (formal, casual, dramatic, etc.)')
    parser.add_argument('--speed', 
                       type=float, 
                       default=1.0,
                       help='Speech speed multiplier (default 1.0)')
    parser.add_argument('--language', 
                       type=str, 
                       help='Language for synthesis (chinese, english, japanese, etc.)')
    
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
    
    # Direct embedding input
    parser.add_argument('--embedding_vector', 
                       type=str, 
                       help='Comma-separated 192-dimensional embedding vector')
    parser.add_argument('--embedding_file', 
                       type=str, 
                       help='Path to file containing embedding vector (JSON/NPY format)')
    
    # Import/Export
    parser.add_argument('--export_path', 
                       type=str, 
                       help='Path to export speaker embeddings')
    parser.add_argument('--import_path', 
                       type=str, 
                       help='Path to import speaker embeddings from')
    
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
            
        elif args.mode == 'add_embedding':
            add_speaker_from_embedding(voice_forge, args)
            
        elif args.mode == 'export':
            export_embeddings(voice_forge, args)
            
        elif args.mode == 'import':
            import_embeddings(voice_forge, args)
            
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
    """Synthesize speech with enhanced options."""
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
    if args.emotion:
        print(f"   Emotion: {args.emotion}")
    if args.tone:
        print(f"   Tone: {args.tone}")
    if args.speed != 1.0:
        print(f"   Speed: {args.speed}x")
    if args.language:
        print(f"   Language: {args.language}")
    
    result_path = voice_forge.synthesize_speech(
        text=args.text,
        output_filename=args.output_name,
        speaker_id=args.speaker_id,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        instruction=args.instruction,
        emotion=args.emotion,
        tone=args.tone,
        speed=args.speed,
        language=args.language
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


def add_speaker_from_embedding(voice_forge: CosyVoiceInterface, args):
    """Add speaker from direct embedding vector."""
    if not args.speaker_id:
        print("❌ --speaker_id required for add_embedding mode")
        sys.exit(1)
    
    embedding_vector = None
    
    # Load embedding from various sources
    if args.embedding_vector:
        try:
            embedding_vector = [float(x.strip()) for x in args.embedding_vector.split(',')]
        except ValueError:
            print("❌ Invalid embedding vector format. Use comma-separated floats.")
            sys.exit(1)
    
    elif args.embedding_file:
        import numpy as np
        import json
        
        embedding_file = Path(args.embedding_file)
        if not embedding_file.exists():
            print(f"❌ Embedding file not found: {embedding_file}")
            sys.exit(1)
        
        try:
            if embedding_file.suffix.lower() == '.npy':
                embedding_array = np.load(embedding_file)
                embedding_vector = embedding_array.flatten().tolist()
            elif embedding_file.suffix.lower() == '.json':
                with open(embedding_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    embedding_vector = data
                elif isinstance(data, dict) and 'embedding' in data:
                    embedding_vector = data['embedding']
                else:
                    print("❌ Invalid JSON format. Expected list or dict with 'embedding' key.")
                    sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading embedding file: {e}")
            sys.exit(1)
    
    else:
        print("❌ Either --embedding_vector or --embedding_file required for add_embedding mode")
        sys.exit(1)
    
    if len(embedding_vector) != 192:
        print(f"❌ Invalid embedding dimension: {len(embedding_vector)}. Expected 192.")
        sys.exit(1)
    
    print(f"🎭 Adding speaker from embedding...")
    print(f"   Speaker ID: {args.speaker_id}")
    print(f"   Embedding dimension: {len(embedding_vector)}")
    
    success = voice_forge.add_speaker_from_embedding(
        speaker_id=args.speaker_id,
        embedding_vector=embedding_vector,
        prompt_text=args.transcript or f"Speaker {args.speaker_id}",
        embedding_metadata={'source': 'direct_input'}
    )
    
    if success:
        print("✅ Speaker added successfully from embedding!")
    else:
        print("❌ Failed to add speaker from embedding")
        sys.exit(1)


def export_embeddings(voice_forge: CosyVoiceInterface, args):
    """Export speaker embeddings."""
    if not args.export_path:
        print("❌ --export_path required for export mode")
        sys.exit(1)
    
    print(f"📤 Exporting speaker embeddings...")
    print(f"   Export path: {args.export_path}")
    
    success = voice_forge.export_speaker_embeddings(args.export_path)
    
    if success:
        print("✅ Speaker embeddings exported successfully!")
    else:
        print("❌ Failed to export speaker embeddings")
        sys.exit(1)


def import_embeddings(voice_forge: CosyVoiceInterface, args):
    """Import speaker embeddings."""
    if not args.import_path:
        print("❌ --import_path required for import mode")
        sys.exit(1)
    
    print(f"📥 Importing speaker embeddings...")
    print(f"   Import path: {args.import_path}")
    
    imported_count = voice_forge.import_speaker_embeddings(args.import_path)
    
    if imported_count > 0:
        print(f"✅ Successfully imported {imported_count} speakers!")
    else:
        print("❌ No speakers imported")
        sys.exit(1)


def show_stats(voice_forge: CosyVoiceInterface):
    """Show interface statistics."""
    stats = voice_forge.get_stats()
    
    print("📊 VoiceForge Statistics:")
    print(f"   Model Path: {stats.get('model_path', 'N/A')}")
    print(f"   Output Directory: {stats.get('output_base_dir', 'N/A')}")
    print(f"   Speaker Database: {stats.get('speaker_database_path', 'N/A')}")
    print(f"   Total Speakers: {stats.get('speaker_count', 0)}")
    
    embedding_counts = stats.get('embedding_counts', {})
    print(f"   With Embeddings: {embedding_counts.get('with_embeddings', 0)}")
    print(f"   Without Embeddings: {embedding_counts.get('without_embeddings', 0)}")
    
    print(f"   Sample Rate: {stats.get('sample_rate', 'N/A')} Hz")
    
    features = stats.get('supported_features', {})
    print("🚀 Supported Features:")
    for feature, supported in features.items():
        status = "✅" if supported else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")


def show_examples():
    """Show usage examples."""
    examples = """
🔥 VoiceForge Enhanced Usage Examples:

📍 BASIC OPERATIONS:

1. Extract Speaker Embedding:
   python main.py --mode extract \\
     --audio samples/speaker1.wav \\
     --transcript "Hello, this is my voice sample" \\
     --speaker_id wizard_voice

2. Add Speaker from Direct Embedding:
   python main.py --mode add_embedding \\
     --speaker_id new_speaker \\
     --embedding_file embeddings/speaker.npy \\
     --transcript "Speaker description"

3. Basic Speech Synthesis:
   python main.py --mode synthesize \\
     --text "Welcome to the magical realm" \\
     --output_name wizard_greeting \\
     --speaker_id wizard_voice

📍 ENHANCED SYNTHESIS:

4. Emotion Control:
   python main.py --mode synthesize \\
     --text "I'm absolutely thrilled to meet you!" \\
     --output_name excited_speech \\
     --speaker_id wizard_voice \\
     --emotion excited \\
     --speed 1.1

5. Tone & Style Control:
   python main.py --mode synthesize \\
     --text "Please consider this proposal carefully" \\
     --output_name formal_speech \\
     --speaker_id business_voice \\
     --tone professional \\
     --emotion confident

6. Multilingual Synthesis:
   python main.py --mode synthesize \\
     --text "今天天气真好" \\
     --output_name chinese_speech \\
     --speaker_id multilingual_voice \\
     --language chinese \\
     --emotion happy

7. Combined Instruction Synthesis:
   python main.py --mode synthesize \\
     --text "The treasure lies hidden in the ancient cave" \\
     --output_name mysterious_narration \\
     --speaker_id narrator_voice \\
     --emotion mysterious \\
     --tone dramatic \\
     --instruction "whisper as if sharing a secret"

📍 DATA MANAGEMENT:

8. Export All Speaker Embeddings:
   python main.py --mode export \\
     --export_path backup/all_speakers.json

9. Import Speaker Embeddings:
   python main.py --mode import \\
     --import_path backup/all_speakers.json

10. Real-time Voice Cloning:
    python main.py --mode synthesize \\
      --text "This voice will match the reference sample" \\
      --output_name cloned_voice \\
      --prompt_audio samples/reference.wav \\
      --prompt_text "Original reference text" \\
      --speed 0.9

📍 DIALOGUE PROCESSING:

11. Enhanced Dialogue Script:
    python main.py --mode dialogue \\
      --script enhanced_story.txt \\
      --dialogue_name epic_tale \\
      --default_speaker narrator_voice

📍 MONITORING:

12. Show Enhanced Statistics:
    python main.py --mode stats

13. List All Speakers:
    python main.py --mode list

📝 Enhanced Dialogue Script Format:
   Basic: Hello, how are you?
   With speaker: [speaker:wizard_voice] Greetings, traveler!
   With emotion: [emotion:excited] I can't wait to begin our journey!
   With tone: [tone:whispering] The guards are coming...
   Combined: [speaker:elf_voice,emotion:gentle,tone:formal] Your Majesty, the realm is at peace.

🎭 Available Emotions: happy, sad, angry, excited, calm, nervous, confident, mysterious, dramatic, gentle, energetic, romantic

🎯 Available Tones: formal, casual, professional, friendly, serious, playful, authoritative, whispering, shouting

🌍 Supported Languages: chinese, english, japanese, korean, cantonese, sichuanese, shanghainese

⚡ Speed Control: Use --speed with values like 0.5 (slow), 1.0 (normal), 1.5 (fast)
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