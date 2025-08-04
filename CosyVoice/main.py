import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse
import torchaudio
import json
import numpy as np

# Add paths
sys.path.append('third_party/Matcha-TTS')
sys.path.append('CosyVoice')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

class CosyVoice2TTS:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B'):
        """Initialize CosyVoice2 model"""
        try:
            self.model_path = Path(model_path)
            self.speaker_db_path = self.model_path / 'custom_speakers.json'
            
            self.cosyvoice = CosyVoice2(
                model_path, 
                load_jit=False, 
                load_trt=False, 
                load_vllm=False, 
                fp16=False
            )
            
            # Load existing speakers
            self._load_custom_speakers()
            
            logging.info("CosyVoice2 model initialized successfully")
            print("✅ CosyVoice2 model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise e

    def _load_custom_speakers(self):
        """Load previously saved speaker embeddings"""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r') as f:
                    speaker_data = json.load(f)
                
                loaded_speakers = []
                for speaker_id, data in speaker_data.items():
                    audio_path = data['audio_path']
                    transcript = data['transcript']
                    
                    if Path(audio_path).exists():
                        audio_16k = load_wav(audio_path, 16000)
                        success = self.cosyvoice.add_zero_shot_spk(transcript, audio_16k, speaker_id)
                        if success:
                            loaded_speakers.append(speaker_id)
                
                if loaded_speakers:
                    print(f"✅ Loaded speakers: {loaded_speakers}")
                else:
                    print("ℹ️ No existing speakers found")
            else:
                print("ℹ️ No speaker database found")
                
        except Exception as e:
            print(f"⚠️ Error loading speakers: {e}")

    def _save_custom_speakers(self, speaker_id, audio_path, transcript):
        """Save speaker info for persistence"""
        try:
            speaker_data = {}
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r') as f:
                    speaker_data = json.load(f)
            
            speaker_data[speaker_id] = {
                'audio_path': str(audio_path),
                'transcript': transcript,
                'created': datetime.now().isoformat()
            }
            
            with open(self.speaker_db_path, 'w') as f:
                json.dump(speaker_data, f, indent=2)
                
            print(f"💾 Saved speaker info to {self.speaker_db_path}")
                
        except Exception as e:
            print(f"⚠️ Error saving speaker: {e}")

    def extract_speaker_embedding(self, audio_path, transcript, speaker_id):
        """Extract speaker embedding from audio file"""
        try:
            if not Path(audio_path).exists():
                print(f"❌ Audio file not found: {audio_path}")
                return False
                
            # Load audio at 16kHz
            audio_16k = load_wav(audio_path, 16000)
            
            # Add speaker embedding
            success = self.cosyvoice.add_zero_shot_spk(transcript, audio_16k, speaker_id)
            
            if success:
                print(f"✅ Extracted speaker embedding: {speaker_id}")
                # Save speaker info for persistence
                self._save_custom_speakers(speaker_id, audio_path, transcript)
                logging.info(f"Extracted and saved speaker embedding: {speaker_id}")
                return True
            else:
                print(f"❌ Failed to extract speaker embedding: {speaker_id}")
                return False
                
        except Exception as e:
            logging.error(f"Error extracting speaker embedding: {e}")
            print(f"❌ Error extracting speaker embedding: {e}")
            return False

    def synthesize_with_speaker_id(self, text, output_path, speaker_id):
        """Synthesize using saved speaker ID"""
        try:
            # Clean speaker ID
            speaker_id = speaker_id.strip('"\'')
            
            # Check if speaker exists in memory
            if not hasattr(self.cosyvoice, 'spk_db') or speaker_id not in self.cosyvoice.spk_db:
                print(f"❌ Speaker '{speaker_id}' not found in active memory")
                
                # Try to reload from file
                if self.speaker_db_path.exists():
                    with open(self.speaker_db_path, 'r') as f:
                        speaker_data = json.load(f)
                    
                    if speaker_id in speaker_data:
                        data = speaker_data[speaker_id]
                        print(f"🔄 Reloading speaker: {speaker_id}")
                        audio_16k = load_wav(data['audio_path'], 16000)
                        success = self.cosyvoice.add_zero_shot_spk(data['transcript'], audio_16k, speaker_id)
                        if not success:
                            print(f"❌ Failed to reload speaker: {speaker_id}")
                            return None
                    else:
                        print(f"❌ Speaker '{speaker_id}' not found in database")
                        return None
                else:
                    print("❌ No speaker database found")
                    return None
            
            print(f"✅ Using speaker: {speaker_id}")
            
            # Use zero-shot with saved speaker ID
            result = self.cosyvoice.inference_zero_shot(
                text, 
                '',  # empty prompt text when using speaker_id
                '',  # empty prompt audio when using speaker_id
                zero_shot_spk_id=speaker_id,
                stream=False
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated audio with speaker {speaker_id}: {final_path}")
                print(f"✅ Audio saved: {final_path}")
                return final_path
            return None
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            print(f"❌ TTS synthesis failed: {e}")
            return None

    def synthesize_with_audio_prompt(self, text, output_path, prompt_audio, prompt_text=""):
        """Synthesize using audio prompt (real-time voice cloning)"""
        try:
            if not Path(prompt_audio).exists():
                print(f"❌ Prompt audio not found: {prompt_audio}")
                return None
                
            # Load prompt audio
            prompt_speech_16k = load_wav(prompt_audio, 16000)
            
            # Use zero-shot with audio prompt
            result = self.cosyvoice.inference_zero_shot(
                text, 
                prompt_text,
                prompt_speech_16k,
                stream=False
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated audio with prompt: {final_path}")
                print(f"✅ Audio saved: {final_path}")
                return final_path
            return None
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            print(f"❌ TTS synthesis failed: {e}")
            return None

    def synthesize_with_instruction(self, text, output_path, instruction, speaker_id=None):
        """Synthesize with natural language instruction (emotion/style)"""
        try:
            if speaker_id:
                # Clean speaker ID
                speaker_id = speaker_id.strip('"\'')
                
                # Get speaker's original audio for instruction synthesis
                if self.speaker_db_path.exists():
                    with open(self.speaker_db_path, 'r') as f:
                        speaker_data = json.load(f)
                    
                    if speaker_id in speaker_data:
                        # Load the original audio
                        audio_path = speaker_data[speaker_id]['audio_path']
                        prompt_speech_16k = load_wav(audio_path, 16000)
                        
                        try:
                            # Use instruct2 with speaker's original audio
                            result = self.cosyvoice.inference_instruct2(
                                text,
                                instruction,
                                prompt_speech_16k,
                                zero_shot_spk_id=speaker_id,
                                stream=False
                            )
                        except Exception as instruct_error:
                            print(f"⚠️ Instruction synthesis failed: {instruct_error}")
                            print(f"🔄 Falling back to regular synthesis with speaker: {speaker_id}")
                            # Fallback to regular zero-shot
                            return self.synthesize_with_speaker_id(text, output_path, speaker_id)
                            
                    else:
                        print(f"❌ Speaker '{speaker_id}' not found in database for instruction synthesis")
                        return None
                else:
                    print(f"❌ Speaker database not found for instruction synthesis")
                    return None
            else:
                print(f"❌ Speaker ID required for instruction synthesis")
                return None
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated instructed audio: {final_path}")
                print(f"✅ Instructed audio saved: {final_path}")
                return final_path
            return None
        except Exception as e:
            logging.error(f"Instructed TTS synthesis failed: {e}")
            print(f"❌ Instructed TTS synthesis failed: {e}")
            # Fallback to regular synthesis
            if speaker_id:
                print(f"🔄 Falling back to regular synthesis")
                return self.synthesize_with_speaker_id(text, output_path, speaker_id.strip('"\''))
            return None

    def process_dialogue_script(self, script_path, output_dir=None, default_speaker_id=None):
        """Process dialogue script from .txt file"""
        script_path = Path(script_path)
        
        if not script_path.exists():
            print(f"❌ Script file not found: {script_path}")
            return
        
        # Create output directory
        if output_dir is None:
            output_dir = Path('output') / script_path.stem
        else:
            output_dir = Path(output_dir) / script_path.stem
            
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
        
        # Read script
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📜 Processing {len(lines)} lines from {script_path}")
        
        results = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            print(f"\n🎤 Processing line {i}: {line[:50]}...")
            
            # Parse line for special formatting
            text, speaker_id, instruction = self._parse_dialogue_line(line)
            
            # Use default speaker if none specified
            if not speaker_id:
                speaker_id = default_speaker_id
            
            # Generate output path
            output_path = output_dir / f"line_{i:03d}"
            
            # Synthesize based on available parameters
            result_path = None
            if instruction:
                print(f"🎭 Using instruction: {instruction}")
                result_path = self.synthesize_with_instruction(text, str(output_path), instruction, speaker_id)
            elif speaker_id:
                print(f"👤 Using speaker: {speaker_id}")
                result_path = self.synthesize_with_speaker_id(text, str(output_path), speaker_id)
            else:
                print("⚠️ No speaker or instruction specified - this may fail")
                print("ℹ️ Consider extracting a speaker embedding first")
                
            results.append({
                'line_number': i,
                'text': text,
                'speaker_id': speaker_id,
                'instruction': instruction,
                'output_path': result_path
            })
        
        # Save processing report
        report_path = output_dir / 'processing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Dialogue Processing Report\n")
            f.write(f"========================\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Default Speaker: {default_speaker_id}\n")
            f.write(f"Processed: {len(results)} lines\n\n")
            
            for result in results:
                f.write(f"Line {result['line_number']}: {result['text'][:50]}...\n")
                if result['speaker_id']:
                    f.write(f"  Speaker: {result['speaker_id']}\n")
                if result['instruction']:
                    f.write(f"  Instruction: {result['instruction']}\n")
                f.write(f"  Output: {result['output_path']}\n\n")
        
        print(f"\n🎉 Processed {len(results)} dialogue lines!")
        print(f"📋 Report saved: {report_path}")
        return results

    def _parse_dialogue_line(self, line):
        """Parse dialogue line for speaker/instruction tags"""
        # Format: [speaker:name] or [instruction:text] or [speaker:name,instruction:text] text
        speaker_id = None
        instruction = None
        text = line
        
        if line.startswith('[') and ']' in line:
            tag_end = line.find(']')
            tag_content = line[1:tag_end]
            text = line[tag_end+1:].strip()
            
            # Parse multiple parameters separated by comma
            for param in tag_content.split(','):
                param = param.strip()
                if ':' in param:
                    key, value = param.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'speaker':
                        speaker_id = value
                    elif key in ['instruction', 'emotion', 'style']:
                        instruction = value
        
        return text, speaker_id, instruction

    def show_available_options(self):
        """Display available options and usage patterns"""
        print(f"\n🎛️ CosyVoice2 Capabilities:")
        print(f"  ✅ Zero-shot voice cloning with audio prompts")
        print(f"  ✅ Speaker embedding extraction and reuse")
        print(f"  ✅ Emotion/Style control via natural language instructions")
        print(f"  ✅ Cross-lingual synthesis")
        print(f"  ✅ Fine-grained control ([laughter], [breath], etc.)")
        
        print(f"\n📝 Dialogue Script Formats:")
        print(f"  Basic: Hello, how are you?")
        print(f"  With speaker: [speaker:john] Hello there!")
        print(f"  With instruction: [instruction:用开心的语气说] I'm so excited!")
        print(f"  Combined: [speaker:alice,instruction:用温柔的语气说] Take care!")
        
        print(f"\n🎭 Instruction Examples:")
        print(f"  - 用开心的语气说 (speak happily)")
        print(f"  - 用悲伤的语气说 (speak sadly)")  
        print(f"  - 用快一点的语度说 (speak faster)")
        print(f"  - 用温柔的语气说 (speak gently)")

def main():
    parser = argparse.ArgumentParser(description='CosyVoice2 TTS with Speaker Embeddings')
    parser.add_argument('--mode', choices=['extract', 'simple', 'instruct', 'dialogue', 'options'], 
                       required=True, help='Mode: extract speaker, simple TTS, instruct TTS, dialogue, or show options')
    
    # Speaker extraction args
    parser.add_argument('--audio', type=str, help='Audio file path for speaker extraction')
    parser.add_argument('--transcript', type=str, help='Transcript of the audio for speaker extraction')
    parser.add_argument('--speaker_id', type=str, help='Speaker ID for extraction/synthesis')
    
    # Text synthesis args  
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--output_path', type=str, help='Output path for audio file')
    
    # Prompt args
    parser.add_argument('--prompt_audio', type=str, help='Prompt audio file for voice cloning')
    parser.add_argument('--prompt_text', type=str, default="", help='Prompt text')
    
    # Instruction args
    parser.add_argument('--instruction', type=str, help='Natural language instruction (emotion/style)')
    
    # Dialogue args
    parser.add_argument('--script', type=str, help='Path to dialogue script .txt file')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output directory')
    parser.add_argument('--default_speaker', type=str, help='Default speaker ID for dialogue')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🐛 Debug logging enabled")

    try:
        # Initialize TTS
        print("🚀 Initializing CosyVoice2...")
        tts = CosyVoice2TTS()
        
        if args.mode == 'options':
            tts.show_available_options()
            
        elif args.mode == 'extract':
            if not args.audio or not args.transcript or not args.speaker_id:
                print("❌ --audio, --transcript, and --speaker_id required for extract mode")
                sys.exit(1)
            
            print(f"🎙️ Extracting speaker embedding from: {args.audio}")
            print(f"📝 Transcript: {args.transcript}")
            print(f"👤 Speaker ID: {args.speaker_id}")
            
            success = tts.extract_speaker_embedding(args.audio, args.transcript, args.speaker_id)
            if success:
                print("✅ Speaker embedding extracted successfully!")
            else:
                print("❌ Speaker embedding extraction failed")
                sys.exit(1)
            
        elif args.mode == 'simple':
            if not args.text or not args.output_path:
                print("❌ --text and --output_path required for simple mode")
                sys.exit(1)
            
            print(f"🎤 Simple TTS: {args.text}")
            
            if args.speaker_id:
                print(f"👤 Using speaker: {args.speaker_id}")
                tts.synthesize_with_speaker_id(args.text, args.output_path, args.speaker_id)
            elif args.prompt_audio:
                print(f"🎙️ Using prompt audio: {args.prompt_audio}")
                tts.synthesize_with_audio_prompt(args.text, args.output_path, args.prompt_audio, args.prompt_text)
            else:
                print("❌ Either --speaker_id or --prompt_audio required")
                sys.exit(1)
            
        elif args.mode == 'instruct':
            if not args.text or not args.output_path or not args.instruction:
                print("❌ --text, --output_path, and --instruction required for instruct mode")
                sys.exit(1)
            
            print(f"🎭 Instruct TTS: {args.text}")
            print(f"📝 Instruction: {args.instruction}")
            if args.speaker_id:
                print(f"👤 Using speaker: {args.speaker_id}")
                
            tts.synthesize_with_instruction(args.text, args.output_path, args.instruction, args.speaker_id)
            
        elif args.mode == 'dialogue':
            if not args.script:
                print("❌ --script required for dialogue mode")
                sys.exit(1)
            
            print(f"📜 Processing dialogue script: {args.script}")
            if args.default_speaker:
                print(f"👤 Default speaker: {args.default_speaker}")
            else:
                print("⚠️ No default speaker - lines without [speaker:...] may fail")
                
            tts.process_dialogue_script(args.script, args.output_dir, args.default_speaker)
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()