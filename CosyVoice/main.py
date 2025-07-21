import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse
import torchaudio

# Add paths
sys.path.append('third_party/Matcha-TTS')
sys.path.append('CosyVoice')

# Direct imports like test_basic.py
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

class CosyVoiceTTS:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B'):
        """Initialize CosyVoice2 model"""
        try:
            self.cosyvoice = CosyVoice2(
                model_path, 
                load_jit=False, 
                load_trt=False, 
                load_vllm=False, 
                fp16=False
            )
            # Get available speakers for SFT mode
            self.available_speakers = self.cosyvoice.list_available_spks()
            if not self.available_speakers:
                self.available_speakers = ['default']
            
            logging.info("CosyVoice2 model initialized successfully")
            print("✅ CosyVoice2 model loaded successfully")
            print(f"📢 Available speakers: {self.available_speakers}")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise e

    def synthesize_simple(self, text, output_path, speaker=None):
        """Simple text-to-speech synthesis using SFT mode"""
        try:
            # Use first available speaker if none specified
            if speaker is None:
                speaker = self.available_speakers[0] if self.available_speakers else 'default'
            
            # Use SFT inference (predefined speakers)
            result = self.cosyvoice.inference_sft(
                text, 
                speaker,
                stream=False
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated audio saved: {final_path}")
                print(f"✅ Audio saved: {final_path}")
                return final_path
            return None
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            print(f"❌ TTS synthesis failed: {e}")
            return None

    def synthesize_with_style(self, text, output_path, emotion=None, style=None, speed=1.0, speaker=None):
        """Synthesize with style control - Note: CosyVoice2 doesn't support instruct mode"""
        print(f"⚠️  CosyVoice2 doesn't support emotion/style instructions")
        print(f"🔄 Using SFT mode with speed control instead")
        
        try:
            # Use first available speaker if none specified
            if speaker is None:
                speaker = self.available_speakers[0] if self.available_speakers else 'default'
            
            # Use SFT inference with speed control
            result = self.cosyvoice.inference_sft(
                text, 
                speaker,
                stream=False,
                speed=speed
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated styled audio: {final_path}")
                print(f"✅ Styled audio saved: {final_path} (speaker: {speaker}, speed: {speed})")
                return final_path
            return None
        except Exception as e:
            logging.error(f"Styled TTS synthesis failed: {e}")
            print(f"❌ Styled TTS synthesis failed: {e}")
            # Fallback to simple synthesis
            print("🔄 Falling back to simple synthesis...")
            return self.synthesize_simple(text, output_path, speaker)

    def process_dialogue_script(self, script_path, output_dir=None):
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
            
            # Parse line for special formatting (speaker:speed:text)
            text, speaker, speed = self._parse_dialogue_line(line)
            
            # Generate output path
            output_path = output_dir / f"line_{i:03d}"
            
            # Synthesize
            if speaker or speed != 1.0:
                result_path = self.synthesize_with_style(text, str(output_path), speaker=speaker, speed=speed)
            else:
                result_path = self.synthesize_simple(text, str(output_path))
                
            results.append({
                'line_number': i,
                'text': text,
                'speaker': speaker,
                'speed': speed,
                'output_path': result_path
            })
        
        # Save processing report
        report_path = output_dir / 'processing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Dialogue Processing Report\n")
            f.write(f"========================\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Processed: {len(results)} lines\n\n")
            
            for result in results:
                f.write(f"Line {result['line_number']}: {result['text'][:50]}...\n")
                if result['speaker']:
                    f.write(f"  Speaker: {result['speaker']}\n")
                if result['speed'] != 1.0:
                    f.write(f"  Speed: {result['speed']}\n")
                f.write(f"  Output: {result['output_path']}\n\n")
        
        print(f"\n🎉 Processed {len(results)} dialogue lines!")
        print(f"📋 Report saved: {report_path}")
        return results

    def _parse_dialogue_line(self, line):
        """Parse dialogue line for emotion/style tags"""
        # Format: [emotion:style] text or [emotion] text or just text
        emotion = None
        style = None
        text = line
        
        if line.startswith('[') and ']' in line:
            tag_end = line.find(']')
            tag_content = line[1:tag_end]
            text = line[tag_end+1:].strip()
            
            if ':' in tag_content:
                emotion, style = tag_content.split(':', 1)
                emotion = emotion.strip()
                style = style.strip()
            else:
                emotion = tag_content.strip()
        
        return text, emotion, style

    def show_available_options(self):
        """Display available options for CosyVoice2"""
        print("\n📢 Available Speakers (SFT mode):")
        for speaker in self.available_speakers:
            print(f"  - {speaker}")
        
        print(f"\n🎛️ CosyVoice2 Capabilities:")
        print(f"  - Text-to-Speech with predefined speakers")
        print(f"  - Speed control (0.5-2.0x)")
        print(f"  - Voice cloning with audio prompts")
        print(f"  - ❌ Emotion/Style instructions (not supported)")
        
        print(f"\n📝 Dialogue Script Format:")
        print(f"  Simple: Hello, how are you?")
        print(f"  With speaker: [speaker:alice] Hello there!")
        print(f"  With speed: [speed:1.5] This will be faster!")
        print(f"  Combined: [speaker:bob,speed:0.8] Slow and steady.")
        print(f"  Empty lines are skipped automatically.")

    def _parse_dialogue_line(self, line):
        """Parse dialogue line for speaker/speed tags"""
        # Format: [speaker:name] or [speed:1.5] or [speaker:name,speed:1.5] text
        speaker = None
        speed = 1.0
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
                        speaker = value
                    elif key == 'speed':
                        try:
                            speed = float(value)
                            speed = max(0.5, min(2.0, speed))  # Clamp to valid range
                        except ValueError:
                            pass
        
        return text, speaker, speed

def main():
    parser = argparse.ArgumentParser(description='CosyVoice TTS with Dialogue Support')
    parser.add_argument('--mode', choices=['simple', 'styled', 'dialogue', 'options'], 
                       required=True, help='Mode: simple TTS, styled TTS, dialogue script, or show options')
    
    # Text synthesis args
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--output_path', type=str, help='Output path for audio file')
    
    # Style args  
    parser.add_argument('--speaker', type=str, help='Speaker name (from available speakers)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed (0.5-2.0)')
    
    # Dialogue args
    parser.add_argument('--script', type=str, help='Path to dialogue script .txt file')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output directory')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("[DEBUG] Debug logging enabled")

    try:
        # Initialize TTS
        if args.mode != 'options':
            print("🚀 Initializing CosyVoice...")
            tts = CosyVoiceTTS()
        
        if args.mode == 'options':
            # Show available options
            tts = CosyVoiceTTS()  # Still need to initialize to show options
            tts.show_available_options()
            
        elif args.mode == 'simple':
            if not args.text or not args.output_path:
                print("❌ --text and --output_path required for simple mode")
                sys.exit(1)
            
            print(f"🎤 Simple TTS: {args.text}")
            tts.synthesize_simple(args.text, args.output_path, args.speaker)
            
        elif args.mode == 'styled':
            if not args.text or not args.output_path:
                print("❌ --text and --output_path required for styled mode")
                sys.exit(1)
            
            print(f"🎛️ Styled TTS: {args.text}")
            if args.speaker:
                print(f"   Speaker: {args.speaker}")
            if args.speed != 1.0:
                print(f"   Speed: {args.speed}")
                
            tts.synthesize_with_style(
                args.text, 
                args.output_path, 
                speaker=args.speaker,
                speed=args.speed
            )
            
        elif args.mode == 'dialogue':
            if not args.script:
                print("❌ --script required for dialogue mode")
                sys.exit(1)
            
            print(f"📜 Processing dialogue script: {args.script}")
            tts.process_dialogue_script(args.script, args.output_dir)
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()