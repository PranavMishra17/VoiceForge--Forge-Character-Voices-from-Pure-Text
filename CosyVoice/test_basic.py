#!/usr/bin/env python3
"""
Basic test script for CosyVoice functionality
"""

import sys
import os
sys.path.append('third_party/Matcha-TTS')

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    print("✅ Successfully imported CosyVoice modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the required dependencies first")
    sys.exit(1)

def test_basic_functionality():
    print("🚀 Testing CosyVoice basic functionality...")
    
    # Check if model directory exists
    model_path = 'pretrained_models/CosyVoice2-0.5B'
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    print(f"✅ Model directory found: {model_path}")
    
    try:
        # Initialize model
        print("📦 Initializing CosyVoice2 model...")
        cosyvoice = CosyVoice2(
            model_path, 
            load_jit=False, 
            load_trt=False, 
            load_vllm=False, 
            fp16=False
        )
        print("✅ Model initialized successfully!")
        
        # Check if test audio exists
        test_audio_path = './asset/zero_shot_prompt.wav'
        if not os.path.exists(test_audio_path):
            print(f"⚠️  Test audio not found: {test_audio_path}")
            print("Creating a simple test without audio prompt...")
            
            # Test with just text (no audio prompt)
            target_text = "Hello, this is a test of CosyVoice text-to-speech."
            print(f"🎤 Testing text synthesis: '{target_text}'")
            
            # Use a simple inference method
            try:
                # Try basic text synthesis
                result = cosyvoice.inference_zero_shot(
                    target_text, 
                    "",  # empty prompt text
                    None,  # no prompt audio
                    stream=False
                )
                
                # Get the first result
                for i, j in enumerate(result):
                    output_path = f'test_output_{i}.wav'
                    torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
                    print(f"✅ Generated audio saved: {output_path}")
                    break
                    
            except Exception as e:
                print(f"❌ Text synthesis failed: {e}")
                return False
                
        else:
            print(f"✅ Test audio found: {test_audio_path}")
            
            # Load test audio
            prompt_speech_16k = load_wav(test_audio_path, 16000)
            prompt_text = '希望你以后能够做的比我还好呦。'
            target_text = 'Hello, this is a test of voice cloning.'
            
            print(f"🎤 Testing voice cloning with prompt: '{prompt_text}'")
            print(f"🎤 Target text: '{target_text}'")
            
            # Generate speech
            for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_speech_16k, stream=False)):
                output_path = f'test_output_{i}.wav'
                torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
                print(f"✅ Generated audio saved: {output_path}")
                break
        
        print("🎉 Basic functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎊 CosyVoice is working correctly!")
    else:
        print("\n💥 CosyVoice test failed. Please check the errors above.") 