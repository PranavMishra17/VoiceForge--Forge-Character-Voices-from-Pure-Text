"""
Demo: Basic TTS Synthesis Examples
Shows how to use the VoiceSynthesizer for basic text-to-speech
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from wrapper.synthesizer import VoiceSynthesizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_simple_synthesis():
    """Demo 1: Simple TTS synthesis with default settings"""
    print("\n" + "="*80)
    print("DEMO 1: Simple TTS Synthesis (Default English)")
    print("="*80)

    try:
        # Initialize synthesizer
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        # Simple synthesis
        text = "Hello, this is a demonstration of the VoiceForge text to speech system."

        result = synthesizer.synthesize_simple(
            text=text,
            language='en',  # English (default)
            speed=1.0,
            output_filename='demo_simple.wav'
        )

        print(f"\n✓ Synthesis successful!")
        print(f"  Output: {result['output_path']}")
        print(f"  Duration: {result['duration']:.2f} seconds")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_different_speeds():
    """Demo 2: Synthesis with different speeds"""
    print("\n" + "="*80)
    print("DEMO 2: Synthesis with Different Speeds")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        text = "This demonstrates different speaking speeds."

        speeds = [0.8, 1.0, 1.2, 1.5]

        for speed in speeds:
            print(f"\nSynthesizing at speed {speed}x...")
            result = synthesizer.synthesize_simple(
                text=text,
                speed=speed,
                output_filename=f'demo_speed_{speed}.wav'
            )
            print(f"  ✓ Saved: {result['output_path']}")
            print(f"  Duration: {result['duration']:.2f}s")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_long_text():
    """Demo 3: Synthesis of longer text"""
    print("\n" + "="*80)
    print("DEMO 3: Long Text Synthesis")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        long_text = """
        The VoiceForge system is a comprehensive wrapper around CosyVoice,
        providing advanced text-to-speech capabilities. It supports multiple
        synthesis modes including zero-shot voice cloning, cross-lingual
        synthesis, and instruction-based voice control. The system handles
        all audio processing, speaker embedding extraction, and output
        management automatically, making it easy to create high-quality
        synthetic speech.
        """

        result = synthesizer.synthesize_simple(
            text=long_text,
            language='en',
            speed=1.0,
            output_filename='demo_long_text.wav'
        )

        print(f"\n✓ Synthesis successful!")
        print(f"  Output: {result['output_path']}")
        print(f"  Duration: {result['duration']:.2f} seconds")
        print(f"  Text length: {len(long_text)} characters")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_model_info():
    """Demo 4: Get model information"""
    print("\n" + "="*80)
    print("DEMO 4: Model Information")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        info = synthesizer.get_model_info()

        print("\nModel Information:")
        print(f"  Sample Rate: {info['sample_rate']} Hz")
        print(f"  Model Directory: {info['model_dir']}")
        print(f"  Available Speakers: {info['available_speakers']}")
        print(f"\nOutput Directories:")
        for key, path in info['output_dirs'].items():
            print(f"  {key}: {path}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VoiceForge - Basic Synthesis Demos")
    print("="*80)

    # Run all demos
    demo_simple_synthesis()
    demo_different_speeds()
    demo_long_text()
    demo_model_info()

    print("\n" + "="*80)
    print("All demos completed successfully!")
    print("="*80 + "\n")
