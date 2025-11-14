"""
Demo: Advanced TTS Synthesis Features
Shows zero-shot, cross-lingual, instruction-based synthesis, and voice conversion
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from wrapper.synthesizer import VoiceSynthesizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_zero_shot_synthesis():
    """Demo 1: Zero-shot voice cloning with audio sample"""
    print("\n" + "="*80)
    print("DEMO 1: Zero-Shot Voice Cloning")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage example for zero-shot synthesis:")
        print("""
        # Synthesize text in a voice from a 3-10 second audio sample
        result = synthesizer.synthesize_zero_shot(
            text='This is the text I want to synthesize in the cloned voice.',
            prompt_text='This is what the speaker says in the audio sample.',
            prompt_audio='path/to/speaker_sample.wav',  # 3-10 seconds of clean audio
            language='en',
            speed=1.0,
            output_filename='zero_shot_cloned.wav'
        )

        print(f"Output: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f}s")

        # The synthesized audio will sound like the speaker in prompt_audio
        # but saying the text you provided
        """)

        print("\n✓ Demo code shown (requires audio sample to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_zero_shot_with_saved_profile():
    """Demo 2: Zero-shot synthesis using pre-saved speaker profile"""
    print("\n" + "="*80)
    print("DEMO 2: Zero-Shot with Saved Speaker Profile")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage example with saved profile:")
        print("""
        # First, create and save a speaker profile (one time)
        from wrapper.encoders.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        profile_result = encoder.extract_embedding_with_features(
            audio_input='path/to/speaker.wav',
            prompt_text='Sample text from the speaker',
            speaker_id='celebrity_voice_001',
            save_embedding=True
        )

        # Then use the speaker_id for synthesis (no need to load audio again)
        result = synthesizer.synthesize_zero_shot(
            text='New text to synthesize',
            prompt_text='',  # Not needed when using speaker_id
            prompt_audio=None,  # Not needed
            speaker_id='celebrity_voice_001',  # Use saved profile
            language='en',
            output_filename='from_saved_profile.wav'
        )

        # This is much faster as we don't need to process the audio again
        """)

        print("\n✓ Demo code shown")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_cross_lingual_synthesis():
    """Demo 3: Cross-lingual synthesis (e.g., English text with Chinese voice)"""
    print("\n" + "="*80)
    print("DEMO 3: Cross-Lingual Synthesis")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage example for cross-lingual synthesis:")
        print("""
        # Synthesize English text with a Chinese speaker's voice
        # Or Chinese text with an English speaker's voice
        result = synthesizer.synthesize_cross_lingual(
            text='Hello, this is English text spoken in a Chinese accent.',
            prompt_audio='path/to/chinese_speaker.wav',  # Audio of Chinese speaker
            language='en',  # Text is in English
            speed=1.0,
            output_filename='cross_lingual.wav'
        )

        # The result will have the voice characteristics of the Chinese speaker
        # but speaking English text

        # Works both ways:
        result2 = synthesizer.synthesize_cross_lingual(
            text='你好，这是中文文本。',  # Chinese text
            prompt_audio='path/to/english_speaker.wav',  # English speaker's audio
            language='zh',
            output_filename='cross_lingual_zh.wav'
        )
        """)

        print("\n✓ Demo code shown (requires audio sample to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_instruction_based_synthesis():
    """Demo 4: Instruction-based synthesis (control emotion, style)"""
    print("\n" + "="*80)
    print("DEMO 4: Instruction-Based Synthesis")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage examples for instruction-based synthesis:")
        print("""
        # Synthesize with emotional control
        instructions = [
            'Speak with excitement and enthusiasm',
            'Speak calmly and slowly',
            'Speak with a whisper',
            'Speak loudly and clearly',
            'Speak in a sad tone'
        ]

        for i, instruction in enumerate(instructions):
            result = synthesizer.synthesize_instruct(
                text='This is a test of instruction-based voice synthesis.',
                instruct_text=instruction,
                prompt_audio='path/to/speaker.wav',
                language='en',
                speed=1.0,
                output_filename=f'instruct_{i}.wav'
            )
            print(f"{instruction}: {result['output_path']}")

        # You can also control speaking style:
        result = synthesizer.synthesize_instruct(
            text='The weather is nice today.',
            instruct_text='Speak like a news anchor, professionally and clearly',
            prompt_audio='path/to/speaker.wav',
            output_filename='news_anchor_style.wav'
        )

        # Or combine emotion and style:
        result = synthesizer.synthesize_instruct(
            text='I cannot believe this happened!',
            instruct_text='Speak with surprise and disbelief, raising pitch',
            prompt_audio='path/to/speaker.wav',
            output_filename='surprised_style.wav'
        )
        """)

        print("\n✓ Demo code shown (requires audio sample to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_voice_conversion():
    """Demo 5: Voice conversion - convert one voice to another"""
    print("\n" + "="*80)
    print("DEMO 5: Voice Conversion")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage example for voice conversion:")
        print("""
        # Convert source audio to sound like target speaker
        result = synthesizer.voice_conversion(
            source_audio='path/to/source_speech.wav',  # Audio to convert
            target_audio='path/to/target_voice.wav',   # Target voice
            speed=1.0,
            output_filename='converted_voice.wav'
        )

        print(f"Converted audio: {result['output_path']}")

        # The output will have:
        # - Content/words from source_audio
        # - Voice characteristics from target_audio

        # Example use cases:
        # 1. Convert your voice to sound like someone else
        # 2. Unify voice across multiple speakers in a podcast
        # 3. Create consistent character voices for audiobooks
        # 4. Privacy protection - change your voice in recordings
        """)

        print("\n✓ Demo code shown (requires audio samples to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_embedding_based_synthesis():
    """Demo 6: Synthesis using speaker embedding vectors"""
    print("\n" + "="*80)
    print("DEMO 6: Synthesis with Speaker Embedding Vectors")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage examples for embedding-based synthesis:")
        print("""
        # Method 1: Use saved embedding file path
        result = synthesizer.synthesize_with_speaker_embedding(
            text='This will use the saved speaker embedding.',
            speaker_embedding='src/outputs/embeddings/speaker_001_profile.pt',
            language='en',
            speed=1.0,
            output_filename='from_embedding_file.wav'
        )

        # Method 2: Use speaker_id (if already loaded in model)
        result = synthesizer.synthesize_with_speaker_embedding(
            text='Using speaker ID directly.',
            speaker_embedding='speaker_001',  # Must exist in model
            language='en',
            output_filename='from_speaker_id.wav'
        )

        # Method 3: Use embedding tensor directly
        import torch
        embedding_data = torch.load('src/outputs/embeddings/speaker_001_embedding.pt')
        embedding_tensor = embedding_data['embedding']

        result = synthesizer.synthesize_with_speaker_embedding(
            text='Using raw embedding tensor.',
            speaker_embedding=embedding_tensor,
            language='en',
            output_filename='from_tensor.wav'
        )

        # Method 4: Use complete profile dictionary
        profile_data = torch.load('src/outputs/embeddings/speaker_001_profile.pt')
        profile_dict = profile_data['profile']

        result = synthesizer.synthesize_with_speaker_embedding(
            text='Using complete profile dictionary.',
            speaker_embedding=profile_dict,
            language='en',
            output_filename='from_profile_dict.wav'
        )

        print(f"All outputs created successfully!")
        """)

        print("\n✓ Demo code shown")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_available_speakers():
    """Demo 7: List and use built-in speakers"""
    print("\n" + "="*80)
    print("DEMO 7: Working with Available Speakers")
    print("="*80)

    try:
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        print("\nUsage example:")
        print("""
        # List all available speakers
        speakers = synthesizer.list_available_speakers()
        print(f"Available speakers: {speakers}")

        # If the model has pre-trained speakers, you can use them
        if len(speakers) > 0:
            for speaker_id in speakers[:3]:  # Use first 3 speakers
                result = synthesizer.synthesize_with_speaker_embedding(
                    text=f'This is {speaker_id} speaking.',
                    speaker_embedding=speaker_id,
                    language='en',
                    output_filename=f'{speaker_id}.wav'
                )
                print(f"Generated: {result['output_path']}")
        """)

        print("\n✓ Demo code shown")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VoiceForge - Advanced Synthesis Demos")
    print("="*80)
    print("\nNote: These demos show usage patterns.")
    print("To run them with real audio, replace 'path/to/audio.wav' with actual file paths.")

    # Run all demos
    demo_zero_shot_synthesis()
    demo_zero_shot_with_saved_profile()
    demo_cross_lingual_synthesis()
    demo_instruction_based_synthesis()
    demo_voice_conversion()
    demo_embedding_based_synthesis()
    demo_available_speakers()

    print("\n" + "="*80)
    print("All demos completed!")
    print("="*80 + "\n")
