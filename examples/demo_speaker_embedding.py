"""
Demo: Speaker Embedding Extraction and Usage
Shows how to extract speaker embeddings and use them for TTS
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from wrapper.encoders.speaker_encoder import SpeakerEncoder
from wrapper.synthesizer import VoiceSynthesizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_extract_embedding():
    """Demo 1: Extract speaker embedding from audio"""
    print("\n" + "="*80)
    print("DEMO 1: Extract Speaker Embedding from Audio")
    print("="*80)

    try:
        # Initialize encoder
        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        # Note: You need to provide an actual audio file path here
        # For demo purposes, we'll show the usage pattern
        print("\nUsage example:")
        print("""
        # Extract embedding from audio file
        result = encoder.extract_embedding(
            audio_input='path/to/speaker_audio.wav',
            speaker_id='my_speaker_001',
            save_embedding=True,
            metadata={'name': 'John Doe', 'gender': 'male'}
        )

        print(f"Embedding saved to: {result['embedding_path']}")
        print(f"Embedding shape: {result['shape']}")
        print(f"Speaker ID: {result['speaker_id']}")
        """)

        print("\n✓ Demo code shown (requires actual audio file to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_extract_full_profile():
    """Demo 2: Extract complete speaker profile with features"""
    print("\n" + "="*80)
    print("DEMO 2: Extract Complete Speaker Profile")
    print("="*80)

    try:
        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        print("\nUsage example for complete speaker profile:")
        print("""
        # Extract complete profile (embedding + speech tokens + features)
        result = encoder.extract_embedding_with_features(
            audio_input='path/to/speaker_audio.wav',
            prompt_text='This is what the speaker is saying in the audio',
            speaker_id='speaker_profile_001',
            save_embedding=True
        )

        print(f"Profile saved to: {result['profile_path']}")
        print(f"Speaker ID: {result['speaker_id']}")

        # This profile can be used directly for zero-shot TTS
        """)

        print("\n✓ Demo code shown (requires actual audio file to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_batch_extraction():
    """Demo 3: Batch extract embeddings from multiple files"""
    print("\n" + "="*80)
    print("DEMO 3: Batch Embedding Extraction")
    print("="*80)

    try:
        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        print("\nUsage example for batch extraction:")
        print("""
        # Extract embeddings from multiple audio files
        audio_files = [
            'path/to/speaker1.wav',
            'path/to/speaker2.wav',
            'path/to/speaker3.wav'
        ]

        speaker_ids = ['speaker_001', 'speaker_002', 'speaker_003']

        results = encoder.batch_extract_embeddings(
            audio_files=audio_files,
            speaker_ids=speaker_ids,
            save_embeddings=True
        )

        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"Speaker {i+1}: {result['speaker_id']} - {result['embedding_path']}")
            else:
                print(f"Speaker {i+1}: Failed - {result['error']}")
        """)

        print("\n✓ Demo code shown (requires actual audio files to run)")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_load_and_use_embedding():
    """Demo 4: Load saved embedding and use for synthesis"""
    print("\n" + "="*80)
    print("DEMO 4: Load Embedding and Use for TTS")
    print("="*80)

    try:
        print("\nWorkflow example:")
        print("""
        # Step 1: Extract and save embedding
        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        result = encoder.extract_embedding_with_features(
            audio_input='path/to/speaker.wav',
            prompt_text='Sample text from the speaker',
            speaker_id='my_custom_voice',
            save_embedding=True
        )

        embedding_path = result['profile_path']

        # Step 2: Use the saved embedding for TTS
        synthesizer = VoiceSynthesizer(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs'
        )

        # Option 1: Load from file path
        tts_result = synthesizer.synthesize_with_speaker_embedding(
            text='Hello, this will be spoken in the custom voice',
            speaker_embedding=embedding_path,  # Path to saved .pt file
            language='en',
            speed=1.0,
            output_filename='custom_voice_output.wav'
        )

        # Option 2: Load embedding first, then use
        loaded = encoder.load_speaker_profile(embedding_path)
        speaker_id = loaded['speaker_id']

        tts_result = synthesizer.synthesize_with_speaker_embedding(
            text='Another sentence with the same voice',
            speaker_embedding=speaker_id,  # Use speaker_id
            language='en',
            output_filename='custom_voice_output2.wav'
        )

        print(f"TTS output: {tts_result['output_path']}")
        """)

        print("\n✓ Demo workflow shown")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


def demo_embedding_management():
    """Demo 5: Managing saved embeddings"""
    print("\n" + "="*80)
    print("DEMO 5: Embedding Management")
    print("="*80)

    try:
        encoder = SpeakerEncoder(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            output_dir='src/outputs/embeddings'
        )

        print("\nEmbedding management examples:")
        print("""
        # Load a saved embedding
        embedding_data = encoder.load_embedding('src/outputs/embeddings/speaker_001_embedding.pt')
        print(f"Speaker ID: {embedding_data['speaker_id']}")
        print(f"Timestamp: {embedding_data['timestamp']}")
        print(f"Shape: {embedding_data['shape']}")
        print(f"Metadata: {embedding_data['metadata']}")

        # Load a complete speaker profile
        profile_data = encoder.load_speaker_profile('src/outputs/embeddings/speaker_001_profile.pt')
        print(f"Speaker ID: {profile_data['speaker_id']}")
        print(f"Prompt text: {profile_data['prompt_text']}")
        print(f"Profile keys: {profile_data['profile'].keys()}")

        # The profile contains:
        # - llm_embedding: Speaker embedding for language model
        # - flow_embedding: Speaker embedding for flow model
        # - prompt_text_token: Tokenized prompt text
        # - llm_prompt_speech_token: Speech tokens for LLM
        # - flow_prompt_speech_token: Speech tokens for flow
        # - prompt_speech_feat: Speech features
        # - And more...
        """)

        print("\n✓ Demo code shown")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VoiceForge - Speaker Embedding Demos")
    print("="*80)
    print("\nNote: These demos show usage patterns.")
    print("To run them with real audio, replace 'path/to/audio.wav' with actual file paths.")

    # Run all demos
    demo_extract_embedding()
    demo_extract_full_profile()
    demo_batch_extraction()
    demo_load_and_use_embedding()
    demo_embedding_management()

    print("\n" + "="*80)
    print("All demos completed!")
    print("="*80 + "\n")
