"""
Comprehensive test script for VoiceForge wrapper
Tests all major functionality without requiring actual audio files
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*80)
    print("TEST 1: Testing Imports")
    print("="*80)

    try:
        from wrapper import VoiceSynthesizer, SpeakerEncoder, VoiceForgeConfig
        print("✓ Main wrapper imports successful")

        from wrapper.synthesizer import VoiceSynthesizer as VS
        print("✓ VoiceSynthesizer import successful")

        from wrapper.encoders.speaker_encoder import SpeakerEncoder as SE
        print("✓ SpeakerEncoder import successful")

        from wrapper.config import VoiceForgeConfig as VFC
        print("✓ VoiceForgeConfig import successful")

        print("\n✅ All imports working correctly!")
        return True

    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False


def test_config():
    """Test 2: Test configuration system"""
    print("\n" + "="*80)
    print("TEST 2: Testing Configuration")
    print("="*80)

    try:
        from wrapper.config import VoiceForgeConfig

        # Test default config
        config = VoiceForgeConfig()
        print(f"✓ Default config created")
        print(f"  Output root: {config.output_root}")

        # Test custom config
        config2 = VoiceForgeConfig(output_root='test_outputs')
        print(f"✓ Custom config created")
        print(f"  Output root: {config2.output_root}")

        # Test directory creation
        assert config.get_tts_output_dir().exists(), "TTS output dir not created"
        assert config.get_embeddings_output_dir().exists(), "Embeddings output dir not created"
        assert config.get_vc_output_dir().exists(), "VC output dir not created"
        print("✓ Output directories created successfully")

        # Test parameter validation
        assert config.validate_speed(1.0) == 1.0, "Speed validation failed"
        assert config.validate_speed(-1.0) == 1.0, "Speed validation failed (negative)"
        assert config.validate_speed(5.0) == 2.0, "Speed validation failed (too high)"
        print("✓ Speed validation working")

        assert config.validate_language('en') == 'en', "Language validation failed"
        assert config.validate_language('english') == 'en', "Language validation failed"
        assert config.validate_language(None) == 'en', "Language validation failed (None)"
        print("✓ Language validation working")

        # Test output path generation
        path = config.get_output_path('tts', 'test.wav')
        assert path.parent == config.get_tts_output_dir(), "Output path generation failed"
        print("✓ Output path generation working")

        print("\n✅ Configuration tests passed!")
        return True

    except Exception as e:
        logger.error(f"Config test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesizer_init():
    """Test 3: Test synthesizer initialization"""
    print("\n" + "="*80)
    print("TEST 3: Testing Synthesizer Initialization")
    print("="*80)

    try:
        from wrapper.synthesizer import VoiceSynthesizer

        # Check if model exists
        model_dir = 'pretrained_models/CosyVoice2-0.5B'
        if not os.path.exists(model_dir):
            print(f"⚠ Model directory not found: {model_dir}")
            print("  Skipping synthesizer initialization test")
            print("  (This is OK if model not downloaded yet)")
            return True

        print(f"Loading model from {model_dir}...")
        synthesizer = VoiceSynthesizer(
            model_dir=model_dir,
            output_dir='test_outputs',
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )

        print(f"✓ Synthesizer initialized")
        print(f"  Sample rate: {synthesizer.sample_rate}")

        # Test methods exist
        assert hasattr(synthesizer, 'synthesize_simple'), "Missing synthesize_simple method"
        assert hasattr(synthesizer, 'synthesize_zero_shot'), "Missing synthesize_zero_shot method"
        assert hasattr(synthesizer, 'synthesize_cross_lingual'), "Missing synthesize_cross_lingual method"
        assert hasattr(synthesizer, 'synthesize_instruct'), "Missing synthesize_instruct method"
        assert hasattr(synthesizer, 'voice_conversion'), "Missing voice_conversion method"
        assert hasattr(synthesizer, 'synthesize_with_speaker_embedding'), "Missing synthesize_with_speaker_embedding method"
        print("✓ All synthesis methods exist")

        # Test utility methods
        speakers = synthesizer.list_available_speakers()
        print(f"✓ Available speakers: {len(speakers)}")

        info = synthesizer.get_model_info()
        print(f"✓ Model info retrieved")
        print(f"  Sample rate: {info['sample_rate']}")
        print(f"  Model dir: {info['model_dir']}")
        print(f"  Output dirs: {len(info['output_dirs'])}")

        print("\n✅ Synthesizer initialization tests passed!")
        return True

    except Exception as e:
        logger.error(f"Synthesizer init test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_encoder_init():
    """Test 4: Test speaker encoder initialization"""
    print("\n" + "="*80)
    print("TEST 4: Testing Speaker Encoder Initialization")
    print("="*80)

    try:
        from wrapper.encoders.speaker_encoder import SpeakerEncoder

        # Check if model exists
        model_dir = 'pretrained_models/CosyVoice2-0.5B'
        if not os.path.exists(model_dir):
            print(f"⚠ Model directory not found: {model_dir}")
            print("  Skipping encoder initialization test")
            print("  (This is OK if model not downloaded yet)")
            return True

        print(f"Loading encoder from {model_dir}...")
        encoder = SpeakerEncoder(
            model_dir=model_dir,
            output_dir='test_outputs/embeddings',
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )

        print(f"✓ Encoder initialized")
        print(f"  Sample rate: {encoder.sample_rate}")
        print(f"  Output dir: {encoder.output_dir}")

        # Test methods exist
        assert hasattr(encoder, 'extract_embedding'), "Missing extract_embedding method"
        assert hasattr(encoder, 'extract_embedding_with_features'), "Missing extract_embedding_with_features method"
        assert hasattr(encoder, 'load_embedding'), "Missing load_embedding method"
        assert hasattr(encoder, 'load_speaker_profile'), "Missing load_speaker_profile method"
        assert hasattr(encoder, 'batch_extract_embeddings'), "Missing batch_extract_embeddings method"
        print("✓ All encoder methods exist")

        print("\n✅ Speaker encoder initialization tests passed!")
        return True

    except Exception as e:
        logger.error(f"Encoder init test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test 5: Verify directory structure is correct"""
    print("\n" + "="*80)
    print("TEST 5: Testing Directory Structure")
    print("="*80)

    try:
        # Check wrapper directory
        assert os.path.exists('wrapper'), "wrapper/ directory missing"
        print("✓ wrapper/ exists")

        # Check wrapper files
        files = [
            'wrapper/__init__.py',
            'wrapper/config.py',
            'wrapper/synthesizer.py',
            'wrapper/README.md',
            'wrapper/encoders/__init__.py',
            'wrapper/encoders/speaker_encoder.py',
        ]

        for file in files:
            if os.path.exists(file):
                print(f"✓ {file} exists")
            else:
                print(f"✗ {file} missing")
                return False

        # Check examples directory
        if os.path.exists('examples'):
            example_files = [
                'examples/demo_basic_synthesis.py',
                'examples/demo_speaker_embedding.py',
                'examples/demo_advanced_synthesis.py',
            ]
            for file in example_files:
                if os.path.exists(file):
                    print(f"✓ {file} exists")
                else:
                    print(f"⚠ {file} missing (optional)")

        print("\n✅ Directory structure tests passed!")
        return True

    except Exception as e:
        logger.error(f"Directory structure test failed: {str(e)}")
        return False


def test_error_handling():
    """Test 6: Test error handling"""
    print("\n" + "="*80)
    print("TEST 6: Testing Error Handling")
    print("="*80)

    try:
        from wrapper.config import VoiceForgeConfig

        config = VoiceForgeConfig()

        # Test invalid output type
        try:
            config.get_output_path('invalid_type', 'test.wav')
            print("✗ Should have raised ValueError for invalid output type")
            return False
        except ValueError:
            print("✓ Correctly raises ValueError for invalid output type")

        # Test speed validation edge cases
        assert config.validate_speed(0) == 1.0, "Should default to 1.0 for invalid speed"
        assert config.validate_speed(-5) == 1.0, "Should default to 1.0 for negative speed"
        print("✓ Speed validation handles edge cases")

        # Test language validation
        assert config.validate_language('invalid') == 'en', "Should default to 'en' for invalid language"
        assert config.validate_language('') == 'en', "Should default to 'en' for empty string"
        print("✓ Language validation handles edge cases")

        print("\n✅ Error handling tests passed!")
        return True

    except Exception as e:
        logger.error(f"Error handling test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("VoiceForge Wrapper - Comprehensive Test Suite")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Synthesizer Initialization", test_synthesizer_init),
        ("Speaker Encoder Initialization", test_speaker_encoder_init),
        ("Directory Structure", test_directory_structure),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
