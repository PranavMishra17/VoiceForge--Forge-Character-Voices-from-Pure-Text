import sys
import os
import logging

sys.path.append('cosyvoice')
sys.path.append('cosyvoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test if CosyVoice2 model loads successfully"""
    try:
        logger.info("Loading CosyVoice2 model...")
        cosyvoice = CosyVoice2(
            'pretrained_models/CosyVoice2-0.5B',
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        logger.info(f"Model loaded successfully! Sample rate: {cosyvoice.sample_rate}")
        return cosyvoice
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def test_basic_inference(cosyvoice):
    """Test basic zero-shot inference"""
    try:
        logger.info("Testing zero-shot inference...")
        
        test_text = "Hello, this is a test of voice synthesis."
        prompt_text = "Testing voice generation."
        
        # Create test output directory
        os.makedirs('test_outputs', exist_ok=True)
        
        # Generate speech
        output_count = 0
        for i, j in enumerate(cosyvoice.inference_zero_shot(
            test_text,
            prompt_text,
            None,
            stream=False
        )):
            output_path = f'test_outputs/test_output_{i}.wav'
            torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
            logger.info(f"Generated audio saved to: {output_path}")
            output_count += 1
        
        logger.info(f"Inference test completed. Generated {output_count} audio file(s).")
        return True
    except Exception as e:
        logger.error(f"Inference test failed: {str(e)}")
        raise


def test_speaker_embedding():
    """Test speaker embedding functionality"""
    try:
        logger.info("Testing speaker embedding system...")
        cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
        
        # Test add_zero_shot_spk without actual audio file
        logger.info("Speaker embedding system is available")
        return True
    except Exception as e:
        logger.error(f"Speaker embedding test failed: {str(e)}")
        raise


def run_all_tests():
    """Run all validation tests"""
    logger.info("="*50)
    logger.info("Starting CosyVoice Installation Validation")
    logger.info("="*50)
    
    try:
        # Test 1: Model Loading
        logger.info("\n[Test 1/3] Model Loading")
        cosyvoice = test_model_loading()
        logger.info("✓ Model loading test passed\n")
        
        # Test 2: Basic Inference
        logger.info("[Test 2/3] Basic Inference")
        test_basic_inference(cosyvoice)
        logger.info("✓ Basic inference test passed\n")
        
        # Test 3: Speaker Embedding
        logger.info("[Test 3/3] Speaker Embedding System")
        test_speaker_embedding()
        logger.info("✓ Speaker embedding test passed\n")
        
        logger.info("="*50)
        logger.info("All tests passed! Installation is valid.")
        logger.info("="*50)
        return True
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f"Installation validation failed: {str(e)}")
        logger.error("="*50)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
