import os
import logging
from pathlib import Path
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceTTS:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B', log_dir='logs'):
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / 'tts.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )

    def synthesize(self, text, output_path, speaker_id=None, speed=None, stream=False):
        """
        Synthesize speech using CosyVoice2's SFT mode. Supports speaker selection and speed control.
        """
        try:
            # List available speakers
            available_spks = self.cosyvoice.list_available_spks() if hasattr(self.cosyvoice, 'list_available_spks') else ['default']
            if not speaker_id or speaker_id not in available_spks:
                speaker_id = available_spks[0]
            # Handle speed control in text
            sft_kwargs = {'spk_id': speaker_id}
            if speed is not None:
                sft_kwargs['speed'] = speed
            # Synthesize
            result = self.cosyvoice.inference_sft(text, **sft_kwargs)
            for i, j in enumerate(result):
                torchaudio.save(f'{output_path}_output_{i}.wav', j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated audio saved: {output_path}_output_{i}.wav")
                break
            return True
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            return False
