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

    def synthesize(self, text, output_path, prompt_text=None, prompt_audio_path=None, speaker_id=None, stream=False):
        try:
            if prompt_audio_path:
                prompt_speech_16k = load_wav(prompt_audio_path, 16000)
            else:
                prompt_speech_16k = None
            if speaker_id:
                result = self.cosyvoice.inference_zero_shot(text, prompt_text or '', prompt_speech_16k, zero_shot_spk_id=speaker_id, stream=stream)
            else:
                result = self.cosyvoice.inference_zero_shot(text, prompt_text or '', prompt_speech_16k, stream=stream)
            for i, j in enumerate(result):
                torchaudio.save(f'{output_path}_output_{i}.wav', j['tts_speech'], self.cosyvoice.sample_rate)
                logging.info(f"Generated audio saved: {output_path}_output_{i}.wav")
                break
            return True
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            return False
