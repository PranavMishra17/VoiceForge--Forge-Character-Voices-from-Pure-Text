import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class SpeakerEmbeddingExtractor:
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B', log_dir='logs'):
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.embeddings_db = {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / 'embedding.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )

    def extract_from_folder(self, audio_folder, output_dir='output/embeddings'):
        import numpy as np
        audio_folder = Path(audio_folder)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_files = [f for f in audio_folder.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
        for audio_file in tqdm(audio_files, desc="Extracting speaker embeddings"):
            speaker_id = audio_file.stem
            try:
                audio_16k = load_wav(str(audio_file), 16000)
                success = self.cosyvoice.add_zero_shot_spk("", audio_16k, speaker_id)
                if success:
                    self.embeddings_db[speaker_id] = {
                        'audio_path': str(audio_file),
                        'extracted': True
                    }
                    speaker_dir = output_dir / speaker_id
                    speaker_dir.mkdir(exist_ok=True)
                    # Save metadata
                    with open(speaker_dir / 'embedding_metadata.json', 'w') as f:
                        json.dump(self.embeddings_db[speaker_id], f, indent=2)
                    # Save actual embedding vector
                    emb = None
                    # Try to get embedding from CosyVoice2 internal db
                    try:
                        emb = self.cosyvoice.spk_db[speaker_id]['spk_emb']
                    except Exception as e:
                        logging.warning(f"Could not retrieve embedding vector for {speaker_id}: {e}")
                    if emb is not None:
                        # Save as .npy
                        np.save(speaker_dir / 'speaker_embedding.npy', emb)
                        # Also save as JSON for inspection
                        with open(speaker_dir / 'speaker_embedding.json', 'w') as f:
                            json.dump(emb.tolist(), f)
                        logging.info(f"Saved embedding vector for {speaker_id} to {speaker_dir}")
                    logging.info(f"Extracted embedding for {speaker_id} from {audio_file}")
                else:
                    logging.warning(f"Failed to extract embedding for {speaker_id}")
            except Exception as e:
                logging.error(f"Error extracting embedding for {speaker_id}: {e}")
        # Save all embeddings metadata summary
        with open(output_dir / 'all_speaker_embeddings_metadata.json', 'w') as f:
            json.dump(self.embeddings_db, f, indent=2)
        logging.info(f"Saved all speaker embeddings metadata to {output_dir}")
