#!/usr/bin/env python3
"""
custom_xtts.py - Custom XTTS Integration
Modified XTTS that accepts textual voice descriptions instead of audio samples
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional


from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts
from TTS.TTS.api import TTS


from voice_encoder import VoiceDescriptionEncoder


class CustomXTTS:
    """
    Modified XTTS that accepts textual voice descriptions
    """
    def __init__(self, tts_repo_path: str, sentence_model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_repo_path = Path(tts_repo_path)
        
        print("Initializing XTTS-v2...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        # Get model path
        home = os.path.expanduser("~")
        self.model_path = Path(home) / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        
        # Load XTTS model directly for customization
        self.config = XttsConfig()
        self.config.load_json(self.model_path / "config.json")
        
        self.xtts_model = Xtts.init_from_config(self.config)
        self.xtts_model.load_checkpoint(self.config, checkpoint_dir=str(self.model_path), eval=True)
        self.xtts_model.to(self.device)
        
        # Load sentence transformer
        print(f"Loading sentence transformer: {sentence_model_name}")
        self.sentence_model = SentenceTransformer(sentence_model_name)
        sentence_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Detect actual XTTS dimensions
        self._detect_xtts_dimensions()
        
        # Initialize voice description encoder
        self.voice_encoder = VoiceDescriptionEncoder(
            sentence_embedding_dim=sentence_dim,
            voice_embedding_dim=self.speaker_dim,
            gpt_cond_latent_dim=self.gpt_cond_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.voice_encoder.parameters(), lr=1e-4)
        self.is_trained = False
        
    def _detect_xtts_dimensions(self):
        """Detect actual XTTS embedding dimensions"""
        print("Detecting XTTS dimensions...")
        
        # Use a sample audio to get dimensions
        sample_audio_path = self.model_path / "samples"
        if sample_audio_path.exists():
            audio_files = list(sample_audio_path.glob("*.wav"))
            if audio_files:
                gpt_cond, speaker_emb = self.xtts_model.get_conditioning_latents(
                    audio_path=[str(audio_files[0])]
                )
                self.speaker_dim = speaker_emb.shape[-1]
                self.gpt_cond_dim = gpt_cond.shape[-1]
                print(f"Speaker embedding dim: {self.speaker_dim}")
                print(f"GPT conditioning dim: {self.gpt_cond_dim}")
                return
        
        # Fallback to default dimensions
        self.speaker_dim = 512
        self.gpt_cond_dim = 1024
        print("Using default dimensions")
    
    def create_training_dataset(self, voice_descriptions: List[str], audio_files: List[str]) -> Dict:
        """Create training dataset from voice descriptions and audio files"""
        print(f"Creating training dataset from {len(voice_descriptions)} samples...")
        
        sentence_embeddings = self.sentence_model.encode(voice_descriptions)
        xtts_speaker_embeddings = []
        xtts_gpt_cond_latents = []
        
        for i, audio_file in enumerate(audio_files):
            if os.path.exists(audio_file):
                try:
                    gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                        audio_path=[audio_file]
                    )
                    xtts_speaker_embeddings.append(speaker_embedding.cpu().numpy())
                    xtts_gpt_cond_latents.append(gpt_cond_latent.cpu().numpy())
                    print(f"Processed {i+1}/{len(audio_files)}: {audio_file}")
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            else:
                print(f"Audio file not found: {audio_file}")
        
        return {
            'sentence_embeddings': np.array(sentence_embeddings),
            'speaker_embeddings': np.array(xtts_speaker_embeddings),
            'gpt_cond_latents': np.array(xtts_gpt_cond_latents),
            'descriptions': voice_descriptions,
            'audio_files': audio_files
        }
    
    def train_voice_encoder(self, training_data: Dict, epochs: int = 100, batch_size: int = 8):
        """Train the voice description encoder"""
        print(f"Training voice encoder for {epochs} epochs...")
        
        sentence_embs = torch.FloatTensor(training_data['sentence_embeddings']).to(self.device)
        speaker_targets = torch.FloatTensor(training_data['speaker_embeddings']).squeeze().to(self.device)
        gpt_targets = torch.FloatTensor(training_data['gpt_cond_latents']).squeeze().to(self.device)
        
        dataset_size = len(sentence_embs)
        self.voice_encoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)
                
                batch_sentence = sentence_embs[i:end_idx]
                batch_speaker = speaker_targets[i:end_idx]
                batch_gpt = gpt_targets[i:end_idx]
                
                self.optimizer.zero_grad()
                
                pred_speaker, pred_gpt = self.voice_encoder(batch_sentence)
                
                speaker_loss = torch.nn.MSELoss()(pred_speaker, batch_speaker)
                gpt_loss = torch.nn.MSELoss()(pred_gpt, batch_gpt)
                
                total_loss_batch = speaker_loss + gpt_loss
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                total_loss_batch.backward()
                self.optimizer.step()
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        print("Training complete!")
    
    def text_to_speech(
        self, 
        text: str, 
        voice_description: str, 
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """Generate speech from text using voice description"""
        if not self.is_trained:
            raise ValueError("Voice encoder not trained! Run training first.")
        
        sentence_embedding = self.sentence_model.encode([voice_description])
        sentence_tensor = torch.FloatTensor(sentence_embedding).to(self.device)
        
        self.voice_encoder.eval()
        with torch.no_grad():
            speaker_embedding, gpt_cond_latent = self.voice_encoder(sentence_tensor)
        
        outputs = self.xtts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=kwargs.get('temperature', 0.75),
            length_penalty=kwargs.get('length_penalty', 1.0),
            repetition_penalty=kwargs.get('repetition_penalty', 10.0),
            top_k=kwargs.get('top_k', 50),
            top_p=kwargs.get('top_p', 0.85)
        )
        
        return outputs['wav']
    
    def save_voice_encoder(self, path: str):
        """Save trained voice encoder"""
        torch.save({
            'model_state_dict': self.voice_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
            'speaker_dim': self.speaker_dim,
            'gpt_cond_dim': self.gpt_cond_dim
        }, path)
        print(f"Voice encoder saved: {path}")
    
    def load_voice_encoder(self, path: str):
        """Load trained voice encoder"""
        checkpoint = torch.load(path, map_location=self.device)
        self.voice_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint['is_trained']
        print(f"Voice encoder loaded: {path}")