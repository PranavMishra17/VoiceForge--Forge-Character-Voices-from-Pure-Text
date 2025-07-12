#!/usr/bin/env python3
"""
voice_encoder.py - Voice Description Encoder
Neural network to map sentence transformer embeddings to XTTS voice embeddings
"""

import torch
import torch.nn as nn
from typing import Tuple

class VoiceDescriptionEncoder(nn.Module):
    """
    Neural network to map sentence transformer embeddings to XTTS voice embeddings
    """
    def __init__(
        self, 
        sentence_embedding_dim: int = 384,  # all-MiniLM-L6-v2 dimension
        voice_embedding_dim: int = 512,     # XTTS speaker embedding dimension
        gpt_cond_latent_dim: int = 1024,    # XTTS GPT conditioning dimension
        hidden_dim: int = 1024
    ):
        super().__init__()
        
        self.sentence_embedding_dim = sentence_embedding_dim
        self.voice_embedding_dim = voice_embedding_dim
        self.gpt_cond_latent_dim = gpt_cond_latent_dim
        
        # Network to map sentence embeddings to speaker embeddings
        self.speaker_projector = nn.Sequential(
            nn.Linear(sentence_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, voice_embedding_dim),
            nn.Tanh()
        )
        
        # Network to map sentence embeddings to GPT conditioning latents
        self.gpt_cond_projector = nn.Sequential(
            nn.Linear(sentence_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, gpt_cond_latent_dim),
            nn.Tanh()
        )
        
    def forward(self, sentence_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sentence embeddings to voice identity vectors
        
        Args:
            sentence_embeddings: [batch_size, sentence_embedding_dim]
            
        Returns:
            speaker_embedding: [batch_size, voice_embedding_dim]
            gpt_cond_latent: [batch_size, gpt_cond_latent_dim]
        """
        speaker_embedding = self.speaker_projector(sentence_embeddings)
        gpt_cond_latent = self.gpt_cond_projector(sentence_embeddings)
        
        return speaker_embedding, gpt_cond_latent