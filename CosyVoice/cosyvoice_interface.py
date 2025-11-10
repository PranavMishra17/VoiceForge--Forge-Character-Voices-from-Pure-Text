"""
CosyVoice Interface - Main class for external usage
Converts the original main.py functionality into a proper interface
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import torchaudio

# Add CosyVoice paths
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError as e:
    raise ImportError(f"Failed to import CosyVoice modules: {e}. Ensure CosyVoice is properly installed.")


class CosyVoiceInterface:
    """
    Main interface for CosyVoice TTS functionality.
    Provides speaker embedding extraction, voice synthesis, and dialogue processing.
    """
    
    def __init__(self, 
                 model_path: str = 'CosyVoice/pretrained_models/CosyVoice2-0.5B',
                 output_base_dir: str = 'voiceforge_output',
                 speaker_db_path: Optional[str] = None,
                 log_level: str = 'INFO'):
        """
        Initialize CosyVoice interface with enhanced embedding storage.
        
        Args:
            model_path: Path to CosyVoice model
            output_base_dir: Base directory for all outputs
            speaker_db_path: Path to speaker database JSON
            log_level: Logging level
        
        Args:
            model_path: Path to CosyVoice model directory
            output_base_dir: Base directory for all outputs (outside CosyVoice folder)
            speaker_db_path: Custom path for speaker database (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.model_path = Path(model_path)
        self.output_base_dir = Path(output_base_dir)
        
        # Setup directory structure
        self.setup_directories()
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Setup speaker database path
        if speaker_db_path:
            self.speaker_db_path = Path(speaker_db_path)
        else:
            self.speaker_db_path = self.output_base_dir / 'speakers' / 'speaker_database.json'
        
        # Initialize model
        self.cosyvoice = None
        self.sample_rate = None
        self._initialize_model()
        
        # Load existing speakers
        self._load_existing_speakers()
        
        self.logger.info("CosyVoice interface initialized successfully")

    def setup_directories(self):
        """Setup output directory structure."""
        directories = [
            self.output_base_dir,
            self.output_base_dir / 'speakers',
            self.output_base_dir / 'audio',
            self.output_base_dir / 'dialogue',
            self.output_base_dir / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.output_base_dir / 'logs' / f"cosyvoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logger
        self.logger = logging.getLogger('CosyVoiceInterface')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _initialize_model(self):
        """Initialize the CosyVoice model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            self.logger.info(f"Initializing CosyVoice model from: {self.model_path}")
            
            self.cosyvoice = CosyVoice2(
                str(self.model_path),
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=False
            )
            
            self.sample_rate = self.cosyvoice.sample_rate
            self.logger.info("CosyVoice model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CosyVoice model: {e}")
            raise

    def _load_existing_speakers(self):
        """Load previously saved speaker embeddings with direct vector restoration."""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                loaded_speakers = []
                for speaker_id, data in speaker_data.items():
                    try:
                        # Check if we have embedding data for fast loading
                        if 'embedding_data' in data and data['embedding_data']:
                            embedding_data = data['embedding_data']
                            
                            # Restore embedding directly to memory
                            spk_info = {}
                            if embedding_data.get('spk_emb'):
                                spk_info['spk_emb'] = torch.tensor(embedding_data['spk_emb'])
                            if embedding_data.get('prompt_text'):
                                spk_info['prompt_text'] = embedding_data['prompt_text']
                            if embedding_data.get('prompt_speech'):
                                spk_info['prompt_speech'] = torch.tensor(embedding_data['prompt_speech'])
                            
                            # Direct embedding restoration (much faster)
                            if not hasattr(self.cosyvoice, 'spk_db'):
                                self.cosyvoice.spk_db = {}
                            self.cosyvoice.spk_db[speaker_id] = spk_info
                            
                            loaded_speakers.append(speaker_id)
                            self.logger.debug(f"Loaded speaker with direct embedding: {speaker_id}")
                        else:
                            # Fallback to audio loading for legacy entries
                            audio_path = data['audio_path']
                            transcript = data['transcript']
                            
                            if Path(audio_path).exists():
                                audio_16k = load_wav(audio_path, 16000)
                                success = self.cosyvoice.add_zero_shot_spk(transcript, audio_16k, speaker_id)
                                if success:
                                    loaded_speakers.append(speaker_id)
                                    self.logger.debug(f"Loaded speaker from audio (legacy): {speaker_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load speaker {speaker_id}: {e}")
                
                if loaded_speakers:
                    self.logger.info(f"Loaded {len(loaded_speakers)} speakers: {loaded_speakers}")
                else:
                    self.logger.info("No existing speakers found")
            else:
                self.logger.info("No speaker database found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading speakers: {e}")

    def _extract_embedding_from_spk_info(self, spk_info):
        """Helper method to extract embedding data from spk_info object."""
        try:
            self.logger.debug(f"Extracting from spk_info type: {type(spk_info)}")
            self.logger.debug(f"spk_info keys: {spk_info.keys() if isinstance(spk_info, dict) else 'Not a dict'}")
            
            embedding_data = {}
            
            # Extract speaker embedding
            if 'spk_emb' in spk_info and spk_info['spk_emb'] is not None:
                if hasattr(spk_info['spk_emb'], 'cpu'):
                    embedding_data['spk_emb'] = spk_info['spk_emb'].cpu().numpy().tolist()
                else:
                    embedding_data['spk_emb'] = spk_info['spk_emb']
                    
            # Extract prompt text
            if 'prompt_text' in spk_info:
                embedding_data['prompt_text'] = spk_info['prompt_text']
                
            # Extract prompt speech
            if 'prompt_speech' in spk_info and spk_info['prompt_speech'] is not None:
                if hasattr(spk_info['prompt_speech'], 'cpu'):
                    embedding_data['prompt_speech'] = spk_info['prompt_speech'].cpu().numpy().tolist()
                else:
                    embedding_data['prompt_speech'] = spk_info['prompt_speech']
            
            if embedding_data.get('spk_emb'):
                self.logger.info(f"Extracted embedding vector: {len(embedding_data['spk_emb'])} dimensions")
                return embedding_data
            else:
                self.logger.warning("No speaker embedding found in spk_info")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting from spk_info: {e}")
            return None

    def _extract_full_embedding_data(self, spk_info, transcript, embedding_tensor):
        """Extract comprehensive embedding data from spk_info."""
        try:
            embedding_data = {
                'spk_emb': embedding_tensor.cpu().numpy().tolist(),
                'spk_emb_shape': list(embedding_tensor.shape),
                'prompt_text': transcript,
                'embedding_type': 'full_extracted',
                'embedding_dim': embedding_tensor.shape[-1] if len(embedding_tensor.shape) > 1 else len(embedding_tensor)
            }
            
            # Extract additional data from spk_info if available
            if isinstance(spk_info, dict):
                # Extract llm_embedding if available
                if 'llm_embedding' in spk_info and spk_info['llm_embedding'] is not None:
                    if hasattr(spk_info['llm_embedding'], 'cpu'):
                        embedding_data['llm_embedding'] = spk_info['llm_embedding'].cpu().numpy().tolist()
                    else:
                        embedding_data['llm_embedding'] = spk_info['llm_embedding']
                
                # Extract flow_embedding if available
                if 'flow_embedding' in spk_info and spk_info['flow_embedding'] is not None:
                    if hasattr(spk_info['flow_embedding'], 'cpu'):
                        embedding_data['flow_embedding'] = spk_info['flow_embedding'].cpu().numpy().tolist()
                    else:
                        embedding_data['flow_embedding'] = spk_info['flow_embedding']
                
                # Extract speech tokens if available
                if 'llm_prompt_speech_token' in spk_info and spk_info['llm_prompt_speech_token'] is not None:
                    if hasattr(spk_info['llm_prompt_speech_token'], 'cpu'):
                        embedding_data['speech_token'] = spk_info['llm_prompt_speech_token'].cpu().numpy().tolist()
                    else:
                        embedding_data['speech_token'] = spk_info['llm_prompt_speech_token']
                
                # Extract text tokens if available
                if 'prompt_text_token' in spk_info and spk_info['prompt_text_token'] is not None:
                    if hasattr(spk_info['prompt_text_token'], 'cpu'):
                        embedding_data['text_token'] = spk_info['prompt_text_token'].cpu().numpy().tolist()
                    else:
                        embedding_data['text_token'] = spk_info['prompt_text_token']
            
            return embedding_data
            
        except Exception as e:
            self.logger.error(f"Error extracting full embedding data: {e}")
            # Fallback to basic extraction
            return {
                'spk_emb': embedding_tensor.cpu().numpy().tolist(),
                'spk_emb_shape': list(embedding_tensor.shape),
                'prompt_text': transcript,
                'embedding_type': 'fallback_extracted'
            }

    def _save_speaker_to_database(self, speaker_id: str, audio_path: str, transcript: str, embedding_data: Optional[Dict] = None):
        """Save speaker information and embedding to database."""
        try:
            # Ensure output directory structure exists
            output_base = Path(self.output_base_dir)
            embeddings_dir = output_base / 'speakers' / 'embeddings'
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            speaker_data = {}
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
            
            speaker_entry = {
                'audio_path': str(audio_path),
                'transcript': transcript,
                'created': datetime.now().isoformat()
            }
            
            # Save embedding data in multiple formats
            if embedding_data and 'spk_emb' in embedding_data:
                # Save JSON format (for human readability and portability)
                embedding_json_path = embeddings_dir / f"{speaker_id}_embedding.json"
                with open(embedding_json_path, 'w') as f:
                    json.dump(embedding_data, f, indent=2)
                
                # Save NumPy format (for fast loading)
                embedding_npy_path = embeddings_dir / f"{speaker_id}_embedding.npy"
                embedding_array = np.array(embedding_data['spk_emb'])
                np.save(embedding_npy_path, embedding_array)
                
                # Add paths to database entry
                speaker_entry['embedding_data'] = embedding_data
                speaker_entry['embedding_paths'] = {
                    'json': str(embedding_json_path.relative_to(output_base)),
                    'npy': str(embedding_npy_path.relative_to(output_base))
                }
                
            speaker_data[speaker_id] = speaker_entry
            
            # Save updated database
            self.speaker_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.speaker_db_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved speaker {speaker_id} {'with embedding' if embedding_data else ''} to database")
                
        except Exception as e:
            self.logger.error(f"Error saving speaker to database: {e}")
            raise

    def extract_speaker_embedding(self, audio_path: str, transcript: str, speaker_id: str) -> bool:
        """
        Extract and save speaker embedding with actual vector storage.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript of the audio
            speaker_id: Unique identifier for the speaker
            
        Returns:
            bool: Success status
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                return False
            
            self.logger.info(f"Extracting speaker embedding for: {speaker_id}")
            
            # Load audio at 16kHz
            audio_16k = load_wav(str(audio_path), 16000)
            
            # Extract embedding directly using frontend method
            embedding_tensor = self.cosyvoice.frontend._extract_spk_embedding(audio_16k)
            self.logger.info(f"Direct embedding extraction: {embedding_tensor.shape}")
            
            # Add speaker with proper spk_info structure
            success = self.cosyvoice.add_zero_shot_spk(transcript, audio_16k, speaker_id)
            
            if success:
                self.logger.info(f"add_zero_shot_spk succeeded for {speaker_id}")
                
                # Extract comprehensive embedding data
                embedding_data = None
                try:
                    # Check if we can access the speaker info
                    if hasattr(self.cosyvoice, 'frontend') and hasattr(self.cosyvoice.frontend, 'spk2info'):
                        spk2info = self.cosyvoice.frontend.spk2info
                        if speaker_id in spk2info:
                            spk_info = spk2info[speaker_id]
                            self.logger.debug(f"Found speaker in spk2info: {speaker_id}")
                            
                            # Extract all relevant embedding information
                            embedding_data = self._extract_full_embedding_data(spk_info, transcript, embedding_tensor)
                            self.logger.info(f"Extracted full embedding data: {len(embedding_data.get('spk_emb', []))} dimensions")
                        else:
                            self.logger.warning(f"Speaker {speaker_id} not found in spk2info")
                    else:
                        self.logger.warning("Could not access spk2info")
                        
                except Exception as embed_error:
                    self.logger.error(f"Embedding extraction failed: {embed_error}")
                    # Fallback to direct tensor
                    embedding_data = {
                        'spk_emb': embedding_tensor.cpu().numpy().tolist(),
                        'spk_emb_shape': list(embedding_tensor.shape),
                        'prompt_text': transcript,
                        'embedding_type': 'direct_extracted'
                    }
                
                # Save to database with embedding
                self._save_speaker_to_database(speaker_id, str(audio_path), transcript, embedding_data)
                self.logger.info(f"Successfully extracted and saved speaker embedding: {speaker_id}")
                return True
            else:
                self.logger.error(f"Failed to extract speaker embedding: {speaker_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting speaker embedding: {e}")
            return False

    def add_speaker_from_embedding(self, 
                                  speaker_id: str, 
                                  embedding_vector: List[float], 
                                  prompt_text: str = "",
                                  embedding_metadata: Optional[Dict] = None) -> bool:
        """
        Add a speaker directly from embedding vector without audio file.
        
        Args:
            speaker_id: Unique identifier for the speaker
            embedding_vector: 192-dimensional speaker embedding vector
            prompt_text: Text associated with the embedding
            embedding_metadata: Additional metadata about the embedding
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Adding speaker from direct embedding: {speaker_id}")
            
            # Validate embedding dimension
            if len(embedding_vector) != 192:
                self.logger.error(f"Invalid embedding dimension: {len(embedding_vector)}. Expected 192.")
                return False
            
            # Convert to tensor
            embedding_tensor = torch.tensor([embedding_vector], dtype=torch.float32).to(self.cosyvoice.frontend.device)
            
            # Tokenize the prompt text
            prompt_text_token, prompt_text_token_len = self.cosyvoice.frontend._extract_text_token(prompt_text)
            
            # Create complete spk_info structure as expected by CosyVoice2
            spk_info = {
                'llm_embedding': embedding_tensor,
                'flow_embedding': embedding_tensor,
                'prompt_text': prompt_text_token,
                'prompt_text_len': prompt_text_token_len
            }
            
            # Add to spk2info
            if not hasattr(self.cosyvoice.frontend, 'spk2info'):
                self.cosyvoice.frontend.spk2info = {}
            
            self.cosyvoice.frontend.spk2info[speaker_id] = spk_info
            
            # Prepare embedding data for database
            embedding_data = {
                'spk_emb': embedding_vector,
                'spk_emb_shape': [1, 192],
                'prompt_text': prompt_text,
                'embedding_type': 'direct_input',
                'embedding_dim': 192
            }
            
            if embedding_metadata:
                embedding_data.update(embedding_metadata)
            
            # Save to database (without audio path)
            self._save_speaker_to_database(speaker_id, "DIRECT_EMBEDDING", prompt_text, embedding_data)
            
            self.logger.info(f"Successfully added speaker from embedding: {speaker_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding speaker from embedding: {e}")
            return False

    def synthesize_speech(self, 
                         text: str, 
                         output_filename: str,
                         speaker_id: Optional[str] = None,
                         prompt_audio: Optional[str] = None,
                         prompt_text: str = "",
                         instruction: Optional[str] = None,
                         emotion: Optional[str] = None,
                         tone: Optional[str] = None,
                         speed: float = 1.0,
                         language: Optional[str] = None) -> Optional[str]:
        """
        Enhanced synthesize speech with comprehensive emotion and tone control.
        
        Args:
            text: Text to synthesize
            output_filename: Output filename (without extension)
            speaker_id: Speaker ID for voice cloning
            prompt_audio: Audio prompt for real-time cloning
            prompt_text: Text for audio prompt
            instruction: Natural language instruction for emotion/style
            emotion: Emotion keyword (happy, sad, angry, excited, etc.)
            tone: Tone specification (formal, casual, dramatic, etc.)
            speed: Speech speed multiplier (default 1.0)
            language: Language specification for multilingual synthesis
            
        Returns:
            str: Path to generated audio file, None if failed
        """
        try:
            output_path = self.output_base_dir / 'audio' / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Synthesizing: {text[:50]}...")
            self.logger.info(f"Parameters - Speaker: {speaker_id}, Emotion: {emotion}, Tone: {tone}, Speed: {speed}")
            
            # Build enhanced instruction from emotion and tone
            enhanced_instruction = self._build_enhanced_instruction(instruction, emotion, tone, language)
            
            # Preprocess text with language tags if specified
            processed_text = self._preprocess_text_with_language(text, language)
            
            # Choose synthesis method based on parameters
            if enhanced_instruction and speaker_id:
                return self._synthesize_with_instruction(processed_text, str(output_path), enhanced_instruction, speaker_id, speed)
            elif speaker_id:
                return self._synthesize_with_speaker_id(processed_text, str(output_path), speaker_id, speed)
            elif prompt_audio:
                return self._synthesize_with_audio_prompt(processed_text, str(output_path), prompt_audio, prompt_text, speed)
            else:
                self.logger.error("No speaker_id or prompt_audio provided")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in synthesize_speech: {e}")
            return None

    def _build_enhanced_instruction(self, instruction: Optional[str], emotion: Optional[str],
                                   tone: Optional[str], language: Optional[str]) -> Optional[str]:
        """
        Build enhanced instruction text from emotion and tone parameters.

        IMPORTANT: Language is NOT included in instructions - language should ONLY be
        specified via language tags (e.g., <|zh|>, <|en|>) prepended to the text.
        Language instructions confuse the model and cause it to output in default language.
        """
        parts = []

        if instruction:
            parts.append(instruction)

        if emotion:
            emotion_instructions = {
                'happy': 'speak with joy and happiness',
                'sad': 'speak with sadness and melancholy',
                'angry': 'speak with anger and intensity',
                'excited': 'speak with excitement and enthusiasm',
                'calm': 'speak calmly and peacefully',
                'nervous': 'speak nervously with hesitation',
                'confident': 'speak with confidence and authority',
                'mysterious': 'speak mysteriously and enigmatically',
                'dramatic': 'speak dramatically with emphasis',
                'gentle': 'speak gently and softly',
                'energetic': 'speak with high energy and vigor',
                'romantic': 'speak romantically and lovingly'
            }
            if emotion.lower() in emotion_instructions:
                parts.append(emotion_instructions[emotion.lower()])
            else:
                parts.append(f"speak with {emotion}")

        if tone:
            tone_instructions = {
                'formal': 'use a formal tone',
                'casual': 'use a casual and relaxed tone',
                'professional': 'use a professional business tone',
                'friendly': 'use a warm and friendly tone',
                'serious': 'use a serious and solemn tone',
                'playful': 'use a playful and lighthearted tone',
                'authoritative': 'use an authoritative and commanding tone',
                'whispering': 'whisper softly',
                'shouting': 'speak loudly and forcefully'
            }
            if tone.lower() in tone_instructions:
                parts.append(tone_instructions[tone.lower()])
            else:
                parts.append(f"use a {tone} tone")

        # NOTE: Language is intentionally NOT added here - it's handled via language tags
        # Adding language instructions here causes the model to ignore language tags

        return ', '.join(parts) if parts else None

    def _preprocess_text_with_language(self, text: str, language: Optional[str]) -> str:
        """
        Preprocess text with language tags for multilingual synthesis.

        Language tags must be prepended to the text for CosyVoice to recognize the target language.
        These tags are processed by the tokenizer and guide the model's language generation.

        Supported languages based on CosyVoice tokenizer:
        - Chinese/English/Japanese/Korean: Primary languages
        - Cantonese/Minnan/Wuyu: Chinese dialects
        - Many other languages via standard codes
        """
        if not language:
            return text

        # Comprehensive language tag mapping based on CosyVoice tokenizer
        # See CosyVoice/cosyvoice/tokenizer/tokenizer.py for full LANGUAGES dict
        language_tags = {
            # Primary languages
            'chinese': '<|zh|>',
            'mandarin': '<|zh|>',
            'english': '<|en|>',
            'japanese': '<|ja|>',  # Fixed: was <|jp|>, should be <|ja|>
            'korean': '<|ko|>',

            # Chinese dialects and variants
            'cantonese': '<|yue|>',
            'yue': '<|yue|>',
            'minnan': '<|minnan|>',
            'wuyu': '<|wuyu|>',
            'sichuanese': '<|dialect|>',  # Generic dialect tag for unsupported dialects
            'shanghainese': '<|wuyu|>',  # Shanghai dialect is part of Wu Chinese
            'tianjinese': '<|dialect|>',
            'wuhanese': '<|dialect|>',

            # Mixed language
            'zh/en': '<|zh/en|>',
            'en/zh': '<|en/zh|>',
            'mixed': '<|zh/en|>',

            # Additional languages supported by tokenizer
            'german': '<|de|>',
            'spanish': '<|es|>',
            'russian': '<|ru|>',
            'french': '<|fr|>',
            'portuguese': '<|pt|>',
            'turkish': '<|tr|>',
            'polish': '<|pl|>',
            'catalan': '<|ca|>',
            'dutch': '<|nl|>',
            'arabic': '<|ar|>',
            'swedish': '<|sv|>',
            'italian': '<|it|>',
            'indonesian': '<|id|>',
            'hindi': '<|hi|>',
            'finnish': '<|fi|>',
            'vietnamese': '<|vi|>',
            'hebrew': '<|he|>',
            'ukrainian': '<|uk|>',
            'greek': '<|el|>',
            'malay': '<|ms|>',
            'czech': '<|cs|>',
            'romanian': '<|ro|>',
            'danish': '<|da|>',
            'hungarian': '<|hu|>',
            'tamil': '<|ta|>',
            'norwegian': '<|no|>',
            'thai': '<|th|>',
        }

        lang_lower = language.lower()
        if lang_lower in language_tags:
            tag = language_tags[lang_lower]
            self.logger.info(f"Applying language tag: {tag} for language: {language}")
            return f"{tag}{text}"
        else:
            self.logger.warning(f"Unknown language '{language}', no tag applied. Text will use default language detection.")
            return text

    def _synthesize_with_speaker_id(self, text: str, output_path: str, speaker_id: str, speed: float = 1.0) -> Optional[str]:
        """Synthesize using saved speaker ID with optimized embedding loading."""
        try:
            speaker_id = speaker_id.strip('"\'')
            
            # Check if speaker exists in memory (spk2info for CosyVoice2)
            spk2info = getattr(self.cosyvoice.frontend, 'spk2info', {})
            if speaker_id not in spk2info:
                self.logger.warning(f"Speaker '{speaker_id}' not found in memory, attempting reload")
                
                # Try to reload from database with fast embedding restoration
                if self.speaker_db_path.exists():
                    with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                        speaker_data = json.load(f)
                    
                    if speaker_id in speaker_data:
                        data = speaker_data[speaker_id]
                        
                        # Try loading from NPY file first (fastest)
                        if 'embedding_paths' in data and 'npy' in data['embedding_paths']:
                            try:
                                npy_path = Path(self.output_base_dir) / data['embedding_paths']['npy']
                                if npy_path.exists():
                                    embedding_array = np.load(npy_path)
                                    # Ensure proper shape - should be (1, 192) for CosyVoice
                                    if embedding_array.ndim == 1:
                                        embedding_array = embedding_array.reshape(1, -1)
                                    elif embedding_array.ndim > 2:
                                        embedding_array = embedding_array.reshape(1, -1)
                                    
                                    # Create proper spk_info for CosyVoice2
                                    embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32).to(self.cosyvoice.frontend.device)
                                    
                                    # Debug logging
                                    self.logger.debug(f"Loaded embedding shape: {embedding_tensor.shape}")
                                    
                                    # Tokenize the prompt text
                                    prompt_text = data.get('transcript', '')
                                    prompt_text_token, prompt_text_token_len = self.cosyvoice.frontend._extract_text_token(prompt_text)
                                    
                                    # Create complete spk_info structure as expected by CosyVoice2
                                    spk_info = {
                                        'llm_embedding': embedding_tensor,
                                        'flow_embedding': embedding_tensor,
                                        'prompt_text': prompt_text_token,
                                        'prompt_text_len': prompt_text_token_len
                                    }
                                    
                                    self.cosyvoice.frontend.spk2info[speaker_id] = spk_info
                                    
                                    self.logger.info(f"Fast-loaded speaker from NPY: {speaker_id}")
                                    
                                else:
                                    raise FileNotFoundError(f"NPY file not found: {npy_path}")
                                    
                            except Exception as npy_error:
                                self.logger.warning(f"NPY loading failed: {npy_error}, trying JSON")
                                
                                # Try JSON as backup
                                try:
                                    if 'embedding_paths' in data and 'json' in data['embedding_paths']:
                                        json_path = Path(self.output_base_dir) / data['embedding_paths']['json']
                                        if json_path.exists():
                                            with open(json_path, 'r') as f:
                                                embedding_data = json.load(f)
                                            
                                            # Create proper spk_info for CosyVoice2
                                            embedding_array = np.array(embedding_data['spk_emb'])
                                            if embedding_array.ndim == 1:
                                                embedding_array = embedding_array.reshape(1, -1)
                                            elif embedding_array.ndim > 2:
                                                embedding_array = embedding_array.reshape(1, -1)
                                            
                                            embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32).to(self.cosyvoice.frontend.device)
                                            
                                            # Debug logging
                                            self.logger.debug(f"Loaded embedding shape: {embedding_tensor.shape}")
                                            
                                            # Tokenize the prompt text
                                            prompt_text = embedding_data.get('prompt_text', data.get('transcript', ''))
                                            prompt_text_token, prompt_text_token_len = self.cosyvoice.frontend._extract_text_token(prompt_text)
                                            
                                            # Create complete spk_info structure as expected by CosyVoice2
                                            spk_info = {
                                                'llm_embedding': embedding_tensor,
                                                'flow_embedding': embedding_tensor,
                                                'prompt_text': prompt_text_token,
                                                'prompt_text_len': prompt_text_token_len
                                            }
                                            
                                            self.cosyvoice.frontend.spk2info[speaker_id] = spk_info
                                            
                                            self.logger.info(f"Loaded speaker from JSON: {speaker_id}")
                                        else:
                                            raise FileNotFoundError(f"JSON file not found: {json_path}")
                                    else:
                                        raise ValueError("No JSON path in database")
                                        
                                except Exception as json_error:
                                    self.logger.warning(f"JSON loading failed: {json_error}, trying audio reload")
                                    # Final fallback to audio loading
                                    audio_16k = load_wav(data['audio_path'], 16000)
                                    success = self.cosyvoice.add_zero_shot_spk(data['transcript'], audio_16k, speaker_id)
                                    if not success:
                                        self.logger.error(f"Failed to reload speaker: {speaker_id}")
                                        return None
                        else:
                            # Legacy loading for old database entries
                            self.logger.warning("No embedding paths found, using legacy audio loading")
                            audio_16k = load_wav(data['audio_path'], 16000)
                            success = self.cosyvoice.add_zero_shot_spk(data['transcript'], audio_16k, speaker_id)
                            if not success:
                                self.logger.error(f"Failed to reload speaker: {speaker_id}")
                                return None
                    else:
                        self.logger.error(f"Speaker '{speaker_id}' not found in database")
                        return None
                else:
                    self.logger.error("No speaker database found")
                    return None
            
            # Generate speech (now much faster with direct embedding)
            result = self.cosyvoice.inference_zero_shot(
                text, '', '', zero_shot_spk_id=speaker_id, stream=False, speed=speed
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.sample_rate)
                self.logger.info(f"Generated audio with speaker {speaker_id}: {final_path}")
                return final_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _synthesize_with_speaker_id: {e}")
            return None

    def _synthesize_with_audio_prompt(self, text: str, output_path: str, prompt_audio: str, prompt_text: str, speed: float = 1.0) -> Optional[str]:
        """Synthesize using audio prompt."""
        try:
            if not Path(prompt_audio).exists():
                self.logger.error(f"Prompt audio not found: {prompt_audio}")
                return None
            
            prompt_speech_16k = load_wav(prompt_audio, 16000)
            
            result = self.cosyvoice.inference_zero_shot(
                text, prompt_text, prompt_speech_16k, stream=False, speed=speed
            )
            
            for i, j in enumerate(result):
                final_path = f'{output_path}_output_{i}.wav'
                torchaudio.save(final_path, j['tts_speech'], self.sample_rate)
                self.logger.info(f"Generated audio with prompt: {final_path}")
                return final_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _synthesize_with_audio_prompt: {e}")
            return None

    def _synthesize_with_instruction(self, text: str, output_path: str, instruction: str, speaker_id: str, speed: float = 1.0) -> Optional[str]:
        """Synthesize with natural language instruction."""
        try:
            speaker_id = speaker_id.strip('"\'')
            
            # Check if speaker exists in memory first
            spk2info = getattr(self.cosyvoice.frontend, 'spk2info', {})
            if speaker_id not in spk2info:
                self.logger.warning(f"Speaker '{speaker_id}' not found in memory, attempting reload")
                # Try to reload speaker using the same logic as _synthesize_with_speaker_id
                reload_result = self._synthesize_with_speaker_id(text, output_path, speaker_id, speed)
                if reload_result is None:
                    return None
            
            # Get speaker's audio or use a default audio for instruction synthesis
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                if speaker_id in speaker_data:
                    audio_path = speaker_data[speaker_id]['audio_path']
                    
                    # Handle direct embedding speakers (no audio file)
                    if audio_path == "DIRECT_EMBEDDING":
                        self.logger.info(f"Using direct embedding for instruction synthesis: {speaker_id}")
                        # For direct embeddings, we'll use the speaker_id approach instead
                        # since inference_instruct2 requires audio input
                        return self._synthesize_with_speaker_id(text, output_path, speaker_id, speed)
                    else:
                        # Load audio file for instruction synthesis
                        prompt_speech_16k = load_wav(audio_path, 16000)
                        
                        try:
                            result = self.cosyvoice.inference_instruct2(
                                text, instruction, prompt_speech_16k,
                                zero_shot_spk_id=speaker_id, stream=False, speed=speed
                            )
                        except Exception as instruct_error:
                            self.logger.warning(f"Instruction synthesis failed: {instruct_error}, falling back to regular synthesis")
                            return self._synthesize_with_speaker_id(text, output_path, speaker_id, speed)
                            
                        for i, j in enumerate(result):
                            final_path = f'{output_path}_output_{i}.wav'
                            torchaudio.save(final_path, j['tts_speech'], self.sample_rate)
                            self.logger.info(f"Generated instructed audio: {final_path}")
                            return final_path
                else:
                    self.logger.error(f"Speaker '{speaker_id}' not found in database")
                    return None
            else:
                self.logger.error("Speaker database not found")
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _synthesize_with_instruction: {e}")
            # Fallback to regular synthesis
            return self._synthesize_with_speaker_id(text, output_path, speaker_id, speed)

    def process_dialogue_script(self, 
                               script_path: str, 
                               dialogue_name: str,
                               default_speaker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process dialogue script from text file.
        
        Args:
            script_path: Path to dialogue script file
            dialogue_name: Name for the dialogue session
            default_speaker_id: Default speaker if none specified in script
            
        Returns:
            List[Dict]: Results for each line processed
        """
        try:
            script_path = Path(script_path)
            if not script_path.exists():
                self.logger.error(f"Script file not found: {script_path}")
                return []
            
            # Create dialogue output directory
            dialogue_dir = self.output_base_dir / 'dialogue' / dialogue_name
            dialogue_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Processing dialogue script: {script_path}")
            
            # Read script
            with open(script_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            results = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                self.logger.info(f"Processing line {i}: {line[:50]}...")
                
                # Parse line for special formatting with full parameter support
                parsed = self._parse_dialogue_line(line)
                text = parsed['text']
                speaker_id = parsed.get('speaker_id')
                instruction = parsed.get('instruction')
                emotion = parsed.get('emotion')
                tone = parsed.get('tone')
                language = parsed.get('language')
                speed = parsed.get('speed', 1.0)

                # Use default speaker if none specified
                if not speaker_id:
                    speaker_id = default_speaker_id

                # Generate output filename
                output_filename = f"line_{i:03d}"

                # Synthesize with full parameter support
                result_path = self.synthesize_speech(
                    text=text,
                    output_filename=str(dialogue_dir / output_filename),
                    speaker_id=speaker_id,
                    instruction=instruction,
                    emotion=emotion,
                    tone=tone,
                    speed=speed,
                    language=language
                )

                results.append({
                    'line_number': i,
                    'text': text,
                    'speaker_id': speaker_id,
                    'instruction': instruction,
                    'emotion': emotion,
                    'tone': tone,
                    'language': language,
                    'speed': speed,
                    'output_path': result_path
                })
            
            # Save processing report
            self._save_dialogue_report(dialogue_dir, script_path, default_speaker_id, results)
            
            self.logger.info(f"Processed {len(results)} dialogue lines")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing dialogue script: {e}")
            return []

    def _parse_dialogue_line(self, line: str) -> dict:
        """
        Parse dialogue line for speaker/instruction/emotion/tone/language/speed tags.

        Supported tags:
        - speaker: Speaker ID
        - instruction: Natural language instruction for voice modulation
        - emotion: Emotion keyword (happy, sad, angry, etc.)
        - tone: Tone specification (formal, casual, dramatic, etc.)
        - language: Language for synthesis (english, chinese, japanese, etc.)
        - speed: Speech speed multiplier (0.5-2.0)

        Example formats:
        - [speaker:wizard_voice] Hello there!
        - [speaker:wizard_voice,emotion:excited] I'm so happy!
        - [speaker:narrator,tone:dramatic,language:english] Once upon a time...
        - [emotion:sad,tone:whispering,speed:0.8] I'm so sorry...
        """
        result = {
            'text': line,
            'speaker_id': None,
            'instruction': None,
            'emotion': None,
            'tone': None,
            'language': None,
            'speed': 1.0
        }

        if line.startswith('[') and ']' in line:
            tag_end = line.find(']')
            tag_content = line[1:tag_end]
            result['text'] = line[tag_end+1:].strip()

            for param in tag_content.split(','):
                param = param.strip()
                if ':' in param:
                    key, value = param.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == 'speaker':
                        result['speaker_id'] = value
                    elif key == 'instruction':
                        result['instruction'] = value
                    elif key == 'emotion':
                        result['emotion'] = value
                    elif key in ['tone', 'style']:  # 'style' is alias for 'tone'
                        result['tone'] = value
                    elif key in ['language', 'lang']:  # 'lang' is alias for 'language'
                        result['language'] = value
                    elif key == 'speed':
                        try:
                            result['speed'] = float(value)
                        except ValueError:
                            self.logger.warning(f"Invalid speed value '{value}', using default 1.0")
                            result['speed'] = 1.0

        return result

    def _save_dialogue_report(self, dialogue_dir: Path, script_path: Path,
                             default_speaker_id: str, results: List[Dict[str, Any]]):
        """Save comprehensive dialogue processing report with all parameters."""
        try:
            report_path = dialogue_dir / 'processing_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Dialogue Processing Report\n")
                f.write(f"========================\n")
                f.write(f"Script: {script_path}\n")
                f.write(f"Default Speaker: {default_speaker_id}\n")
                f.write(f"Processed: {len(results)} lines\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")

                for result in results:
                    f.write(f"Line {result['line_number']}: {result['text'][:50]}...\n")
                    if result.get('speaker_id'):
                        f.write(f"  Speaker: {result['speaker_id']}\n")
                    if result.get('language'):
                        f.write(f"  Language: {result['language']}\n")
                    if result.get('emotion'):
                        f.write(f"  Emotion: {result['emotion']}\n")
                    if result.get('tone'):
                        f.write(f"  Tone: {result['tone']}\n")
                    if result.get('instruction'):
                        f.write(f"  Instruction: {result['instruction']}\n")
                    if result.get('speed') and result['speed'] != 1.0:
                        f.write(f"  Speed: {result['speed']}x\n")
                    f.write(f"  Output: {result['output_path']}\n\n")

            self.logger.info(f"Saved dialogue report: {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving dialogue report: {e}")

    def list_speakers(self) -> List[str]:
        """List available speaker IDs."""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                return list(speaker_data.keys())
            return []
        except Exception as e:
            self.logger.error(f"Error listing speakers: {e}")
            return []

    def get_speaker_info(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific speaker."""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                return speaker_data.get(speaker_id)
            return None
        except Exception as e:
            self.logger.error(f"Error getting speaker info: {e}")
            return None

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the database."""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                if speaker_id in speaker_data:
                    del speaker_data[speaker_id]
                    
                    with open(self.speaker_db_path, 'w', encoding='utf-8') as f:
                        json.dump(speaker_data, f, indent=2, ensure_ascii=False)
                    
                    # Remove from memory if loaded
                    if hasattr(self.cosyvoice, 'spk_db') and speaker_id in self.cosyvoice.spk_db:
                        del self.cosyvoice.spk_db[speaker_id]
                    
                    self.logger.info(f"Deleted speaker: {speaker_id}")
                    return True
                else:
                    self.logger.warning(f"Speaker not found: {speaker_id}")
                    return False
            return False
        except Exception as e:
            self.logger.error(f"Error deleting speaker: {e}")
            return False

    def get_speaker_embedding(self, speaker_id: str) -> Optional[List[float]]:
        """Get the embedding vector for a specific speaker."""
        try:
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                if speaker_id in speaker_data and 'embedding_data' in speaker_data[speaker_id]:
                    embedding_data = speaker_data[speaker_id]['embedding_data']
                    if 'spk_emb' in embedding_data:
                        return embedding_data['spk_emb']
            return None
        except Exception as e:
            self.logger.error(f"Error getting speaker embedding: {e}")
            return None

    def export_speaker_embeddings(self, output_path: str) -> bool:
        """Export all speaker embeddings to a single file."""
        try:
            embeddings = {}
            speakers = self.list_speakers()
            
            for speaker_id in speakers:
                embedding = self.get_speaker_embedding(speaker_id)
                if embedding:
                    embeddings[speaker_id] = {
                        'embedding': embedding,
                        'info': self.get_speaker_info(speaker_id)
                    }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(embeddings)} speaker embeddings to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting speaker embeddings: {e}")
            return False

    def import_speaker_embeddings(self, import_path: str) -> int:
        """Import speaker embeddings from a file."""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return 0
            
            with open(import_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            imported_count = 0
            for speaker_id, data in embeddings_data.items():
                if 'embedding' in data and data['embedding']:
                    success = self.add_speaker_from_embedding(
                        speaker_id=speaker_id,
                        embedding_vector=data['embedding'],
                        prompt_text=data.get('info', {}).get('transcript', ''),
                        embedding_metadata=data.get('info', {})
                    )
                    if success:
                        imported_count += 1
            
            self.logger.info(f"Imported {imported_count} speakers from {import_path}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Error importing speaker embeddings: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get interface statistics."""
        try:
            speakers = self.list_speakers()
            embedding_counts = {'with_embeddings': 0, 'without_embeddings': 0}
            
            for speaker_id in speakers:
                if self.get_speaker_embedding(speaker_id):
                    embedding_counts['with_embeddings'] += 1
                else:
                    embedding_counts['without_embeddings'] += 1
            
            stats = {
                'model_path': str(self.model_path),
                'output_base_dir': str(self.output_base_dir),
                'speaker_count': len(speakers),
                'embedding_counts': embedding_counts,
                'sample_rate': self.sample_rate,
                'speaker_database_path': str(self.speaker_db_path),
                'supported_features': {
                    'direct_embedding_input': True,
                    'emotion_control': True,
                    'tone_control': True,
                    'speed_control': True,
                    'multilingual': True,
                    'instruction_synthesis': True
                }
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}