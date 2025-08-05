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
        Initialize CosyVoice interface.
        
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

    def _save_speaker_to_database(self, speaker_id: str, audio_path: str, transcript: str, embedding_data: Optional[Dict] = None):
        """Save speaker information and embedding to database."""
        try:
            speaker_data = {}
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
            
            speaker_entry = {
                'audio_path': str(audio_path),
                'transcript': transcript,
                'created': datetime.now().isoformat()
            }
            
            # Add embedding data if available
            if embedding_data:
                speaker_entry['embedding_data'] = embedding_data
                
            speaker_data[speaker_id] = speaker_entry
            
            # Ensure directory exists
            self.speaker_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.speaker_db_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved speaker {speaker_id} {'with embedding' if embedding_data else ''} to database")
                
        except Exception as e:
            self.logger.error(f"Error saving speaker to database: {e}")
            raise
        """Save speaker information and embedding to database."""
        try:
            speaker_data = {}
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
            
            speaker_entry = {
                'audio_path': str(audio_path),
                'transcript': transcript,
                'created': datetime.now().isoformat()
            }
            
            # Add embedding data if available
            if embedding_data:
                speaker_entry['embedding_data'] = embedding_data
                
            speaker_data[speaker_id] = speaker_entry
            
            # Ensure directory exists
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
            
            # Add speaker embedding
            success = self.cosyvoice.add_zero_shot_spk(transcript, audio_16k, speaker_id)
            
            if success:
                self.logger.info(f"add_zero_shot_spk succeeded for {speaker_id}")
                
                # Call save_spkinfo to ensure spk_db is created
                try:
                    self.cosyvoice.save_spkinfo()
                    self.logger.debug("Called save_spkinfo successfully")
                except Exception as save_error:
                    self.logger.debug(f"save_spkinfo failed: {save_error}")
                
                # Now check for spk_db after save_spkinfo
                embedding_data = None
                try:
                    if hasattr(self.cosyvoice, 'spk_db') and self.cosyvoice.spk_db and speaker_id in self.cosyvoice.spk_db:
                        spk_info = self.cosyvoice.spk_db[speaker_id]
                        self.logger.debug(f"Found speaker in spk_db: {speaker_id}")
                        
                        # Extract embedding tensor directly
                        if 'spk_emb' in spk_info and spk_info['spk_emb'] is not None:
                            embedding_tensor = spk_info['spk_emb']
                            
                            # Convert tensor to saveable format
                            embedding_data = {
                                'spk_emb': embedding_tensor.cpu().numpy().tolist(),
                                'spk_emb_shape': list(embedding_tensor.shape),
                                'prompt_text': spk_info.get('prompt_text', transcript),
                                'embedding_type': 'tensor_converted'
                            }
                            
                            self.logger.info(f"Extracted embedding: {embedding_tensor.shape} tensor -> {len(embedding_data['spk_emb'])} values")
                        else:
                            self.logger.warning("No spk_emb found in spk_info")
                    else:
                        self.logger.warning(f"spk_db not accessible: has_attr={hasattr(self.cosyvoice, 'spk_db')}, exists={bool(getattr(self.cosyvoice, 'spk_db', None))}")
                        
                except Exception as embed_error:
                    self.logger.error(f"Embedding extraction failed: {embed_error}")
                    embedding_data = None
                
                # Save to database with embedding
                self._save_speaker_to_database(speaker_id, str(audio_path), transcript, embedding_data)
                self.logger.info(f"Successfully extracted and saved speaker embedding: {speaker_id}")
                return True
                self.logger.info(f"Successfully extracted and saved speaker embedding: {speaker_id}")
                return True
            else:
                self.logger.error(f"Failed to extract speaker embedding: {speaker_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting speaker embedding: {e}")
            return False

    def synthesize_speech(self, 
                         text: str, 
                         output_filename: str,
                         speaker_id: Optional[str] = None,
                         prompt_audio: Optional[str] = None,
                         prompt_text: str = "",
                         instruction: Optional[str] = None) -> Optional[str]:
        """
        Synthesize speech with various options.
        
        Args:
            text: Text to synthesize
            output_filename: Output filename (without extension)
            speaker_id: Speaker ID for voice cloning
            prompt_audio: Audio prompt for real-time cloning
            prompt_text: Text for audio prompt
            instruction: Natural language instruction for emotion/style
            
        Returns:
            str: Path to generated audio file, None if failed
        """
        try:
            output_path = self.output_base_dir / 'audio' / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Synthesizing: {text[:50]}...")
            
            # Choose synthesis method based on parameters
            if instruction and speaker_id:
                return self._synthesize_with_instruction(text, str(output_path), instruction, speaker_id)
            elif speaker_id:
                return self._synthesize_with_speaker_id(text, str(output_path), speaker_id)
            elif prompt_audio:
                return self._synthesize_with_audio_prompt(text, str(output_path), prompt_audio, prompt_text)
            else:
                self.logger.error("No speaker_id or prompt_audio provided")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in synthesize_speech: {e}")
            return None

    def _synthesize_with_speaker_id(self, text: str, output_path: str, speaker_id: str) -> Optional[str]:
        """Synthesize using saved speaker ID - now with fast embedding loading."""
        try:
            speaker_id = speaker_id.strip('"\'')
            
            # Check if speaker exists in memory
            if not hasattr(self.cosyvoice, 'spk_db') or speaker_id not in self.cosyvoice.spk_db:
                self.logger.warning(f"Speaker '{speaker_id}' not found in memory, attempting reload")
                
                # Try to reload from database with fast embedding restoration
                if self.speaker_db_path.exists():
                    with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                        speaker_data = json.load(f)
                    
                    if speaker_id in speaker_data:
                        data = speaker_data[speaker_id]
                        
                        # Try direct embedding restoration first (much faster)
                        if 'embedding_data' in data and data['embedding_data']:
                            try:
                                embedding_data = data['embedding_data']
                                spk_info = {}
                                
                                if embedding_data.get('spk_emb'):
                                    spk_info['spk_emb'] = torch.tensor(embedding_data['spk_emb'])
                                if embedding_data.get('prompt_text'):
                                    spk_info['prompt_text'] = embedding_data['prompt_text']
                                if embedding_data.get('prompt_speech'):
                                    spk_info['prompt_speech'] = torch.tensor(embedding_data['prompt_speech'])
                                
                                if not hasattr(self.cosyvoice, 'spk_db'):
                                    self.cosyvoice.spk_db = {}
                                self.cosyvoice.spk_db[speaker_id] = spk_info
                                
                                self.logger.info(f"Fast-loaded speaker from embedding: {speaker_id}")
                            except Exception as embed_error:
                                self.logger.warning(f"Embedding restoration failed: {embed_error}, trying audio reload")
                                # Fallback to slower audio loading
                                audio_16k = load_wav(data['audio_path'], 16000)
                                success = self.cosyvoice.add_zero_shot_spk(data['transcript'], audio_16k, speaker_id)
                                if not success:
                                    self.logger.error(f"Failed to reload speaker: {speaker_id}")
                                    return None
                        else:
                            # Legacy audio loading for old database entries
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
                text, '', '', zero_shot_spk_id=speaker_id, stream=False
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

    def _synthesize_with_audio_prompt(self, text: str, output_path: str, prompt_audio: str, prompt_text: str) -> Optional[str]:
        """Synthesize using audio prompt."""
        try:
            if not Path(prompt_audio).exists():
                self.logger.error(f"Prompt audio not found: {prompt_audio}")
                return None
            
            prompt_speech_16k = load_wav(prompt_audio, 16000)
            
            result = self.cosyvoice.inference_zero_shot(
                text, prompt_text, prompt_speech_16k, stream=False
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

    def _synthesize_with_instruction(self, text: str, output_path: str, instruction: str, speaker_id: str) -> Optional[str]:
        """Synthesize with natural language instruction."""
        try:
            speaker_id = speaker_id.strip('"\'')
            
            # Get speaker's original audio
            if self.speaker_db_path.exists():
                with open(self.speaker_db_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                
                if speaker_id in speaker_data:
                    audio_path = speaker_data[speaker_id]['audio_path']
                    prompt_speech_16k = load_wav(audio_path, 16000)
                    
                    try:
                        result = self.cosyvoice.inference_instruct2(
                            text, instruction, prompt_speech_16k,
                            zero_shot_spk_id=speaker_id, stream=False
                        )
                    except Exception as instruct_error:
                        self.logger.warning(f"Instruction synthesis failed: {instruct_error}, falling back to regular synthesis")
                        return self._synthesize_with_speaker_id(text, output_path, speaker_id)
                        
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
            return self._synthesize_with_speaker_id(text, output_path, speaker_id)

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
                
                # Parse line for special formatting
                text, speaker_id, instruction = self._parse_dialogue_line(line)
                
                # Use default speaker if none specified
                if not speaker_id:
                    speaker_id = default_speaker_id
                
                # Generate output filename
                output_filename = f"line_{i:03d}"
                
                # Synthesize
                result_path = None
                if instruction:
                    self.logger.info(f"Using instruction: {instruction}")
                    result_path = self._synthesize_with_instruction(
                        text, str(dialogue_dir / output_filename), instruction, speaker_id
                    )
                elif speaker_id:
                    self.logger.info(f"Using speaker: {speaker_id}")
                    result_path = self._synthesize_with_speaker_id(
                        text, str(dialogue_dir / output_filename), speaker_id
                    )
                else:
                    self.logger.warning("No speaker or instruction specified")
                
                results.append({
                    'line_number': i,
                    'text': text,
                    'speaker_id': speaker_id,
                    'instruction': instruction,
                    'output_path': result_path
                })
            
            # Save processing report
            self._save_dialogue_report(dialogue_dir, script_path, default_speaker_id, results)
            
            self.logger.info(f"Processed {len(results)} dialogue lines")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing dialogue script: {e}")
            return []

    def _parse_dialogue_line(self, line: str) -> tuple:
        """Parse dialogue line for speaker/instruction tags."""
        speaker_id = None
        instruction = None
        text = line
        
        if line.startswith('[') and ']' in line:
            tag_end = line.find(']')
            tag_content = line[1:tag_end]
            text = line[tag_end+1:].strip()
            
            for param in tag_content.split(','):
                param = param.strip()
                if ':' in param:
                    key, value = param.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'speaker':
                        speaker_id = value
                    elif key in ['instruction', 'emotion', 'style']:
                        instruction = value
        
        return text, speaker_id, instruction

    def _save_dialogue_report(self, dialogue_dir: Path, script_path: Path, 
                             default_speaker_id: str, results: List[Dict[str, Any]]):
        """Save dialogue processing report."""
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
                    if result['speaker_id']:
                        f.write(f"  Speaker: {result['speaker_id']}\n")
                    if result['instruction']:
                        f.write(f"  Instruction: {result['instruction']}\n")
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

    def get_stats(self) -> Dict[str, Any]:
        """Get interface statistics."""
        try:
            stats = {
                'model_path': str(self.model_path),
                'output_base_dir': str(self.output_base_dir),
                'speaker_count': len(self.list_speakers()),
                'sample_rate': self.sample_rate,
                'speaker_database_path': str(self.speaker_db_path)
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}