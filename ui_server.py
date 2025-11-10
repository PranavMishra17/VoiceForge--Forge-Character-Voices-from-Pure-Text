#!/usr/bin/env python3
import os
import io
import json
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the same interface used by main.py
from CosyVoice.cosyvoice_interface import CosyVoiceInterface

ROOT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_PATH = str(ROOT_DIR / 'CosyVoice' / 'pretrained_models' / 'CosyVoice2-0.5B')
DEFAULT_OUTPUT_DIR = str(ROOT_DIR / 'voiceforge_output')
DEFAULT_SPEAKER_DB = str(Path(DEFAULT_OUTPUT_DIR) / 'speakers' / 'speaker_database.json')
WEB_DIR = ROOT_DIR / 'webui'

# Ensure default dirs exist using DEFAULT_OUTPUT_DIR (before interface is created)
Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
(Path(DEFAULT_OUTPUT_DIR) / 'audio').mkdir(parents=True, exist_ok=True)
(Path(DEFAULT_OUTPUT_DIR) / 'uploads').mkdir(parents=True, exist_ok=True)

# Create app
app = FastAPI(title="VoiceForge UI Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global interface instance
voice_forge = CosyVoiceInterface(
    model_path=DEFAULT_MODEL_PATH,
    output_base_dir=DEFAULT_OUTPUT_DIR,
    speaker_db_path=DEFAULT_SPEAKER_DB,
    log_level='INFO'
)

# Helper accessors that follow the current runtime config
def get_audio_dir() -> Path:
    try:
        return Path(voice_forge.output_base_dir) / 'audio'
    except Exception:
        return Path(DEFAULT_OUTPUT_DIR) / 'audio'

def get_upload_dir() -> Path:
    try:
        return Path(voice_forge.output_base_dir) / 'uploads'
    except Exception:
        return Path(DEFAULT_OUTPUT_DIR) / 'uploads'

# Serve static UI and audio directory
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="webui")
    
    @app.get("/")
    def serve_index():
        index_file = WEB_DIR / 'index.html'
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h1>VoiceForge UI</h1><p>UI not found. Ensure webui/index.html exists.</p>")
 # Audio files are served via /api/audio/{filename} to honor dynamic output dirs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.get("/api/speakers")
def api_list_speakers():
    speakers = voice_forge.list_speakers()
    return {"speakers": speakers}


@app.get("/api/stats")
def api_stats():
    return voice_forge.get_stats()


@app.post("/api/delete_speaker")
def api_delete_speaker(speaker_id: str = Form(...)):
    ok = voice_forge.delete_speaker(speaker_id)
    return {"ok": ok}


@app.post("/api/extract")
async def api_extract(
    audio: UploadFile = File(...),
    transcript: str = Form(...),
    speaker_id: str = Form(...)
):
    # Save uploaded audio
    ensure_dir(get_upload_dir())
    dest_path = get_upload_dir() / audio.filename
    with open(dest_path, 'wb') as f:
        f.write(await audio.read())

    ok = voice_forge.extract_speaker_embedding(str(dest_path), transcript, speaker_id)
    return {"ok": ok, "speaker_id": speaker_id}


@app.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(...),
    output_name: str = Form(...),
    speaker_id: str = Form(""),
    instruction: str = Form(""),
    emotion: str = Form(""),
    tone: str = Form(""),
    speed: float = Form(1.0),
    language: str = Form(""),
    prompt_text: str = Form(""),
    prompt_audio: Optional[UploadFile] = File(None)
):
    prompt_audio_path = None
    if prompt_audio is not None:
        ensure_dir(get_upload_dir())
        prompt_audio_path = str(get_upload_dir() / prompt_audio.filename)
        with open(prompt_audio_path, 'wb') as f:
            f.write(await prompt_audio.read())

    result_path = voice_forge.synthesize_speech(
        text=text,
        output_filename=output_name,
        speaker_id=speaker_id if speaker_id else None,
        prompt_audio=prompt_audio_path,
        prompt_text=prompt_text,
        instruction=instruction if instruction else None,
        emotion=emotion if emotion else None,
        tone=tone if tone else None,
        speed=float(speed) if speed else 1.0,
        language=language if language else None,
    )
    return {"ok": result_path is not None, "output_path": result_path}


@app.post("/api/add_embedding")
async def api_add_embedding(
    speaker_id: str = Form(...),
    transcript: str = Form(""),
    embedding_file: UploadFile = File(...)
):
    # Save uploaded embedding file
    ensure_dir(get_upload_dir())
    dest_path = get_upload_dir() / embedding_file.filename
    with open(dest_path, 'wb') as f:
        f.write(await embedding_file.read())

    # Load embedding vector similar to main.py
    import numpy as np
    embedding_vector = None
    try:
        if dest_path.suffix.lower() == '.npy':
            arr = np.load(dest_path)
            embedding_vector = arr.flatten().tolist()
        elif dest_path.suffix.lower() == '.json':
            with open(dest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                embedding_vector = data
            elif isinstance(data, dict):
                # Accept keys 'embedding' or 'spk_emb'
                if 'embedding' in data:
                    embedding_vector = data['embedding']
                elif 'spk_emb' in data:
                    # Some JSON may store [[...]] shape
                    val = data['spk_emb']
                    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                        embedding_vector = val[0]
                    else:
                        embedding_vector = val
            else:
                return JSONResponse({"ok": False, "error": "Invalid JSON embedding format"}, status_code=400)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to load embedding: {e}"}, status_code=400)

    if not embedding_vector or len(embedding_vector) != 192:
        return JSONResponse({"ok": False, "error": f"Invalid embedding dimension: {len(embedding_vector) if embedding_vector else 'None'}"}, status_code=400)

    ok = voice_forge.add_speaker_from_embedding(
        speaker_id=speaker_id,
        embedding_vector=embedding_vector,
        prompt_text=transcript or f"Speaker {speaker_id}",
        embedding_metadata={'source': 'ui_upload'}
    )
    return {"ok": ok, "speaker_id": speaker_id}


@app.post("/api/dialogue")
async def api_dialogue(
    script_text: str = Form(...),
    dialogue_name: str = Form(...),
    default_speaker: str = Form("")
):
    # Save script to a temp file
    ensure_dir(get_upload_dir())
    script_path = get_upload_dir() / f"{dialogue_name}.txt"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_text)

    results = voice_forge.process_dialogue_script(
        script_path=str(script_path),
        dialogue_name=dialogue_name,
        default_speaker_id=default_speaker if default_speaker else None
    )
    ok = results is not None
    return {"ok": ok, "results": results}


@app.get("/api/audio_list")
def api_audio_list():
    ensure_dir(get_audio_dir())
    files = []
    for p in sorted(get_audio_dir().glob('*.wav')):
        files.append({
            "name": p.name,
            "path": f"/api/audio/{p.name}",
            "size": p.stat().st_size
        })
    return {"audio": files}


@app.get("/api/audio/{filename}")
def api_audio(filename: str):
    file_path = get_audio_dir() / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(file_path), media_type='audio/wav', filename=filename)


@app.post("/api/export")
def api_export():
    # Export to an in-memory JSON
    data = voice_forge.export_speaker_embeddings(None)  # implement method returns dict when None
    if isinstance(data, dict):
        return JSONResponse(data)
    # Fallback: read from default export path if implemented
    return JSONResponse({"error": "Export not available"}, status_code=400)


@app.post("/api/import")
async def api_import(file: UploadFile = File(...)):
    ensure_dir(get_upload_dir())
    dest_path = get_upload_dir() / file.filename
    with open(dest_path, 'wb') as f:
        f.write(await file.read())
    try:
        count = voice_forge.import_speaker_embeddings(str(dest_path))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "imported": count}


@app.post("/api/config")
def api_config(output_dir: Optional[str] = Form(None), speaker_db: Optional[str] = Form(None)):
    # Update output directory and/or speaker db at runtime
    if output_dir:
        new_out = Path(output_dir)
        ensure_dir(new_out)
        voice_forge.output_base_dir = str(new_out)
        # Update audio dir mount too (requires restart to remount). We'll just ensure path exists.
    if speaker_db:
        # Point to new DB and attempt reload
        voice_forge.speaker_db_path = Path(speaker_db)
        # Load existing speakers from the new DB if exists
        try:
            voice_forge._load_existing_speakers()
        except Exception:
            pass
    return {"ok": True, "output_dir": voice_forge.output_base_dir, "speaker_db": str(voice_forge.speaker_db_path)}


if __name__ == "__main__":
    # Run server and serve the UI
    port = int(os.environ.get('VOICEFORGE_UI_PORT', '8008'))
    print(f"Starting VoiceForge UI at http://127.0.0.1:{port}")
    uvicorn.run("ui_server:app", host="0.0.0.0", port=port, reload=False)