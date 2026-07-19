import os
import sys
import uuid
import tempfile

# Fix Windows console encoding so emoji/unicode in LLM responses don't crash the app
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import imageio_ffmpeg
from elevenlabs.client import ElevenLabs
from gtts import gTTS

# Point pydub to the bundled ffmpeg binary (no system install needed)
_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = _ffmpeg_path
AudioSegment.ffmpeg = _ffmpeg_path
AudioSegment.ffprobe = _ffmpeg_path.replace("ffmpeg", "ffprobe")

from config import Config
from agents.agent_decision import process_query
from database import get_db, init_db
from auth import get_current_user
from models import User, Conversation, Message
from routers.auth import router as auth_router
from routers.conversations import router as conversations_router

# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# CORS — required for React dev server (localhost:5173) to send cookies to API (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(conversations_router)

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
BRAIN_TUMOR_OUTPUT = "uploads/brain_tumor_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, BRAIN_TUMOR_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files for uploads/data (specific prefixes — safe to register early)
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates (kept for fallback during development)
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.on_event("startup")
def startup():
    init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_audio():
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)


cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()


class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"


def _save_messages(db: Session, conversation_id: int, user_text: str, response_data: dict, result_image: Optional[str] = None):
    """Persist user + assistant messages and auto-title the conversation."""
    response_text = response_data['messages'][-1].content
    agent_name = response_data.get("agent_name", "")

    # Save user message
    db.add(Message(
        conversation_id=conversation_id,
        role="user",
        content=user_text,
    ))

    # Save assistant message
    db.add(Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        agent_name=agent_name,
        result_image=result_image,
    ))

    # Auto-set title from first user message
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv and conv.title == "New Conversation":
        conv.title = user_text[:60].strip() or "New Conversation"
    if conv:
        conv.updated_at = datetime.utcnow()

    db.commit()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat")
def chat(
    request: QueryRequest,
    conversation_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Process user text query through the multi-agent system."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        response_data = process_query(
            request.query,
            user_id=current_user.id,
            conversation_id=conversation_id,
        )
        response_text = response_data['messages'][-1].content
        agent_name = response_data.get("agent_name") or "CONVERSATION_AGENT"
        print(f"[/chat] agent_name='{agent_name}'")

        result = {
            "status": "success",
            "response": response_text,
            "agent": agent_name,
        }

        if agent_name == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            seg_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"

        _save_messages(db, conversation_id, request.query, response_data, result.get("result_image"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    text: str = Form(""),
    conversation_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Process medical image uploads with optional text input."""
    if not allowed_file(image.filename):
        return JSONResponse(status_code=400, content={
            "status": "error", "agent": "System",
            "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG",
        })

    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:
        return JSONResponse(status_code=413, content={
            "status": "error", "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
        })

    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(
            query,
            user_id=current_user.id,
            conversation_id=conversation_id,
        )
        response_text = response_data['messages'][-1].content
        agent_name = response_data.get("agent_name") or "SKIN_LESION_AGENT"
        print(f"[/upload] agent_name='{agent_name}'")

        result = {
            "status": "success",
            "response": response_text,
            "agent": agent_name,
        }

        if agent_name == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            seg_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"

        if agent_name == "BRAIN_TUMOR_AGENT, HUMAN_VALIDATION":
            seg_path = os.path.join(BRAIN_TUMOR_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/brain_tumor_output/segmentation_plot.png"

        user_text = text or "(image uploaded)"
        _save_messages(db, conversation_id, user_text, response_data, result.get("result_image"))

        try:
            os.remove(file_path)
        except Exception:
            pass

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
def validate_medical_output(
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    conversation_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Handle human validation for medical AI outputs."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"

        response_data = process_query(
            validation_query,
            user_id=current_user.id,
            conversation_id=conversation_id,
        )
        response_text = response_data['messages'][-1].content

        _save_messages(db, conversation_id, validation_query, response_data)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_text,
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_text,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Transcribe speech using ElevenLabs API."""
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No audio file selected"})

    try:
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"

        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)

        if os.path.getsize(temp_audio) == 0:
            return JSONResponse(status_code=400, content={"error": "Received empty audio file"})

        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        try:
            audio_seg = AudioSegment.from_file(temp_audio)
            audio_seg.export(mp3_path, format="mp3")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )

            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
            except Exception:
                pass

            if transcription.text:
                return {"transcript": transcription.text}
            return JSONResponse(status_code=500, content={"error": "Transcription returned empty text"})

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Error processing audio: {str(e)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-speech")
async def generate_speech(
    request: SpeechRequest,
    current_user: User = Depends(get_current_user),
):
    """Generate speech using gTTS."""
    text = request.text
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is required"})

    os.makedirs(SPEECH_DIR, exist_ok=True)
    temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(temp_audio_path)

    return FileResponse(path=temp_audio_path, media_type="audio/mpeg", filename="generated_speech.mp3")


@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(status_code=413, content={
        "status": "error", "agent": "System",
        "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
    })


# Serve built React app in production — must be registered LAST so API routes take priority
_frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
