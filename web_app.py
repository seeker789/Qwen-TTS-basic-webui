"""
Qwen TTS Web Interface
A Flask-based web UI for Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import soundfile as sf
import io
import os
import base64
from datetime import datetime

try:
    from qwen_tts import Qwen3TTSModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: qwen-tts not installed. Run: pip install qwen-tts")

app = Flask(__name__)

# Global model instance
model = None
model_status = {"loaded": False, "device": None, "error": None}

# Configuration
SPEAKERS = [
    {"name": "Vivian", "desc": "Bright, slightly edgy young female voice", "lang": "Chinese"},
    {"name": "Serena", "desc": "Warm, gentle young female voice", "lang": "Chinese"},
    {"name": "Uncle_Fu", "desc": "Seasoned male voice with low, mellow timbre", "lang": "Chinese"},
    {"name": "Dylan", "desc": "Youthful Beijing male voice", "lang": "Chinese (Beijing)"},
    {"name": "Eric", "desc": "Lively Chengdu male voice", "lang": "Chinese (Sichuan)"},
    {"name": "Ryan", "desc": "Dynamic male voice with strong rhythmic drive", "lang": "English"},
    {"name": "Aiden", "desc": "Sunny American male voice with clear midrange", "lang": "English"},
    {"name": "Ono_Anna", "desc": "Playful Japanese female voice", "lang": "Japanese"},
    {"name": "Sohee", "desc": "Warm Korean female voice with rich emotion", "lang": "Korean"},
]

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean",
             "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


def load_model():
    """Load the TTS model"""
    global model, model_status

    if not MODEL_AVAILABLE:
        model_status["error"] = "qwen-tts package not installed"
        return False

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Try flash attention, fallback to eager
        try:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )

        model_status["loaded"] = True
        model_status["device"] = device
        return True
    except Exception as e:
        model_status["error"] = str(e)
        return False


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html",
                         speakers=SPEAKERS,
                         languages=LANGUAGES,
                         model_status=model_status)


@app.route("/api/status")
def status():
    """Get model status"""
    return jsonify(model_status)


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """API endpoint to load the model"""
    success = load_model()
    return jsonify({"success": success, **model_status})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate speech from text"""
    if not model_status["loaded"]:
        return jsonify({"error": "Model not loaded"}), 400

    data = request.json
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        # Convert to base64 for web playback
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format="WAV")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "audio": audio_base64,
            "sample_rate": sr,
            "duration": len(wavs[0]) / sr
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download", methods=["POST"])
def download():
    """Download generated audio as file"""
    if not model_status["loaded"]:
        return jsonify({"error": "Model not loaded"}), 400

    data = request.json
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        # Save to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_tts_{speaker}_{timestamp}.wav"

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format="WAV")
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("Qwen TTS Web Interface")
    print("=" * 50)
    print(f"Model available: {MODEL_AVAILABLE}")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=False)
