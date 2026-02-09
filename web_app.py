"""
Qwen TTS Web Interface with Voice Cloning
A Flask-based web UI for Qwen3-TTS supporting preset voices and voice cloning
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import soundfile as sf
import io
import os
import base64
import tempfile
from datetime import datetime

try:
    from qwen_tts import Qwen3TTSModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: qwen-tts not installed. Run: pip install qwen-tts")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

# Global model instances
model_custom = None  # For preset voices
model_clone = None   # For voice cloning

# Separate status for each model
custom_model_status = {"loaded": False, "device": None, "error": None}
clone_model_status = {"loaded": False, "device": None, "error": None}

# Cached voice clone prompts
voice_clone_cache = {}

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


def load_custom_model():
    """Load the CustomVoice model for preset speakers"""
    global model_custom, custom_model_status

    if not MODEL_AVAILABLE:
        custom_model_status["error"] = "qwen-tts package not installed"
        return False

    if model_custom is not None:
        custom_model_status["loaded"] = True
        return True

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading CustomVoice model on {device}...")
        try:
            model_custom = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model_custom = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )

        custom_model_status["loaded"] = True
        custom_model_status["device"] = device
        custom_model_status["error"] = None
        print("CustomVoice model loaded successfully!")
        return True
    except Exception as e:
        custom_model_status["error"] = str(e)
        print(f"Failed to load CustomVoice model: {e}")
        return False


def load_clone_model():
    """Load the Base model for voice cloning"""
    global model_clone, clone_model_status

    if not MODEL_AVAILABLE:
        clone_model_status["error"] = "qwen-tts package not installed"
        return False

    if model_clone is not None:
        clone_model_status["loaded"] = True
        return True

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading Base (cloning) model on {device}...")
        try:
            model_clone = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model_clone = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )

        clone_model_status["loaded"] = True
        clone_model_status["device"] = device
        clone_model_status["error"] = None
        print("Base (cloning) model loaded successfully!")
        return True
    except Exception as e:
        clone_model_status["error"] = str(e)
        print(f"Failed to load Base model: {e}")
        return False


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html",
                         speakers=SPEAKERS,
                         languages=LANGUAGES,
                         custom_status=custom_model_status,
                         clone_status=clone_model_status)


@app.route("/api/status")
def status():
    """Get both model statuses"""
    return jsonify({
        "custom": custom_model_status,
        "clone": clone_model_status
    })


@app.route("/api/load_custom_model", methods=["POST"])
def api_load_custom_model():
    """API endpoint to load the CustomVoice model"""
    success = load_custom_model()
    return jsonify({"success": success, **custom_model_status})


@app.route("/api/load_clone_model", methods=["POST"])
def api_load_clone_model():
    """API endpoint to load the Base (cloning) model"""
    success = load_clone_model()
    return jsonify({"success": success, **clone_model_status})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate speech from text using preset voices"""
    if not custom_model_status["loaded"] or not model_custom:
        return jsonify({"error": "CustomVoice model not loaded. Please load the model in the Preset Voices section."}), 400

    data = request.json
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        wavs, sr = model_custom.generate_custom_voice(
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


@app.route("/api/clone_voice", methods=["POST"])
def clone_voice():
    """Generate speech using voice cloning"""
    if not clone_model_status["loaded"] or not model_clone:
        return jsonify({"error": "Base model not loaded. Please load the model in the Voice Cloning section."}), 400

    text = request.form.get("text", "").strip()
    language = request.form.get("language", "English")
    ref_text = request.form.get("ref_text", "").strip()
    use_cache = request.form.get("use_cache", "false").lower() == "true"
    cache_key = request.form.get("cache_key", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        voice_clone_prompt = None

        # Use cached prompt if available
        if use_cache and cache_key and cache_key in voice_clone_cache:
            voice_clone_prompt = voice_clone_cache[cache_key]
        elif "ref_audio" in request.files:
            # Process new reference audio
            ref_file = request.files["ref_audio"]
            if ref_file.filename == "":
                return jsonify({"error": "No reference audio file"}), 400

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                ref_file.save(tmp.name)
                ref_audio_path = tmp.name

            try:
                # Create voice clone prompt for reuse
                if ref_text:
                    voice_clone_prompt = model_clone.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        x_vector_only_mode=False,
                    )
                    # Cache the prompt
                    cache_key = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    voice_clone_cache[cache_key] = voice_clone_prompt
                else:
                    # Use x_vector_only mode if no transcript provided
                    voice_clone_prompt = model_clone.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        x_vector_only_mode=True,
                    )
            finally:
                os.unlink(ref_audio_path)
        else:
            return jsonify({"error": "No reference audio provided"}), 400

        # Generate cloned voice
        wavs, sr = model_clone.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
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
            "duration": len(wavs[0]) / sr,
            "cache_key": cache_key if cache_key else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download", methods=["POST"])
def download():
    """Download generated audio as file"""
    data = request.json
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not model_custom:
        return jsonify({"error": "CustomVoice model not loaded"}), 400

    try:
        wavs, sr = model_custom.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

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
    print("=" * 60)
    print("Qwen TTS Web Interface with Voice Cloning")
    print("=" * 60)
    print(f"Model package available: {MODEL_AVAILABLE}")
    print("")
    print("Features:")
    print("  - Preset Voices: 9 built-in speakers (load CustomVoice model)")
    print("  - Voice Cloning: Clone any voice (load Base model)")
    print("")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
