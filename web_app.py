"""
Qwen TTS Web Interface with Voice Cloning
A Flask-based web UI for Qwen3-TTS supporting preset voices and voice cloning
Only one model is kept in memory at a time to save VRAM.
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import soundfile as sf
import io
import os
import base64
import tempfile
import gc
from datetime import datetime

try:
    from qwen_tts import Qwen3TTSModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: qwen-tts not installed. Run: pip install qwen-tts")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

# Global model instance (only one loaded at a time)
current_model = None
current_model_type = None  # 'custom' or 'clone'

# Model status
model_status = {"loaded": False, "device": None, "error": None, "type": None}

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


def get_gpu_info():
    """Get GPU information - supports both NVIDIA (CUDA) and AMD (ROCm)"""
    if torch.cuda.is_available():
        # Check if it's ROCm (AMD) or CUDA (NVIDIA)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return {
                'available': True,
                'type': 'AMD ROCm',
                'device': 'hip',
                'name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'AMD GPU'
            }
        else:
            return {
                'available': True,
                'type': 'NVIDIA CUDA',
                'device': 'cuda',
                'name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NVIDIA GPU'
            }
    return {'available': False, 'type': 'CPU', 'device': 'cpu', 'name': 'CPU'}


def unload_current_model():
    """Unload the current model to free up VRAM"""
    global current_model, current_model_type

    if current_model is not None:
        print(f"Unloading {current_model_type} model to free VRAM...")
        del current_model
        current_model = None
        current_model_type = None

        # Force garbage collection and clear GPU cache
        gc.collect()
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Model unloaded successfully.")


def load_model(model_type):
    """Load the specified TTS model, unloading any existing model first"""
    global current_model, current_model_type, model_status

    if not MODEL_AVAILABLE:
        model_status["error"] = "qwen-tts package not installed"
        return False

    # If requested model is already loaded, just return success
    if current_model_type == model_type and current_model is not None:
        model_status["loaded"] = True
        model_status["type"] = model_type
        return True

    # Unload any existing model first
    unload_current_model()

    try:
        # Get GPU info (NVIDIA or AMD)
        gpu_info = get_gpu_info()
        device_name = gpu_info['device']
        device_id = f"{device_name}:0" if gpu_info['available'] else "cpu"
        dtype = torch.bfloat16 if gpu_info['available'] else torch.float32

        print(f"GPU Info: {gpu_info}")
        if gpu_info['available']:
            print(f"GPU: {gpu_info['name']}")
            print(f"GPU memory before load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        if model_type == "custom":
            # Load CustomVoice model for preset speakers
            print(f"Loading CustomVoice model...")
            try:
                current_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    torch_dtype=dtype,
                    attn_implementation="flash_attention_2",
                )
            except Exception as e:
                print(f"Flash attention failed ({e}), trying eager...")
                current_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    torch_dtype=dtype,
                    attn_implementation="eager",
                )

            # Explicitly move to GPU if available
            if gpu_info['available']:
                current_model = current_model.to(device_id)
                print(f"Model moved to {device_id}")

            current_model_type = "custom"
            print("CustomVoice model loaded successfully!")
        else:
            # Load Base model for voice cloning
            print(f"Loading Base (cloning) model...")
            try:
                current_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    torch_dtype=dtype,
                    attn_implementation="flash_attention_2",
                )
            except Exception as e:
                print(f"Flash attention failed ({e}), trying eager...")
                current_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    torch_dtype=dtype,
                    attn_implementation="eager",
                )

            # Explicitly move to GPU if available
            if gpu_info['available']:
                current_model = current_model.to(device_id)
                print(f"Model moved to {device_id}")

            current_model_type = "clone"
            print("Base (cloning) model loaded successfully!")

        # Verify model is on GPU
        if gpu_info['available']:
            print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            # Check model device
            model_device = next(current_model.parameters()).device
            print(f"Model is on device: {model_device}")
            device_id = str(model_device)

        model_status["loaded"] = True
        model_status["device"] = device_id
        model_status["error"] = None
        model_status["type"] = model_type
        return True
    except Exception as e:
        model_status["error"] = str(e)
        model_status["loaded"] = False
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
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
    """Get current model status"""
    return jsonify(model_status)


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """API endpoint to load a model (unloads previous if any)"""
    data = request.json or {}
    model_type = data.get("type", "custom")  # "custom" or "clone"

    success = load_model(model_type)
    return jsonify({"success": success, **model_status})


@app.route("/api/unload_model", methods=["POST"])
def api_unload_model():
    """API endpoint to unload the current model"""
    unload_current_model()
    model_status["loaded"] = False
    model_status["device"] = None
    model_status["type"] = None
    return jsonify({"success": True, **model_status})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate speech from text using preset voices"""
    if not model_status["loaded"] or current_model is None:
        return jsonify({"error": "No model loaded. Please load the CustomVoice model first."}), 400

    if current_model_type != "custom":
        return jsonify({"error": "Wrong model loaded. Please switch to Preset Voices mode and load the model."}), 400

    data = request.json
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "Auto")
    instruct = data.get("instruct", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        wavs, sr = current_model.generate_custom_voice(
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
    if not model_status["loaded"] or current_model is None:
        return jsonify({"error": "No model loaded. Please load the Base model first."}), 400

    if current_model_type != "clone":
        return jsonify({"error": "Wrong model loaded. Please switch to Voice Cloning mode and load the model."}), 400

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
                    voice_clone_prompt = current_model.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        x_vector_only_mode=False,
                    )
                    # Cache the prompt
                    cache_key = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    voice_clone_cache[cache_key] = voice_clone_prompt
                else:
                    # Use x_vector_only mode if no transcript provided
                    voice_clone_prompt = current_model.create_voice_clone_prompt(
                        ref_audio=ref_audio_path,
                        x_vector_only_mode=True,
                    )
            finally:
                os.unlink(ref_audio_path)
        else:
            return jsonify({"error": "No reference audio provided"}), 400

        # Generate cloned voice
        wavs, sr = current_model.generate_voice_clone(
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

    if not model_status["loaded"] or current_model is None or current_model_type != "custom":
        return jsonify({"error": "CustomVoice model not loaded"}), 400

    try:
        wavs, sr = current_model.generate_custom_voice(
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

    # Show GPU info at startup
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} ({gpu_info['type']})")
    print(f"GPU Available: {gpu_info['available']}")

    print("")
    print("Features:")
    print("  - Preset Voices: 9 built-in speakers")
    print("  - Voice Cloning: Clone any voice")
    print("")
    print("Note: Only one model is kept in memory at a time to save VRAM.")
    print("      Switching modes will automatically unload the previous model.")
    print("")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
