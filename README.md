# Qwen-TTS Basic WebUI

A clean, user-friendly web interface for the [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) text-to-speech model with **voice cloning** support.

![Qwen TTS WebUI Screenshot](qwenttsscreenshot.jpg)

## Features

- **Modern Web Interface** - Clean, responsive design with gradient styling
- **9 Premium Preset Voices** - Choose from multiple speakers with different languages, genders, and styles
- **Voice Cloning** - Clone any voice from just 5-15 seconds of audio
- **Style Control** - Use natural language instructions to control tone, emotion, and prosody
- **Multi-language Support** - 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **In-Browser Playback** - Listen to generated speech directly in the browser
- **WAV Export** - Download generated audio as WAV files
- **Voice Caching** - Reuse cloned voices without re-uploading reference audio

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM) or CPU

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Qwen-TTS-basic-webui.git
cd Qwen-TTS-basic-webui
```

2. Create a virtual environment:
```bash
conda create -n qwen-tts python=3.12 -y
conda activate qwen-tts
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Optional: Install FlashAttention 2 for reduced GPU memory usage:
```bash
pip install flash-attn --no-build-isolation
```

### Usage

1. Start the web server:
```bash
python web_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. **Choose Mode:**
   - **Preset Voices** - Use one of the 9 built-in speakers
   - **Voice Cloning** - Clone any voice from audio

4. **For Preset Voices:**
   - Click **"Load Model"** to download the CustomVoice model (~3.5GB on first run)
   - Select a **Speaker** from the dropdown
   - (Optional) Enter an **Instruction** to control style/tone
   - Enter your **text** and click **"Generate Speech"**

5. **For Voice Cloning:**
   - Click **"Load Model"** to download the Base model (~3.5GB on first run)
   - Upload a **Reference Audio** file (5-15 seconds of clear speech)
   - Enter the **Reference Transcript** (what's being said in the audio)
   - Enter your **text** and click **"Clone Voice"**
   - The cloned voice will be cached for reuse

## Voice Cloning

### Requirements for Reference Audio

| Parameter | Recommendation |
|-----------|----------------|
| **Duration** | 5-15 seconds (minimum 3 seconds) |
| **Format** | WAV, MP3, M4A, OGG |
| **Quality** | Clear speech, no background noise/music |
| **Content** | Single speaker, continuous speech |
| **Transcript** | Exact text spoken (improves quality significantly) |

### Voice Cloning Tips

- Use high-quality, noise-free recordings for best results
- The transcript should match the audio exactly
- Once cloned, the voice is cached and can be reused without re-uploading
- You can clone voices in any of the 10 supported languages

## Available Preset Speakers

| Speaker | Description | Language |
|---------|-------------|----------|
| Vivian | Bright, slightly edgy young female voice | Chinese |
| Serena | Warm, gentle young female voice | Chinese |
| Uncle_Fu | Seasoned male voice with low, mellow timbre | Chinese |
| Dylan | Youthful Beijing male voice | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive | English |
| Aiden | Sunny American male voice with clear midrange | English |
| Ono_Anna | Playful Japanese female voice | Japanese |
| Sohee | Warm Korean female voice with rich emotion | Korean |

## Style Control Examples (Preset Voices)

Use natural language instructions to control the voice:

- **Emotion**: "speak happily", "sound sad", "angry tone", "excited"
- **Style**: "whisper", "shout", "sing", "narrate"
- **Speed**: "speak slowly", "speak quickly"
- **Combined**: "whisper sadly", "speak excitedly and quickly", "calm and gentle"

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | CPU only | NVIDIA GPU with 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Storage | 8GB free | 15GB free |
| Network | - | Broadband (for model download) |

**Note:** Both models (CustomVoice and Base) are ~3.5GB each. If you want to use both preset voices and voice cloning, you'll need ~7GB for model files.

## Model Cache Location

Models are downloaded and cached by HuggingFace:

- **Windows**: `C:\Users\<username>\.cache\huggingface\hub`
- **Linux/macOS**: `~/.cache/huggingface/hub`

To change the cache location, set the `HF_HOME` environment variable before running.

## Alternative: Desktop GUI

This repository also includes a desktop GUI version built with tkinter (preset voices only):

```bash
python qwen_tts_gui.py
```

## Troubleshooting

### Model fails to load
- Ensure you have enough disk space (~3.5GB per model)
- Check GPU memory availability
- The model will automatically fall back to CPU if CUDA is unavailable

### Voice cloning quality is poor
- Use clearer audio with less background noise
- Ensure the reference transcript matches the audio exactly
- Try using longer audio samples (10-15 seconds)

### Slow generation
- Use a GPU for significantly faster inference
- Close other memory-intensive applications
- Consider installing FlashAttention 2

### Web interface not accessible
- Check that port 5000 is not in use by another application
- To allow external access, modify `web_app.py` and change `host="0.0.0.0"`

## Project Structure

```
Qwen-TTS-basic-webui/
├── web_app.py              # Flask web application (preset + cloning)
├── qwen_tts_gui.py         # Desktop GUI (preset voices only)
├── templates/
│   └── index.html          # Web interface template
├── requirements.txt        # Python dependencies
├── qwenttsscreenshot.jpg   # Screenshot of the web UI
└── README.md              # This file
```

## Dependencies

- [qwen-tts](https://pypi.org/project/qwen-tts/) - Qwen TTS model package
- [torch](https://pytorch.org/) - PyTorch deep learning framework
- [soundfile](https://pysoundfile.readthedocs.io/) - Audio file I/O
- [flask](https://flask.palletsprojects.com/) - Web framework
- [pygame](https://www.pygame.org/) - Audio playback (desktop GUI)

## Acknowledgments

This project uses the Qwen3-TTS models from Alibaba Cloud's Qwen series:
- [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) - Preset voices
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) - Voice cloning

## Sources

- [Qwen3-TTS Voice Cloning API Reference](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning)
- [Qwen3-TTS GitHub Repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS: 3-Second Voice Cloning Beats ElevenLabs](https://byteiota.com/qwen3-tts-3-second-voice-cloning-beats-elevenlabs/)

## License

This interface code is provided as-is. Please refer to the [Qwen3-TTS model license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) for terms regarding the use of the underlying TTS model.
