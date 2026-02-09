"""
Qwen TTS Desktop Interface
A simple GUI for Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import torch
import soundfile as sf
import os
import io
import base64

# Audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# For playing audio without external files
try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class QwenTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen TTS - Text to Speech")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)

        # Model instance
        self.model = None
        self.current_audio = None
        self.current_sr = None

        # Available speakers
        self.speakers = [
            ("Vivian", "Bright, slightly edgy young female voice (Chinese)"),
            ("Serena", "Warm, gentle young female voice (Chinese)"),
            ("Uncle_Fu", "Seasoned male voice with low, mellow timbre (Chinese)"),
            ("Dylan", "Youthful Beijing male voice (Chinese - Beijing Dialect)"),
            ("Eric", "Lively Chengdu male voice (Chinese - Sichuan Dialect)"),
            ("Ryan", "Dynamic male voice with strong rhythmic drive (English)"),
            ("Aiden", "Sunny American male voice with clear midrange (English)"),
            ("Ono_Anna", "Playful Japanese female voice (Japanese)"),
            ("Sohee", "Warm Korean female voice with rich emotion (Korean)"),
        ]

        # Language options
        self.languages = ["Auto", "Chinese", "English", "Japanese", "Korean",
                         "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

        self.setup_ui()
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required packages are installed"""
        missing = []
        try:
            import qwen_tts
        except ImportError:
            missing.append("qwen-tts")
        try:
            import torch
        except ImportError:
            missing.append("torch")
        try:
            import soundfile
        except ImportError:
            missing.append("soundfile")

        if missing:
            messagebox.showwarning(
                "Missing Dependencies",
                f"Please install missing packages:\n\n"
                f"pip install {' '.join(missing)}\n\n"
                f"For the complete setup:\n"
                f"pip install qwen-tts torch soundfile pygame"
            )

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Qwen TTS",
            font=("Helvetica", 20, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky="w")

        subtitle_label = ttk.Label(
            main_frame,
            text="Text-to-Speech with Style Control",
            font=("Helvetica", 10),
            foreground="gray"
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky="w")

        # Model Loading Section
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="10")
        model_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        model_frame.columnconfigure(0, weight=1)

        self.load_model_btn = ttk.Button(
            model_frame,
            text="Load Model",
            command=self.load_model
        )
        self.load_model_btn.grid(row=0, column=0, sticky="w")

        self.model_status = ttk.Label(model_frame, text="Not loaded", foreground="red")
        self.model_status.grid(row=0, column=1, padx=(10, 0))

        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        settings_frame.columnconfigure(1, weight=1)

        # Speaker selection
        ttk.Label(settings_frame, text="Speaker:").grid(row=0, column=0, sticky="w", pady=5)
        self.speaker_var = tk.StringVar(value="Vivian")
        speaker_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.speaker_var,
            values=[s[0] for s in self.speakers],
            state="readonly",
            width=20
        )
        speaker_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        speaker_combo.bind("<<ComboboxSelected>>", self.update_speaker_info)

        # Speaker description
        self.speaker_info = ttk.Label(
            settings_frame,
            text=self.speakers[0][1],
            font=("Helvetica", 9),
            foreground="gray",
            wraplength=400
        )
        self.speaker_info.grid(row=1, column=1, sticky="w", padx=(10, 0))

        # Language selection
        ttk.Label(settings_frame, text="Language:").grid(row=2, column=0, sticky="w", pady=5)
        self.language_var = tk.StringVar(value="Auto")
        language_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.language_var,
            values=self.languages,
            state="readonly",
            width=20
        )
        language_combo.grid(row=2, column=1, sticky="w", padx=(10, 0))

        # Instruction/Style control
        ttk.Label(settings_frame, text="Instruction:").grid(row=3, column=0, sticky="nw", pady=5)
        self.instruct_var = tk.StringVar()
        instruct_entry = ttk.Entry(settings_frame, textvariable=self.instruct_var)
        instruct_entry.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=5)

        instruct_hint = ttk.Label(
            settings_frame,
            text="Optional: e.g., 'speak happily', 'whisper', 'angry tone'",
            font=("Helvetica", 8),
            foreground="gray"
        )
        instruct_hint.grid(row=4, column=1, sticky="w", padx=(10, 0))

        # Text Input Section
        text_frame = ttk.LabelFrame(main_frame, text="Text to Synthesize", padding="10")
        text_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 15))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.text_input = tk.Text(text_frame, height=6, wrap="word", font=("Helvetica", 11))
        self.text_input.grid(row=0, column=0, sticky="nsew")

        # Scrollbar for text
        text_scroll = ttk.Scrollbar(text_frame, command=self.text_input.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.text_input.config(yscrollcommand=text_scroll.set)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(0, 15))

        self.generate_btn = ttk.Button(
            btn_frame,
            text="Generate Speech",
            command=self.generate_speech,
            width=20
        )
        self.generate_btn.pack(side="left", padx=5)

        self.play_btn = ttk.Button(
            btn_frame,
            text="Play",
            command=self.play_audio,
            width=15,
            state="disabled"
        )
        self.play_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(
            btn_frame,
            text="Save Audio",
            command=self.save_audio,
            width=15,
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status_label.grid(row=7, column=0, columnspan=2, sticky="w")

        # Sample texts button
        sample_btn = ttk.Button(
            main_frame,
            text="Load Sample Text",
            command=self.load_sample
        )
        sample_btn.grid(row=7, column=1, sticky="e")

    def update_speaker_info(self, event=None):
        """Update speaker description when selection changes"""
        speaker = self.speaker_var.get()
        for name, desc in self.speakers:
            if name == speaker:
                self.speaker_info.config(text=desc)
                break

    def load_model(self):
        """Load the Qwen TTS model in a separate thread"""
        def load():
            try:
                from qwen_tts import Qwen3TTSModel

                self.status_label.config(text="Loading model... This may take a minute.")
                self.load_model_btn.config(state="disabled")
                self.progress.start()

                # Determine device and dtype
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                # Try flash attention, fall back to eager
                try:
                    self.model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        device_map=device,
                        dtype=dtype,
                        attn_implementation="flash_attention_2",
                    )
                except Exception:
                    self.model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        device_map=device,
                        dtype=dtype,
                        attn_implementation="eager",
                    )

                self.root.after(0, self.on_model_loaded, device)
            except Exception as e:
                self.root.after(0, self.on_model_error, str(e))

        threading.Thread(target=load, daemon=True).start()

    def on_model_loaded(self, device):
        """Callback when model is loaded"""
        self.progress.stop()
        self.load_model_btn.config(state="normal")
        self.model_status.config(text=f"Loaded on {device}", foreground="green")
        self.status_label.config(text="Model ready!")

    def on_model_error(self, error):
        """Callback when model fails to load"""
        self.progress.stop()
        self.load_model_btn.config(state="normal")
        self.model_status.config(text="Failed to load", foreground="red")
        self.status_label.config(text="Error loading model")
        messagebox.showerror("Error", f"Failed to load model:\n{error}")

    def generate_speech(self):
        """Generate speech from text"""
        if self.model is None:
            messagebox.showwarning("Model Not Loaded", "Please load the model first!")
            return

        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter some text to synthesize!")
            return

        def generate():
            try:
                self.root.after(0, self.on_generation_start)

                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=self.language_var.get(),
                    speaker=self.speaker_var.get(),
                    instruct=self.instruct_var.get().strip(),
                )

                self.current_audio = wavs[0]
                self.current_sr = sr

                self.root.after(0, self.on_generation_complete)
            except Exception as e:
                self.root.after(0, self.on_generation_error, str(e))

        threading.Thread(target=generate, daemon=True).start()

    def on_generation_start(self):
        """Callback when generation starts"""
        self.generate_btn.config(state="disabled")
        self.status_label.config(text="Generating speech...")
        self.progress.start()

    def on_generation_complete(self):
        """Callback when generation completes"""
        self.progress.stop()
        self.generate_btn.config(state="normal")
        self.play_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.status_label.config(text="Speech generated successfully!")

    def on_generation_error(self, error):
        """Callback when generation fails"""
        self.progress.stop()
        self.generate_btn.config(state="normal")
        self.status_label.config(text="Generation failed")
        messagebox.showerror("Error", f"Failed to generate speech:\n{error}")

    def play_audio(self):
        """Play the generated audio"""
        if self.current_audio is None:
            return

        try:
            # Save to temporary file and play
            temp_file = "_temp_qwen_tts.wav"
            sf.write(temp_file, self.current_audio, self.current_sr)

            if PYGAME_AVAILABLE:
                pygame.mixer.init()
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
            elif PYDUB_AVAILABLE:
                audio = AudioSegment.from_wav(temp_file)
                play(audio)
            else:
                # Fallback: use system default player
                import platform
                import subprocess

                system = platform.system()
                if system == "Windows":
                    os.startfile(temp_file)
                elif system == "Darwin":  # macOS
                    subprocess.call(["afplay", temp_file])
                else:  # Linux
                    subprocess.call(["aplay", temp_file])

            self.status_label.config(text="Playing audio...")

            # Clean up temp file after a delay
            def cleanup():
                import time
                time.sleep(2)
                try:
                    os.remove(temp_file)
                except:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio:\n{e}")

    def save_audio(self):
        """Save the generated audio to a file"""
        if self.current_audio is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            try:
                sf.write(file_path, self.current_audio, self.current_sr)
                self.status_label.config(text=f"Saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio:\n{e}")

    def load_sample(self):
        """Load sample text based on selected speaker"""
        speaker = self.speaker_var.get()

        samples = {
            "Vivian": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
            "Serena": "在这个世界上，最美好的事情就是和家人一起度过温馨的时光。",
            "Uncle_Fu": "年轻人啊，做事要稳扎稳打，不能急于求成。",
            "Dylan": "这事儿吧，我觉得得这么办，您说是不是这个理儿？",
            "Eric": "哎呀，这个火锅巴适得很，咱们再去整一顿嘛！",
            "Ryan": "She said she would be here by noon, but I haven't seen her yet.",
            "Aiden": "The sun is shining bright today, perfect for a picnic in the park!",
            "Ono_Anna": "おはようございます！今日も一日頑張りましょうね！",
            "Sohee": "안녕하세요! 오늘 날씨가 정말 좋네요.",
        }

        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", samples.get(speaker, "Enter your text here..."))


def main():
    root = tk.Tk()
    app = QwenTTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
