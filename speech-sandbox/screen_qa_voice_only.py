import os, io, sys, base64, argparse, threading, queue, tempfile, time
import mss, numpy as np, cv2
from PIL import Image
import sounddevice as sd, soundfile as sf
import webrtcvad

# ---- STT: faster-whisper (recommended for mac) ----
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# ---- TTS ----
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---- Gemini ----
try:
    import google.generativeai as genai
except ImportError:
    print("Missing dependency: google-generativeai. Install with `pip install google-generativeai`")
    sys.exit(1)


# ---------------- Utility: beep ----------------
def beep():
    try:
        import winsound
        winsound.Beep(800, 120)  # Windows
    except Exception:
        sys.stdout.write("\a"); sys.stdout.flush()  # mac/Linux terminal bell


# ---------------- Vision / Gemini ----------------
def image_to_jpeg_bytes(pil_img, width_limit=1280, quality=80) -> bytes:
    w, h = pil_img.size
    if w > width_limit:
        r = width_limit / float(w)
        pil_img = pil_img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()

def ask_gemini(image_bytes, question, model_name="gemini-2.5-flash") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set your GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    alias = {
        "gemini-1.5-flash": "gemini-1.5-flash-002",
        "gemini-1.5-pro":   "gemini-1.5-pro-002",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }
    model_id = alias.get(model_name, model_name)
    model = genai.GenerativeModel(model_id)

    parts = [
        {"text": (
            "You are a screen-reading assistant.\n"
            "- Explain in clear, numbered steps what to click or type next.\n"
            "- Use on-screen labels verbatim when referencing UI elements.\n"
            "- Keep answers concise (<= 8 bullets)."
        )},
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(image_bytes).decode()}},
        {"text": f"User question: {question}"}
    ]
    try:
        resp = model.generate_content(contents=[{"role": "user", "parts": parts}])
        return resp.text
    except Exception as e:
        return f"[Gemini error] {e}"

def speak(text):
    if pyttsx3 is None:
        print("(TTS unavailable; `pip install pyttsx3`)")
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# --------------- Device & monitor pickers ---------------
def pick_input_device() -> int:
    devices = sd.query_devices()
    inputs = []
    print("\nAvailable input devices (microphones):")
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append(idx)
            print(f"  [{idx}] {d.get('name','?')}  —  {d['max_input_channels']} ch")
    if not inputs:
        raise RuntimeError("No input devices found. On macOS, allow Microphone for your terminal/IDE and restart.")

    try:
        default_in = sd.default.device[0]  # (input, output)
    except Exception:
        default_in = None
    if default_in not in inputs:
        default_in = inputs[0]

    sel = input(f"Select mic index [{default_in}]: ").strip()
    chosen = default_in if sel == "" else (int(sel) if sel.isdigit() else default_in)
    if chosen not in inputs:
        print("Invalid selection, using default.")
        chosen = default_in

    print(f"✅ Using input device [{chosen}] {devices[chosen]['name']}\n")
    return chosen

def list_monitors(sct):
    mons = []
    print("Available monitors:")
    for i, m in enumerate(sct.monitors[1:], start=1):
        print(f"  ({i}) left={m['left']} top={m['top']} width={m['width']} height={m['height']}")
        mons.append(i)
    return mons

def pick_monitor(sct) -> int:
    mons = list_monitors(sct)
    if not mons:
        raise RuntimeError("No monitors reported by MSS. On macOS, enable Screen Recording for your terminal/IDE and restart.")
    default_mon = 1 if 1 in mons else mons[0]
    sel = input(f"Select monitor to capture ({'/'.join(map(str, mons))}) [{default_mon}]: ").strip()
    chosen = default_mon if sel == "" else (int(sel) if sel.isdigit() else default_mon)
    if chosen not in mons:
        print("Invalid selection, using default.")
        chosen = default_mon
    print(f"✅ Capturing monitor {chosen}\n")
    return chosen


# --------------- Continuous Voice Listener (VAD + energy + faster-whisper) ---------------
class VoiceListener:
    """
    Listens continuously; when speech ends, saves WAV and transcribes with faster-whisper.
    Adds:
      - live level meter,
      - energy fallback if VAD misses,
      - saves utterance to temp if transcript empty for debugging.
    """
    def __init__(self, samplerate=16000, frame_ms=20,
                 vad_level=0, min_speech_ms=100, max_silence_ms=250,
                 whisper_model="base", enabled=True, debug=False,
                 input_device=None, energy_fallback=True, energy_dbfs_thresh=-60.0):
        self.samplerate = samplerate
        self.frame_ms = frame_ms
        self.frame_bytes = int(samplerate * frame_ms / 1000) * 2  # int16
        self.vad = webrtcvad.Vad(vad_level)
        self.min_speech_frames = int(min_speech_ms / frame_ms)
        self.max_silence_frames = int(max_silence_ms / frame_ms)
        self.enabled = enabled
        self.debug = debug

        self._audio_q = queue.Queue()
        self._utt_q = queue.Queue()
        self._level_q = queue.Queue()
        self._stop = threading.Event()
        self._on_text = None

        self._whisper_name = whisper_model
        self._whisper = None

        self.input_device = input_device
        self.energy_fallback = energy_fallback
        self.energy_dbfs_thresh = energy_dbfs_thresh  # e.g., -60 dBFS

    def _log(self, *a):
        if self.debug: print(*a)

    @staticmethod
    def _dbfs_from_pcm16(frame_bytes: bytes) -> float:
        if not frame_bytes:
            return -120.0
        s = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(s)) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        return max(-120.0, min(0.0, db))

    def start(self, on_text):
        self._on_text = on_text
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=int(self.frame_bytes/2),
            callback=self._audio_cb,
            device=self.input_device
        )
        self.stream.start()
        threading.Thread(target=self._vad_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()
        threading.Thread(target=self._level_meter_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

    def set_enabled(self, flag: bool):
        self.enabled = flag
        state = "ON" if flag else "OFF"
        print(f"\n🎙️  Mic listening: {state}")
        beep()

    def _audio_cb(self, indata, frames, time_info, status):
        if status: self._log("Audio status:", status)
        pcm16 = (indata[:,0] * 32767.0).astype(np.int16).tobytes()
        self._audio_q.put(pcm16)
        self._level_q.put(pcm16)

    def _level_meter_loop(self):
        last = 0
        while not self._stop.is_set():
            try:
                chunk = self._level_q.get(timeout=0.3)
            except queue.Empty:
                continue
            db = self._dbfs_from_pcm16(chunk)
            now = time.time()
            if now - last > 0.5:
                bar_len = int(max(0, (db + 60) / 60 * 30))  # -60..0 dBFS -> 0..30
                bar = "#" * bar_len
                sys.stdout.write(f"\r🎚️ Mic level: {db:6.1f} dBFS [{bar:<30}] ")
                sys.stdout.flush()
                last = now

    def _vad_loop(self):
        buf = b""
        speech_frames = []
        in_speech = False
        silence = 0

        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            buf += chunk
            while len(buf) >= self.frame_bytes:
                frame = buf[:self.frame_bytes]
                buf = buf[self.frame_bytes:]

                is_speech = False
                if self.enabled:
                    try:
                        is_speech = self.vad.is_speech(frame, self.samplerate)
                    except Exception:
                        is_speech = False
                    if not is_speech and self.energy_fallback:
                        db = self._dbfs_from_pcm16(frame)
                        if db >= self.energy_dbfs_thresh:
                            is_speech = True
                            self._log(f"Energy fallback: {db:.1f} dBFS >= {self.energy_dbfs_thresh} dBFS")

                if is_speech:
                    speech_frames.append(frame)
                    silence = 0
                    if not in_speech and len(speech_frames) >= self.min_speech_frames:
                        in_speech = True
                        self._log("VAD: start")
                else:
                    if in_speech:
                        silence += 1
                        if silence >= self.max_silence_frames:
                            pcm = b"".join(speech_frames)
                            speech_frames.clear()
                            in_speech = False
                            silence = 0
                            self._log("VAD: end → enqueue utterance")
                            self._enqueue_utterance(pcm)
                    else:
                        speech_frames.clear()

    def _enqueue_utterance(self, pcm_bytes: bytes):
        if not pcm_bytes: return
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        sf.write(tmp.name, samples, self.samplerate)
        self._utt_q.put(tmp.name)

    def _process_loop(self):
        # Lazy-load faster-whisper
        if WhisperModel is None:
            print("\n[STT error] faster-whisper not installed. Run: pip install faster-whisper\n")
            return
        if self._whisper is None:
            try:
                # Good CPU defaults on Mac; try small for better accuracy if fast enough
                self._whisper = WhisperModel(self._whisper_name, device="cpu", compute_type="int8")
                print(f"\n[STT] faster-whisper model loaded: {self._whisper_name}\n")
            except Exception as e:
                print("[STT error] faster-whisper load error:", e)
                return

        while not self._stop.is_set():
            try:
                wav_path = self._utt_q.get(timeout=0.1)
            except queue.Empty:
                continue

            text = ""
            try:
                print("🧠 Transcribing…", os.path.basename(wav_path))
                segments, info = self._whisper.transcribe(
                    wav_path,
                    vad_filter=False,     # we already did VAD
                    beam_size=1,          # low latency
                    best_of=1,
                    language=None,        # auto
                )
                parts = [seg.text for seg in segments]
                text = " ".join(parts).strip()
            except Exception as e:
                print("[STT error] faster-whisper transcribe error:", e)

            if not text:
                # Keep the file for inspection so you can play it back
                keep = wav_path
                print(f"[STT warn] Empty transcript. Saved utterance for debugging: {keep}")
            else:
                # Clean up successful utterances
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
                if self._on_text:
                    self._on_text(text)


# ---------------- App ----------------
def main():
    parser = argparse.ArgumentParser(description="Voice-only Screen QA (mac, faster-whisper, robust logging)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--whisper-model", default="base")  # base|small|medium|large-v3
    parser.add_argument("--mute", action="store_true", help="Start with mic OFF")
    parser.add_argument("--debug", action="store_true", help="Verbose VAD logs")
    parser.add_argument("--energy-fallback", action="store_true", help="Use energy fallback for speech detection")
    parser.add_argument("--energy-thresh", type=float, default=-60.0, help="Energy fallback threshold (dBFS)")
    args = parser.parse_args()

    # 1) Choose microphone
    try:
        mic_index = pick_input_device()
    except Exception as e:
        print(f"Microphone selection failed: {e}")
        sys.exit(1)

    # 2) Choose monitor
    with mss.mss() as sct:
        try:
            mon_idx = pick_monitor(sct)
            mon = sct.monitors[mon_idx]
        except Exception as e:
            print(f"Monitor selection failed: {e}")
            sys.exit(1)

        title = "Screen (M=toggle mic, Q=quit)"
        if not args.no_preview:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 960, 540)

        last_pil = None

        def on_transcript(text):
            nonlocal last_pil
            if not text.strip() or last_pil is None:
                return
            print(f"\n🎤 Detected utterance: {text!r}")
            beep()
            img_bytes = image_to_jpeg_bytes(last_pil, width_limit=args.width, quality=args.quality)
            print("Sending to Gemini...")
            answer = ask_gemini(img_bytes, text, model_name=args.model)
            print("\n=== Answer ===\n" + str(answer) + "\n==============\n")
            beep()
            speak(answer)

        listener = VoiceListener(
            whisper_model=args.whisper_model,
            enabled=not args.mute,
            input_device=mic_index,
            vad_level=0,
            min_speech_ms=100,
            max_silence_ms=250,
            debug=args.debug,
            energy_fallback=args.energy_fallback or True,  # force on by default
            energy_dbfs_thresh=args.energy_thresh,
        )
        listener.start(on_transcript)
        print("Running. Keys: M=toggle mic, Q=quit")
        print("🎙️  Mic listening:", "OFF" if args.mute else "ON")

        delay = max(1, int(1000 / args.fps))
        while True:
            raw = sct.grab(mon)
            frame_bgra = np.array(raw)
            frame_bgr = frame_bgra[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_pil = Image.fromarray(frame_rgb)

            if not args.no_preview:
                cv2.imshow(title, frame_bgr)
                key = cv2.waitKey(delay) & 0xFF
            else:
                key = cv2.waitKey(delay) & 0xFF

            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('m'), ord('M')):
                listener.set_enabled(not listener.enabled)

        listener.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()