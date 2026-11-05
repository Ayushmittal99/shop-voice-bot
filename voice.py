"""Speech-to-text (Groq Whisper) and text-to-speech (offline, pyttsx3)."""
import io
import os
import re
import time
import wave

from config import SAMPLE_RATE, STT_LANGUAGE, STT_MODEL, TTS_RATE
from generator import _get_client
from logger_setup import get_logger, log_event

logger = get_logger()


def record_until_enter() -> bytes:
    """Records from the microphone until the user presses Enter. Returns WAV bytes."""
    import numpy as np
    import sounddevice as sd  # imported here so a missing mic library never breaks typing

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input("  Recording... press Enter to stop. ")

    if not frames:
        return b""
    audio = np.concatenate(frames)
    if len(audio) < SAMPLE_RATE * 0.3:  # shorter than 0.3 s: nothing useful
        return b""

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def speech_to_text(wav_bytes: bytes, filename: str = "question.wav") -> str:
    start = time.perf_counter()
    kwargs = {"language": STT_LANGUAGE} if STT_LANGUAGE else {}
    text = _get_client().audio.transcriptions.create(
        file=(filename, wav_bytes),
        model=STT_MODEL,
        response_format="text",
        **kwargs,
    )
    text = str(text).strip()
    log_event(logger, "Speech to text", model=STT_MODEL, chars=len(text),
              seconds=round(time.perf_counter() - start, 3))
    return text


def clean_for_speech(text: str) -> str:
    """Make the answer sound natural: 'Rs 2,499' -> '2,499 rupees', no product ids or markdown."""
    text = re.sub(r"\[P?\w*\d+\]", "", text)                       # product ids like [P012]
    text = re.sub(r"(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", text)
    text = re.sub(r"\bMRP\b", "M R P", text)
    text = re.sub(r"[*_`#>|]+", " ", text)                          # markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def _speak_windows(text: str):
    """Windows built-in voice (System.Speech) through PowerShell. Nothing to install."""
    import subprocess

    # TTS_RATE ~175 wpm is normal speed; System.Speech rate runs from -10 to 10
    rate = max(-10, min(10, round((TTS_RATE - 175) / 15)))
    script = (
        "[Console]::InputEncoding=[Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.Rate={rate};"
        "$s.Speak([Console]::In.ReadToEnd())"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        input=text.encode("utf-8"), check=True, timeout=300,
    )


def text_to_speech(text: str) -> float:
    """Speaks the text through the speakers. Returns seconds spent speaking."""
    start = time.perf_counter()
    spoken = clean_for_speech(text)
    if not spoken:
        return 0.0

    try:
        import pyttsx3
    except ImportError:
        pyttsx3 = None

    if pyttsx3 is not None:
        # a fresh engine per call avoids a known pyttsx3 bug where the 2nd call stays silent
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.say(spoken)
        engine.runAndWait()
        engine.stop()
        engine_name = "pyttsx3"
    elif os.name == "nt":
        _speak_windows(spoken)
        engine_name = "windows-builtin"
    else:
        raise RuntimeError("No text-to-speech engine: run `pip install pyttsx3`")

    seconds = round(time.perf_counter() - start, 3)
    log_event(logger, "Text to speech", engine=engine_name, chars=len(spoken), seconds=seconds)
    return seconds
