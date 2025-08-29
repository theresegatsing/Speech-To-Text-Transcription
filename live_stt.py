import queue
import sys
import signal
import re

import numpy as np
import sounddevice as sd
from google.cloud import speech

# ========= Settings =========
LANGUAGE = "en-US"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 10        # 100ms chunks
SHOW_LIVE_PREVIEW = False     # True = show a single updating line; False = silent until final print
REMOVE_FILLERS = True         # remove small fillers like "um", "uh", "hmm"

# ========= Helpers =========
audio_q = queue.Queue()
final_chunks = []             # store final segments from Google
last_final = ""               # track last final to avoid accidental duplicates

FILLER_RE = re.compile(r"\b(?:um+|uh+|hmm+|erm+|eh+)\b[,.\s]*", flags=re.IGNORECASE)

def clean_text(t: str) -> str:
    """Normalize spacing/punctuation (and optionally remove fillers)."""
    t = t.strip()
    if REMOVE_FILLERS:
        t = FILLER_RE.sub("", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t)
    # Fix space before punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio warning] {status}", file=sys.stderr, flush=True)
    # float32 (-1..1) -> int16 bytes (LINEAR16)
    audio_q.put((indata.copy() * 32767).astype(np.int16).tobytes())

def request_generator():
    from google.cloud import speech as _speech
    while True:
        chunk = audio_q.get()
        if chunk is None:
            return
        yield _speech.StreamingRecognizeRequest(audio_content=chunk)

def main():
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE,
        enable_automatic_punctuation=True,
        # model="latest_long",  # uncomment if available in your region/quota
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=SHOW_LIVE_PREVIEW,  # only show partials if preview enabled
        single_utterance=False,             # keep listening until Ctrl+C
    )

    blocksize = int(SAMPLE_RATE / BLOCKS_PER_SECOND)
    print("🎙️  Listening… press Ctrl+C to stop. (Final-only mode)\n")

    # Optional: single-line live preview state
    live_preview_line = ""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        requests = request_generator()
        responses = client.streaming_recognize(streaming_config, requests)

        try:
            for resp in responses:
                for result in resp.results:
                    alt = result.alternatives[0]
                    txt = clean_text(alt.transcript)

                    # Optional single-line preview of interim
                    if SHOW_LIVE_PREVIEW and not result.is_final:
                        # overwrite same line
                        live_preview_line = f"\rpreview: {txt}..."
                        print(live_preview_line, end="", flush=True)
                        continue

                    # Handle finals only
                    if result.is_final:
                        # Clear preview line if it was shown
                        if SHOW_LIVE_PREVIEW and live_preview_line:
                            print("\r" + " " * max(len(live_preview_line) - 1, 0), end="\r")
                            live_preview_line = ""

                        # Avoid duplicates (API sometimes resends the same final)
                        global last_final
                        if txt and txt != last_final:
                            final_chunks.append(txt)
                            last_final = txt

        except KeyboardInterrupt:
            pass
        finally:
            audio_q.put(None)

    # ===== Print a single clean paragraph at the end =====
    if final_chunks:
        paragraph = clean_text(" ".join(final_chunks))
        print("\n📝 Transcript (single paragraph):")
        print(paragraph)
    else:
        print("\n(No final transcript captured.)")

if __name__ == "__main__":
    # Make Ctrl+C behave nicely on Windows
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Exiting.")
