import queue
import sys
import signal
import re
import shutil
import os

import numpy as np
import sounddevice as sd
from google.cloud import speech
try:
    from colorama import init as colorama_init
    colorama_init()  # enable ANSI on Windows consoles
except Exception:
    pass

# ========= Settings =========
LANGUAGE = "en-US"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 10          # 100ms chunks
REMOVE_FILLERS = True           # remove small fillers like "um", "uh", "hmm"

# ========= Helpers =========
audio_q = queue.Queue()
committed_text = ""             # finalized text from API
last_len = 0                    # for clearing leftover characters on the line

FILLER_RE = re.compile(r"\b(?:um+|uh+|hmm+|erm+|eh+)\b[,.\s]*", flags=re.IGNORECASE)

def clean_text(t: str) -> str:
    if REMOVE_FILLERS:
        t = FILLER_RE.sub("", t)
    t = re.sub(r"\s+", " ", t.strip())
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()

def one_line_preview(paragraph: str):
    """
    Render a single updating line (no newlines).
    It truncates to terminal width and shows an ellipsis prefix if needed.
    """
    global last_len
    paragraph = clean_text(paragraph)

    # figure available width (keep at least 10 chars fallback)
    try:
        width = shutil.get_terminal_size((100, 20)).columns
    except Exception:
        width = 100
    width = max(10, width)

    # reserve a little room so we don't wrap
    max_chars = max(10, width - 2)

    if len(paragraph) > max_chars:
        view = "…" + paragraph[-(max_chars - 1):]  # tail view with ellipsis
    else:
        view = paragraph

    # clear line and rewrite (ANSI: \r + clear line)
    # If ANSI not available, we also pad with spaces to overwrite leftovers.
    clear_seq = "\r\033[2K"
    print(clear_seq + view, end="", flush=True)

    # Extra spaces in case ANSI clear didn't fully clear (older shells)
    pad = max(0, last_len - len(view))
    if pad:
        print(" " * pad + "\r" + view, end="", flush=True)

    last_len = len(view)

def audio_callback(indata, frames, time_info, status):
    if status:
        # Put a newline so the warning doesn't break the single line UX
        print("\n[Audio warning]", status, file=sys.stderr, flush=True)
    audio_q.put((indata.copy() * 32767).astype(np.int16).tobytes())

def request_generator():
    from google.cloud import speech as _speech
    while True:
        chunk = audio_q.get()
        if chunk is None:
            return
        yield _speech.StreamingRecognizeRequest(audio_content=chunk)

def main():
    global committed_text
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE,
        enable_automatic_punctuation=True,
        # model="latest_long",  # optional
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,       # we need interim to show live
        single_utterance=False,     # keep listening until Ctrl+C
    )

    blocksize = int(SAMPLE_RATE / BLOCKS_PER_SECOND)
    print("🎙️  Single-line live dictation… press Ctrl+C to stop.", end="", flush=True)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        requests = request_generator()
        responses = client.streaming_recognize(streaming_config, requests)

        current_interim = ""

        try:
            for resp in responses:
                for result in resp.results:
                    alt = result.alternatives[0]
                    txt = clean_text(alt.transcript)

                    if result.is_final:
                        if txt:
                            committed_text = (committed_text + " " + txt).strip() if committed_text else txt
                        current_interim = ""
                        one_line_preview(committed_text)  # show committed only
                    else:
                        current_interim = txt
                        # committed + interim (single line)
                        live = (committed_text + " " + current_interim).strip() if current_interim else committed_text
                        one_line_preview(live)

        except KeyboardInterrupt:
            pass
        finally:
            audio_q.put(None)

    # Finish line, then print final paragraph on a new line
    print()  # move to next line cleanly
    final_paragraph = clean_text(committed_text)
    print("\n📝 Transcript:")
    print(final_paragraph if final_paragraph else "(No transcript captured.)")

if __name__ == "__main__":
    # Make Ctrl+C behave nicely on Windows
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Exiting.")
