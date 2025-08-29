import queue
import sys
import signal
import re
import shutil

import numpy as np
import sounddevice as sd
from google.cloud import speech

# ========= Settings =========
LANGUAGE = "en-US"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 10          # 100ms chunks
REMOVE_FILLERS = True           # remove small fillers like "um", "uh", "hmm"
SOFT_WRAP = True                # wrap to terminal width (keeps output tidy)

# ========= Helpers =========
audio_q = queue.Queue()
committed_text = ""             # finalized text from API
last_rendered = ""              # avoid repainting same string

FILLER_RE = re.compile(r"\b(?:um+|uh+|hmm+|erm+|eh+)\b[,.\s]*", flags=re.IGNORECASE)

def clean_text(t: str) -> str:
    """Normalize spacing/punctuation (and optionally remove fillers)."""
    if REMOVE_FILLERS:
        t = FILLER_RE.sub("", t)
    t = re.sub(r"\s+", " ", t.strip())
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()

def render_live(committed: str, interim: str):
    """Render a single updating line/paragraph: committed + interim."""
    global last_rendered
    text = (committed + " " + interim).strip() if interim else committed.strip()
    text = clean_text(text)

    # Optional soft wrap to terminal width for nicer display
    if SOFT_WRAP:
        try:
            width = shutil.get_terminal_size((100, 20)).columns
        except Exception:
            width = 100
        # Simple wrap: break long text into lines at spaces
        out_lines = []
        line = ""
        for word in text.split(" "):
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= width:
                line += " " + word
            else:
                out_lines.append(line)
                line = word
        if line:
            out_lines.append(line)
        text_to_print = "\n".join(out_lines)
        # Clear previously printed lines by printing enough newlines and carriage returns
        # Simplest reliable approach: print a leading carriage return, then full block, then trailing spaces on last line.
        if text_to_print != last_rendered:
            # Clear screen section (cheap): print \r then the new block and a final newline-less line
            # For PowerShell/cmd this is fine; Git Bash also works.
            print("\r", end="")
            # Move cursor up for multi-line re-render: print enough newlines to reset, then repaint
            # (This simple version just repaints; the extra blank line avoids leftover chars)
            print(text_to_print + ("\n" if not text_to_print.endswith("\n") else ""), end="", flush=True)
            last_rendered = text_to_print
    else:
        # Single line: overwrite the same line using carriage return
        if text != last_rendered:
            print("\r" + text + " " * max(0, len(last_rendered) - len(text)), end="", flush=True)
            last_rendered = text

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"\n[Audio warning] {status}", file=sys.stderr, flush=True)
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
    global committed_text
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE,
        enable_automatic_punctuation=True,
        # model="latest_long",  # uncomment if available to you
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,       # we need interim for live preview
        single_utterance=False,     # keep listening until Ctrl+C
    )

    blocksize = int(SAMPLE_RATE / BLOCKS_PER_SECOND)
    print("🎙️  Live dictation… press Ctrl+C to stop.\n")

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
                        # Commit finalized chunk and clear interim
                        if txt:
                            # Add space if needed before appending
                            committed_text = (committed_text + " " + txt).strip() if committed_text else txt
                        current_interim = ""
                        render_live(committed_text, current_interim)
                    else:
                        # Update live with interim appended to committed
                        current_interim = txt
                        render_live(committed_text, current_interim)

        except KeyboardInterrupt:
            pass
        finally:
            audio_q.put(None)

    # Final clean paragraph
    final_paragraph = clean_text(committed_text)
    print("\n\n📝 Transcript:")
    print(final_paragraph if final_paragraph else "(No transcript captured.)")

if __name__ == "__main__":
    # Make Ctrl+C behave nicely on Windows
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Exiting.")
