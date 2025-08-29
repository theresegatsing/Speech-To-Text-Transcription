# 🎙️ Live Speech-to-Text (Google Cloud) — Mic → One Paragraph

A Python tool that transcribes your **live microphone audio** using **Google Cloud Speech-to-Text**.  
While you speak, the console shows a **single, non-scrolling line** that updates in place.  
When you press **Ctrl+C**, it prints one **clean paragraph** (with optional filler-word removal).



## 📸 Demo (What you see)
*🎙️  Single-line live dictation… press Ctrl+C to stop.* <br>
*…in real time the last part of your paragraph appears here…*
*📝 Transcript:*
*Here is the final paragraph with punctuation.*



## Features

- ✅ Live single-line preview (no scrolling)

- ✅ Final paragraph printout

- ✅ Optional filler removal (um, uh, hmm)

- ✅ Language configurable (e.g., en-US, fr-FR)

- ✅ Uses Google’s Streaming Recognize for low latency

- ✅ Designed for Windows terminal (works in PowerShell, cmd, Git Bash)

---

## Requirements

- Python 3.11+

- A Google Cloud project with Speech-to-Text API enabled

- A Service Account JSON key (downloaded to your machine)

- Windows mic permission enabled:

- Settings → Privacy & security → Microphone → allow desktop apps









