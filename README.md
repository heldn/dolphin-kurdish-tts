<div align="center">

# 🐬 Dolphin KURDISH TTS

### Free & Open-Source Kurdish Text-to-Speech  
**By Heldn Hastyar Abdullah**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Attribution--Required-orange)](LICENSE)
[![Gradio](https://img.shields.io/badge/Powered%20by-Gradio-FF4B4B)](https://gradio.app)
[![Kurdish](https://img.shields.io/badge/Supports-Sorani%20%26%20Kurmanji-green)]()

<br>

> **“Voice for every Kurdish word”**  
> Convert Kurdish text into natural-sounding speech — **offline, unlimited, and free**.

</div>

![Dolphin KURDISH TTS Demo](demo.png)

---

## 🌟 About

**Dolphin KURDISH TTS** is a free, open-source Text-to-Speech application designed specifically for the Kurdish language.  
It supports **Sorani** and **Kurmanji** dialects with clean text processing, high-quality audio output, and an easy-to-use web interface.

The goal is simple:  
**Make Kurdish accessible, audible, and future-proof.**

---

## ✨ Features

| Feature | Description |
|------|------------|
| 🗣️ **Unlimited Text** | Convert long articles, poems, books, or stories without limits |
| 🇰🇷 **Full Kurdish Support** | Sorani (Arabic script) • Kurmanji (Arabic & Latin scripts) |
| 🧹 **Smart Text Cleaner** | Fixes Arabic/Persian characters, numbers, and common typing issues |
| 🎧 **Professional Output** | WAV / MP3 audio + SRT subtitles |
| 🎬 **Video-Ready** | ZIP bundle (audio + subtitles) for editors |
| ⚙️ **Speech Controls** | Speed, pitch, and natural pauses |
| 📁 **File Upload** | Process `.txt` files directly |
| 🔒 **Offline Mode** | Works without internet after first run |

---

📖 How to Use
🎛️ Studio Tab
Enter text or upload a .txt file

Choose dialect:

Sorani – Central Kurdish (Arabic script)

Kurmanji (Arabic) – Northern Kurdish (Arabic script)

Kurmanji (Latin) – Northern Kurdish (Latin script)

(Optional) Adjust speech settings:

⏱️ Speed: 0.5x → 2.0x

🔊 Pitch: -5 → +5

⏸️ Pauses: Sentence spacing

Click Generate Speech

Download:

🎧 Audio (WAV / MP3)

📝 Subtitles (.srt)

📦 ZIP bundle

🧹 Text Cleaner Tab
Automatically fixes:

Broken Arabic/Persian characters

ك → ک

ي → ی

Number formatting

123 → ١٢٣

Common Kurdish typing mistakes

⚙️ Technical Details
Model: Meta AI MMS-TTS (Massively Multilingual Speech)

Audio Quality: 16 kHz

RAM: 8 GB+ recommended for long texts

MP3 Support: Requires FFmpeg (WAV works by default)

Offline Mode: Models cached after first use

🙏 Acknowledgements
Meta AI — MMS-TTS models

Hugging Face — model hosting

Gradio — web interface

The Kurdish language community ❤️

⚖️ License & Attribution
This project is free and open-source, but attribution is required.

You must credit:

“Dolphin KURDISH TTS by Heldn Hastyar Abdullah”

Required in:

Application UI

Documentation

Source code comments

Promotional materials

📄 See full terms in the LICENSE file.

<div align="center">
Made with 💙 for the Kurdish Nation
بۆ گەلی کورد، بە زمانی کوردی
</div>

## 🚀 Quick Start

### ▶️ Run Locally (Recommended)

```bash
# Clone the repository
git clone https://github.com/heldn/dolphin-kurdish-tts.git
cd dolphin-kurdish-tts

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
After launch, open the local Gradio link in your browser.