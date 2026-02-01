import gradio as gr
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import os
import re
import json
from datetime import datetime
import librosa
import soundfile as sf
from pydub import AudioSegment
import zipfile

# --- CONFIGURATION ---
MODELS = {
    "Sorani": "razhan/mms-tts-ckb",
    "Kurmanji (Arabic Script)": "facebook/mms-tts-kmr-script_arabic",
    "Kurmanji (Latin Script)": "facebook/mms-tts-kmr-script_latin"
}

OUTPUT_FOLDER = "audio_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model_cache = {}

# --- TEXT CLEANER (Kurdish-specific) ---
def normalize_kurdish_text(text):
    """Fixes common Kurdish/Arabic/Persian character mapping issues."""
    eng_to_ku = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    text = text.translate(eng_to_ku)
    
    replacements = {
        'ك': 'ک', 'ي': 'ی', 'ى': 'ی', 'ه': 'ه', 'ە': 'ە',
        'ڇ': 'چ', 'گ': 'گ', 'ڵ': 'ڵ', 'ڕ': 'ڕ', 'ڤ': 'ڤ', 'ێ': 'ێ', 'ۆ': 'ۆ', 'ه‌': 'ه'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def auto_punctuate(text):
    if not text.strip():
        return text
    text = re.sub(r'([.؟!?،])(\S)', r'\1 \2', text)
    if not re.search(r'[.؟!]\s*$', text.strip()):
        text = text.rstrip() + '.'
    return text

# --- SMART TEXT SPLITTER FOR KURDISH (FIXED) ---
def split_into_chunks(text, max_chars=400):
    """
    Split text into chunks <= max_chars without breaking sentences.
    Handles Kurdish punctuation: . ؟ ! 
    """
    # Clean up whitespace first
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentence endings (keep delimiters)
    parts = re.split(r'([.؟!]+)', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i] + parts[i + 1]
        if sent.strip():
            sentences.append(sent.strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    
    # Fallback: if no sentence breaks found, split by words
    if not sentences:
        words = text.split()
        temp = ""
        for word in words:
            if len(temp) + len(word) + 1 <= max_chars:
                temp = temp + " " + word if temp else word
            else:
                if temp:
                    sentences.append(temp)
                temp = word
        if temp:
            sentences.append(temp)
        return sentences if sentences else [text]
    
    # Build chunks
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        # Skip empty sentences
        if not sent.strip():
            continue
            
        # If adding this sentence exceeds limit, finalize current chunk
        if current_chunk and len(current_chunk) + len(sent) + 1 > max_chars:
            chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk = current_chunk + " " + sent if current_chunk else sent
    
    # Add last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Final safety: remove any empty chunks
    chunks = [c for c in chunks if c.strip()]
    
    return chunks if chunks else [text]

# --- AUDIO ENGINE (Handles Long Texts) ---
def load_model(dialect_name):
    if dialect_name in model_cache:
        return model_cache[dialect_name]
    try:
        model = VitsModel.from_pretrained(MODELS[dialect_name])
        tokenizer = AutoTokenizer.from_pretrained(MODELS[dialect_name])
        model_cache[dialect_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        return None, str(e)

def format_timestamp(seconds):
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def generate_audio_engine(text, dialect, speed, pitch, use_mp3, pause_short, pause_long):
    if len(text.strip()) == 0:
        raise gr.Error("❌ Please enter some text!")
    
    # Clean & Fix Text
    if "Sorani" in dialect: 
        text = normalize_kurdish_text(text)
    
    if not re.search(r'[.؟!,،]', text[:50]):
        text = auto_punctuate(text)
    
    # Ensure we have valid text after cleaning
    text = text.strip()
    if not text:
        raise gr.Error("❌ Text became empty after cleaning!")
    
    # Load Model
    model_obj = load_model(dialect)
    if model_obj[0] is None:
        raise gr.Error(f"Model Error: {model_obj[1]}")
    model, tokenizer = model_obj
    sr = model.config.sampling_rate

    # Split into manageable chunks
    chunks = split_into_chunks(text, max_chars=400)
    
    # Critical safety check
    if not chunks:
        raise gr.Error("❌ Could not process text - please check your input!")
    
    print(f"Split text into {len(chunks)} chunks")
    
    full_audio_segments = []
    srt_entries = []
    current_time = 0.0
    silence_between_chunks = np.zeros(int(sr * pause_long))  # Natural pause between chunks

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Process chunk like before
        parts = re.split(r'([.؟!:\n]+|\[p\]|\[s\])', chunk)
        audio_segments = []
        chunk_srt = []
        chunk_time = 0.0
        
        silence_short_dur = np.zeros(int(sr * pause_short))
        silence_long_dur = np.zeros(int(sr * pause_long))
        silence_between_words = np.zeros(int(sr * 0.1))

        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if part == "[p]":
                audio_segments.append(silence_long_dur)
                chunk_time += pause_long
                continue
            if part == "[s]" or re.match(r'^[.؟!:\n]+$', part):
                audio_segments.append(silence_short_dur)
                chunk_time += pause_short
                continue

            inputs = tokenizer(part, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            segment = output.float().numpy().T.flatten()

            if speed != 1.0:
                segment = librosa.effects.time_stretch(segment, rate=speed)
            if pitch != 0:
                segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=pitch)

            duration = len(segment) / sr
            start = format_timestamp(current_time + chunk_time)
            end = format_timestamp(current_time + chunk_time + duration)
            chunk_srt.append(f"{len(srt_entries)+len(chunk_srt)+1}\n{start} --> {end}\n{part}\n")
            
            audio_segments.append(segment)
            audio_segments.append(silence_between_words)
            chunk_time += duration + 0.1

        if audio_segments:
            chunk_audio = np.concatenate(audio_segments)
            full_audio_segments.append(chunk_audio)
            srt_entries.extend(chunk_srt)
            current_time += chunk_time
            
            # Add natural pause between chunks (except after last chunk)
            if i < len(chunks) - 1:
                full_audio_segments.append(silence_between_chunks)
                current_time += pause_long

    if not full_audio_segments:
        return None, None, None, None

    full_audio = np.concatenate(full_audio_segments)
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"{OUTPUT_FOLDER}/audio_{timestamp}.wav"
    sf.write(wav_path, full_audio, sr)
    
    final_audio_path = wav_path
    if use_mp3:
        mp3_path = f"{OUTPUT_FOLDER}/audio_{timestamp}.mp3"
        try:
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format="mp3", bitrate="192k")
            final_audio_path = mp3_path
        except Exception as e:
            print(f"MP3 conversion failed: {e}")
    
    srt_path = f"{OUTPUT_FOLDER}/audio_{timestamp}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.writelines(srt_entries)
    
    zip_path = f"{OUTPUT_FOLDER}/audio_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(final_audio_path, os.path.basename(final_audio_path))
        zf.write(srt_path, os.path.basename(srt_path))

    return (sr, full_audio), final_audio_path, srt_path, zip_path

# --- UI ---
theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")

import warnings
warnings.filterwarnings("ignore", message=".*theme.*launch.*")

with gr.Blocks(theme=theme, title="Dolphin KURDISH TTS") as demo:
    gr.Markdown("# 🐬 Dolphin KURDISH TTS\n**By Heldn Hastyar Abdullah**")
    
    with gr.Tabs():
        # TAB 1: MAIN STUDIO
        with gr.TabItem("🎛️ Studio"):
            with gr.Row():
                with gr.Column():
                    dialect = gr.Dropdown(
                        list(MODELS.keys()), 
                        value="Sorani", 
                        label="Dialect"
                    )
                    
                    text_upload = gr.File(
                        label="📄 Upload .txt file", 
                        file_types=[".txt"], 
                        type="filepath"
                    )
                    
                    gr.Markdown("✅ Supports unlimited text length!")
                    
                    text_input = gr.Textbox(
                        lines=10, 
                        label="Input Kurdish Text", 
                        rtl=True, 
                        placeholder="دەقی کوردی لێرە بنوسە... (هەر چەند پیتێک بێت)"
                    )
                    
                    with gr.Accordion("⏸️ Natural Pauses", open=False):
                        pause_short = gr.Slider(0.2, 0.8, value=0.4, label="Comma pause (seconds)")
                        pause_long = gr.Slider(0.8, 2.0, value=1.3, label="Sentence pause (seconds)")
                    
                    with gr.Accordion("⚙️ Audio Settings", open=True):
                        speed = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                        pitch = gr.Slider(-5, 5, value=0, step=1, label="Pitch")
                        use_mp3 = gr.Checkbox(label="Export as MP3 (Smaller File)", value=False)
                    
                    btn = gr.Button("🔊 Generate Speech", variant="primary")
                
                with gr.Column():
                    audio_out = gr.Audio(label="Audio Preview")
                    file_out = gr.File(label="Audio File")
                    srt_out = gr.File(label="Subtitles (.srt)")
                    zip_out = gr.File(label="📦 ZIP Bundle")
            
            text_upload.change(
                lambda f: open(f.name, encoding='utf-8').read() if f else "", 
                inputs=text_upload, 
                outputs=text_input
            )
            
            btn.click(
                generate_audio_engine,
                inputs=[text_input, dialect, speed, pitch, use_mp3, pause_short, pause_long],
                outputs=[audio_out, file_out, srt_out, zip_out],
                show_progress=True
            )

        # TAB 2: TEXT CLEANER
        with gr.TabItem("🧹 Text Cleaner"):
            gr.Markdown("### Fix broken Kurdish characters\nPaste messy text to normalize Kurdish letters and numbers.")
            raw_text = gr.Textbox(
                lines=6, 
                label="Original Text", 
                rtl=True,
                placeholder="دەقی ناخوازراو لێرە دابنێ..."
            )
            clean_btn = gr.Button("Clean Text", variant="secondary")
            clean_text = gr.Textbox(
                lines=6, 
                label="Cleaned Text", 
                rtl=True,
                placeholder="دەقی ڕێکخراو ئەمەیە..."
            )

            clean_btn.click(
                normalize_kurdish_text,
                inputs=raw_text,
                outputs=clean_text
            )
        
        # TAB 3: ABOUT (Documentation - NOW LAST)
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            # 🐬 Dolphin KURDISH TTS
            
            **Created by:** Heldn Hastyar Abdullah  
            **License:** Free & Open Source
            
            ### What is this?
            A free tool to convert **Kurdish text to natural-sounding speech**.
            
            ### Attribution Required
            This project is free to use and modify, but **you must credit "Heldn Hastyar Abdullah"** 
            if you reuse this code or distribute the application.
            
            ### Supported Dialects
            - **Sorani** (Central Kurdish - Arabic script)
            - **Kurmanji** (Northern Kurdish - Arabic or Latin script)
            
            ### How to Use
            1. Go to the **🎛️ Studio** tab
            2. Enter your Kurdish text (or upload a `.txt` file)
            3. Select your dialect
            4. Adjust settings if needed (speed, pitch, pauses)
            5. Click **"Generate Speech"**
            6. Download your audio, subtitles, or ZIP bundle
            
            ### Text Cleaner
            - Use the **🧹 Text Cleaner** tab to fix:
              - Broken Arabic/Persian characters
              - Number formatting issues
              - Common typing errors
            
            ### Tips for Best Results
            - Use proper Kurdish punctuation (، . ؟ !)
            - For long texts: Upload `.txt` files instead of pasting
            - Sorani dialect automatically fixes character mappings
            - Use `[p]` for long pauses, `[s]` for short pauses
            
            ### Technical Notes
            - **No internet required** after first run (models cached locally)
            - **MP3 conversion** requires FFmpeg (WAV always works)
            - Processing time: ~1 second per 100 characters
            """)

    gr.Markdown(
        "### Usage Tips\n"
        "- **Max quality**: Use proper Kurdish punctuation\n"
        "- **Long texts**: Upload .txt files for best results\n"
        "- **Sorani users**: Automatic character fixing enabled\n"
        "- **Video creators**: Download ZIP bundle (audio + subtitles)\n\n"
        "<div style='text-align: center; padding: 15px; background-color: #e6f7ff; border-radius: 8px;'>"
        "<strong>🐬 Dolphin KURDISH TTS</strong> • Created by <em>Heldn Hastyar Abdullah</em> • "
        "Free & Open Source (Attribution Required)"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)