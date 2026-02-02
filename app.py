import sys
import os

# 1. SETUP DIRECTORIES (Must happen before AI imports)
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Redirect all AI models to a local cache folder for portability
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
os.environ["PYTHONHASHSEED"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Force offline mode if models are already present
if os.path.exists(os.path.join(MODEL_CACHE_DIR, "models--razhan--mms-tts-ckb")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("Status: Offline Mode Active (Using local models)")
else:
    print("Status: Online Mode (Will download models on first run)")

import re
import json
import zipfile
from datetime import datetime
import logging

# Handle console output
print("====================================")
print("   🐬 Dolphin KURDISH TTS 🐬")
print("====================================")
print("Status: Initializing AI Engine...")
print(f"Model cache directory: {os.environ['HF_HOME']}")
print("Note: First launch takes longer to unpack libraries.")
print("------------------------------------")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import gradio as gr
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

# --- CONFIGURATION ---
OUTPUT_FOLDER_NAME = "audio_output"
OUTPUT_FOLDER = os.path.join(BASE_DIR, OUTPUT_FOLDER_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODELS = {
    "Sorani": "razhan/mms-tts-ckb",
    "Sorani (Alternative)": "akam-ot/ckb-tts",
    "Kurmanji (Arabic Script)": "facebook/mms-tts-kmr-script_arabic",
    "Kurmanji (Latin Script)": "facebook/mms-tts-kmr-script_latin"
}

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "English": {
        "title": "🐬 Dolphin KURDISH TTS",
        "author": "By Heldn Hastyar Abdullah",
        "studio_tab": "🎛️ Studio",
        "cleaner_tab": "🧹 Text Cleaner",
        "about_tab": "ℹ️ About",
        "dialect": "Dialect",
        "upload_txt": "📄 Upload .txt file",
        "unlimited_text": "✅ Supports unlimited text length!",
        "input_placeholder": "Enter Kurdish text here...",
        "input_label": "Input Kurdish Text",
        "pauses_accordion": "⏸️ Natural Pauses",
        "comma_pause": "Comma pause (seconds)",
        "sentence_pause": "Sentence pause (seconds)",
        "audio_settings": "⚙️ Audio Settings",
        "speed": "Speed",
        "pitch": "Pitch",
        "export_mp3": "Export as MP3 (Smaller File)",
        "generate_btn": "🔊 Generate Speech",
        "audio_preview": "Audio Preview",
        "audio_file": "Audio File",
        "subtitles": "Subtitles (.srt)",
        "zip_bundle": "📦 ZIP Bundle",
        "clean_title": "### Fix broken Kurdish characters",
        "clean_desc": "Paste messy text to normalize Kurdish letters and numbers.",
        "original_text": "Original Text",
        "clean_btn": "Clean Text",
        "cleaned_text": "Cleaned Text",
        "usage_tips": "### Usage Tips",
        "tip_q": "- **Max quality**: Use proper Kurdish punctuation",
        "tip_l": "- **Long texts**: Upload .txt files for best results",
        "tip_s": "- **Sorani users**: Automatic character fixing enabled",
        "tip_v": "- **Video creators**: Download ZIP bundle (audio + subtitles)",
        "footer": "<strong>🐬 Dolphin KURDISH TTS</strong> • Created by <em>Heldn Hastyar Abdullah</em> • Free & Open Source",
        "error_empty": "❌ Please enter some text!",
        "error_clean_empty": "❌ Text became empty after cleaning!",
        "error_process": "❌ Could not process text!"
    },
    "Kurdish": {
        "title": "🐬 دۆڵفین بۆ گۆڕینی دەق بۆ دەنگ",
        "author": "لەلایەن هێڵدن هەستیار عەبدوڵا",
        "studio_tab": "🎛️ ستۆدیۆ",
        "cleaner_tab": "🧹 چاکسازی دەق",
        "about_tab": "ℹ️ دەربارە",
        "dialect": "شێوەزار",
        "upload_txt": "📄 بارکردنی فایلی .txt",
        "unlimited_text": "✅ دەتوانی دەقی بێسنوور بنوسی!",
        "input_label": "دەقی کوردی داخڵ بکە",
        "input_placeholder": "دەقی کوردی لێرە بنوسە...",
        "pauses_accordion": "⏸️ وەستانە سروشتییەکان",
        "comma_pause": "وەستانی فاریزە (چرکە)",
        "sentence_pause": "وەستانی خاڵ (چرکە)",
        "audio_settings": "⚙️ ڕێکخستنی دەنگ",
        "speed": "خێرایی",
        "pitch": "تۆنی دەنگ",
        "export_mp3": "هەناردەکردن بە MP3",
        "generate_btn": "🔊 دروستکردنی دەنگ",
        "audio_preview": "گوێگرتن",
        "audio_file": "فایلی دەنگ",
        "subtitles": "ژێرنووس (.srt)",
        "zip_bundle": "📦 فایلی ZIP",
        "clean_title": "### چاککردنی پیتە تێکچووەکان",
        "clean_desc": "دەقی تێکچوو لێرە دابنێ بۆ ئەوەی پیت و ژمارەکانی ڕێکبخەیتەوە.",
        "original_text": "دەقی سەرەکی",
        "clean_btn": "ڕێکخستنی دەق",
        "cleaned_text": "دەقی ڕێکخراو",
        "usage_tips": "### ئامۆژگارییەکان",
        "tip_q": "- **باشترین کوالێتی**: نیشانەکانی (، . ؟ !) بەکاربهێنە",
        "tip_l": "- **دەقی درێژ**: فایلی .txt بەکاربهێنە",
        "tip_s": "- **بۆ سۆرانی**: چاکسازی پیتەکان چالاککراوە",
        "tip_v": "- **بۆ ڤیدیۆ**: فایلی ZIP دابەزێنە",
        "footer": "<strong>دۆڵفین بۆ گۆڕینی دەق بۆ دەنگ</strong> • لەلایەن <em>هێڵدن هەستیار عەبدوڵا</em>",
        "error_empty": "❌ تکایە دەقێک بنوسە!",
        "error_clean_empty": "❌ دەقەکە خاڵییە!",
        "error_process": "❌ کێشەیەک ڕوویدا!"
    }
}

model_cache = {}

# --- TEXT CLEANER ---
def normalize_kurdish_text(text: str) -> str:
    if not text: return ""
    text = text.replace('4', '٤').replace('5', '٥').replace('6', '٦')
    text = text.replace('۴', '٤').replace('۵', '٥').replace('۶', '٦')
    eng_to_ku = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    text = text.translate(eng_to_ku)
    replacements = {
        'ك': 'ک', 'ي': 'ی', 'ى': 'ی', 'ة': 'ە',
        'ڇ': 'چ', 'ڤ': 'ڤ', 'ڥ': 'ڤ', 'ڦ': 'پ',
        'ه‌': 'ە', 'ە‌': 'ە', 'ۆ': 'ۆ', 'ێ': 'ێ',
        'ڕ': 'ڕ', 'ڵ': 'ڵ', '?': '؟', ',': '،', ';': '؛'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def auto_punctuate(text):
    if not text.strip(): return text
    text = re.sub(r'([.؟!?،])(\S)', r'\1 \2', text)
    if not re.search(r'[.؟!]\s*$', text.strip()):
        text = text.rstrip() + '.'
    return text

def split_into_chunks(text, max_chars=400):
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) <= max_chars: return [text]
    parts = re.split(r'([.؟!]+)', text)
    sentences = []
    for i in range(0, len(parts)-1, 2):
        s = parts[i] + parts[i+1]
        if s.strip(): sentences.append(s.strip())
    if len(parts)%2==1 and parts[-1].strip(): sentences.append(parts[-1].strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else: current = current + " " + s if current else s
    if current: chunks.append(current)
    return chunks

# --- AUDIO ENGINE ---
def load_model(dialect_name):
    if dialect_name in model_cache: return model_cache[dialect_name]
    try:
        logger.info(f"🚀 Loading model for {dialect_name}...")
        try:
            # First attempt: Try loading from local cache ONLY (true offline)
            model = VitsModel.from_pretrained(MODELS[dialect_name], cache_dir=MODEL_CACHE_DIR, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(MODELS[dialect_name], cache_dir=MODEL_CACHE_DIR, local_files_only=True)
        except Exception as offline_err:
            # Second attempt: If not in cache, download it
            logger.info(f"📡 Model not found in local cache or checking for updates... ({dialect_name})")
            model = VitsModel.from_pretrained(MODELS[dialect_name], cache_dir=MODEL_CACHE_DIR, local_files_only=False)
            tokenizer = AutoTokenizer.from_pretrained(MODELS[dialect_name], cache_dir=MODEL_CACHE_DIR, local_files_only=False)
            
        model_cache[dialect_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        error_msg = str(e)
        if "incomplete metadata" in error_msg or "deserializing" in error_msg:
            error_msg = "❌ Corrupted model file detected! Please delete the 'models_cache' folder and restart the app to redownload."
        logger.error(f"Failed: {error_msg}")
        return None, error_msg

def format_timestamp(s):
    ms = int((s % 1) * 1000)
    s = int(s)
    return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},{ms:03}"

def generate_audio_engine(text, dialect, speed, pitch, use_mp3, p_s, p_l):
    if not text.strip(): raise gr.Error("Empty!")
    text = normalize_kurdish_text(text)
    if not re.search(r'[.؟!,،]', text[:50]): text = auto_punctuate(text)
    m_obj = load_model(dialect)
    if not m_obj[0]: raise gr.Error(str(m_obj[1]))
    model, tok = m_obj
    sr = model.config.sampling_rate
    chunks = split_into_chunks(text.strip())
    
    aud_segs, srt_segs, cur_t = [], [], 0.0
    for i, ch in enumerate(chunks):
        parts = re.split(r'([.؟!:\n]+|\[p\]|\[s\])', ch)
        ch_aud, ch_t = [], 0.0
        for p in parts:
            p = p.strip()
            if not p: continue
            if p == "[p]": 
                ch_aud.append(np.zeros(int(sr*p_l))); ch_t+=p_l; continue
            if p == "[s]" or re.match(r'^[.؟!:\n]+$', p):
                ch_aud.append(np.zeros(int(sr*p_s))); ch_t+=p_s; continue
            if len(p) < 2: continue
            ins = tok(p, return_tensors="pt")
            if ins['input_ids'].shape[-1] == 0: continue
            with torch.no_grad(): out = model(**ins).waveform
            seg = out.float().numpy().T.flatten()
            if speed != 1.0: seg = librosa.effects.time_stretch(seg, rate=speed)
            if pitch != 0: seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=pitch)
            dur = len(seg)/sr
            srt_segs.append(f"{len(srt_segs)+1}\n{format_timestamp(cur_t+ch_t)} --> {format_timestamp(cur_t+ch_t+dur)}\n{p}\n\n")
            ch_aud.append(seg); ch_aud.append(np.zeros(int(sr*0.1))); ch_t += dur+0.1
        if ch_aud:
            aud_segs.append(np.concatenate(ch_aud))
            cur_t += ch_t
            if i < len(chunks)-1:
                aud_segs.append(np.zeros(int(sr*p_l))); cur_t += p_l

    if not aud_segs: return None, None, None, None
    f_aud = np.concatenate(aud_segs)
    
    # Safety: Handle NaNs or Infinity
    f_aud = np.nan_to_num(f_aud)
    
    mv = np.max(np.abs(f_aud))
    if mv > 1e-6: # Prevent division by zero or near-zero
        f_aud = (f_aud / mv * 32767).astype(np.int16)
    else:
        f_aud = f_aud.astype(np.int16)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    w_p = os.path.join(OUTPUT_FOLDER, f"audio_{ts}.wav")
    sf.write(w_p, f_aud, sr)
    f_p = w_p
    if use_mp3:
        m_p = w_p.replace(".wav", ".mp3")
        try:
            AudioSegment.from_wav(w_p).export(m_p, format="mp3", bitrate="192k")
            f_p = m_p
        except: pass
    s_p = w_p.replace(".wav", ".srt")
    with open(s_p, "w", encoding="utf-8") as f: f.writelines(srt_segs)
    z_p = w_p.replace(".wav", ".zip")
    with zipfile.ZipFile(z_p, 'w') as z:
        z.write(f_p, os.path.basename(f_p))
        z.write(s_p, os.path.basename(s_p))
    return (sr, f_aud), f_p, s_p, z_p

# --- UI LOGIC ---
def ui_lang(l):
    d = TRANSLATIONS[l]
    return [
        gr.update(value="# "+d["title"]), 
        gr.update(label=d["dialect"]),
        gr.update(label=d["upload_txt"]),
        gr.update(value=d["unlimited_text"]),
        gr.update(label=d["input_label"], placeholder=d["input_placeholder"]),
        gr.update(label=d["pauses_accordion"]),
        gr.update(label=d["comma_pause"]),
        gr.update(label=d["sentence_pause"]),
        gr.update(label=d["audio_settings"]),
        gr.update(label=d["speed"]),
        gr.update(label=d["pitch"]),
        gr.update(label=d["export_mp3"]),
        gr.update(value=d["generate_btn"]),
        gr.update(label=d["audio_preview"]),
        gr.update(label=d["audio_file"]),
        gr.update(label=d["subtitles"]),
        gr.update(label=d["zip_bundle"]),
        gr.update(value=t["clean_title"]), # wait t is not defined, fixing to d
        gr.update(value=d["clean_title"]),
        gr.update(value=d["clean_desc"]),
        gr.update(label=d["original_text"]),
        gr.update(value=d["clean_btn"]),
        gr.update(label=d["cleaned_text"]),
        gr.update(value=d["usage_tips"]),
        gr.update(value=d["tip_q"]),
        gr.update(value=d["tip_l"]),
        gr.update(value=d["tip_s"]),
        gr.update(value=d["tip_v"]),
        gr.update(value=d["footer"]),
        gr.update(label=d["studio_tab"]),
        gr.update(label=d["cleaner_tab"]),
        gr.update(label=d["about_tab"])
    ]

# Fixed typo in ui_lang (d vs t)
def ui_lang_fixed(l):
    d = TRANSLATIONS[l]
    return [
        gr.update(value="# "+d["title"]), 
        gr.update(label=d["dialect"]),
        gr.update(label=d["upload_txt"]),
        gr.update(value=d["unlimited_text"]),
        gr.update(label=d["input_label"], placeholder=d["input_placeholder"]),
        gr.update(label=d["pauses_accordion"]),
        gr.update(label=d["comma_pause"]),
        gr.update(label=d["sentence_pause"]),
        gr.update(label=d["audio_settings"]),
        gr.update(label=d["speed"]),
        gr.update(label=d["pitch"]),
        gr.update(label=d["export_mp3"]),
        gr.update(value=d["generate_btn"]),
        gr.update(label=d["audio_preview"]),
        gr.update(label=d["audio_file"]),
        gr.update(label=d["subtitles"]),
        gr.update(label=d["zip_bundle"]),
        gr.update(value=d["clean_title"]),
        gr.update(value=d["clean_desc"]),
        gr.update(label=d["original_text"]),
        gr.update(value=d["clean_btn"]),
        gr.update(label=d["cleaned_text"]),
        gr.update(value=d["usage_tips"]),
        gr.update(value=d["tip_q"]),
        gr.update(value=d["tip_l"]),
        gr.update(value=d["tip_s"]),
        gr.update(value=d["tip_v"]),
        gr.update(value=d["footer"]),
        gr.update(label=d["studio_tab"]),
        gr.update(label=d["cleaner_tab"]),
        gr.update(label=d["about_tab"])
    ]

theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
with gr.Blocks(title="Dolphin KURDISH TTS") as demo:
    with gr.Row():
        tit = gr.Markdown("# 🐬 Dolphin KURDISH TTS")
        ls = gr.Radio(["Kurdish", "English"], value="English", label="Language / زمان")
    
    with gr.Tabs() as ts:
        with gr.TabItem("🎛️ Studio", id=0) as t1:
            with gr.Row():
                with gr.Column():
                    dia = gr.Dropdown(list(MODELS.keys()), value="Sorani", label="Dialect")
                    upl = gr.File(label="📄 Upload .txt file", file_types=[".txt"], type="filepath")
                    lm = gr.Markdown("✅ Supports unlimited text length!")
                    txt = gr.Textbox(lines=10, label="Input Kurdish Text", rtl=True, placeholder="Enter text...")
                    with gr.Accordion("⏸️ Natural Pauses", open=False) as a1:
                        ps = gr.Slider(0.2, 0.8, value=0.4, label="Comma pause")
                        pl = gr.Slider(0.8, 2.0, value=1.3, label="Sentence pause")
                    with gr.Accordion("⚙️ Audio Settings", open=True) as a2:
                        sp = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                        pt = gr.Slider(-5, 5, value=0, step=1, label="Pitch")
                        mp3 = gr.Checkbox(label="Export as MP3", value=False)
                    btn = gr.Button("🔊 Generate Speech", variant="primary")
                with gr.Column():
                    a_p = gr.Audio(label="Audio Preview")
                    a_f = gr.File(label="Audio File")
                    s_f = gr.File(label="Subtitles (.srt)")
                    z_f = gr.File(label="📦 ZIP Bundle")
        
        with gr.TabItem("🧹 Text Cleaner", id=1) as t2:
            c1 = gr.Markdown("### Fix broken Kurdish characters"); c2 = gr.Markdown("Paste messy text...")
            raw = gr.Textbox(lines=6, label="Original Text", rtl=True)
            cbtn = gr.Button("Clean Text", variant="secondary")
            cout = gr.Textbox(lines=6, label="Cleaned Text", rtl=True)
            
        with gr.TabItem("ℹ️ About", id=2) as t3:
            gr.Markdown("# 🐬 Dolphin KURDISH TTS\nCreated by: Heldn Hastyar Abdullah")

    ut = gr.Markdown("### Usage Tips")
    m1 = gr.Markdown("- **Max quality**: Use punctuation"); m2 = gr.Markdown("- **Long texts**: Use .txt files")
    m3 = gr.Markdown("- **Sorani**: Auto-fix enabled"); m4 = gr.Markdown("- **Video**: Download ZIP")
    ft = gr.HTML("<div style='text-align: center; padding: 15px;'><strong>🐬 Dolphin KURDISH TTS</strong></div>")

    ls.change(ui_lang_fixed, [ls], [tit, dia, upl, lm, txt, a1, ps, pl, a2, sp, pt, mp3, btn, a_p, a_f, s_f, z_f, c1, c2, raw, cbtn, cout, ut, m1, m2, m3, m4, ft, t1, t2, t3])
    upl.change(lambda f: open(f.name, encoding='utf-8', errors='ignore').read() if f else "", [upl], [txt])
    btn.click(generate_audio_engine, [txt, dia, sp, pt, mp3, ps, pl], [a_p, a_f, s_f, z_f])
    cbtn.click(normalize_kurdish_text, [raw], [cout])

if __name__ == "__main__":
    print("Status: Ready! Launching browser...")
    try:
        import pyi_splash
        pyi_splash.close()
    except:
        pass
    demo.launch(inbrowser=True, theme=theme)