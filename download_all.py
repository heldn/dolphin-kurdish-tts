
import os
import sys
import time
import shutil

# 1. SETUP DIRECTORIES (Must happen before imports that use HF_HOME)
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# FORCE CACHE DIRECTORY and ONLINE mode for this script
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# We want to download, so verify we aren't unknowingly offline
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]

from huggingface_hub import snapshot_download

MODELS = {
    "Sorani": "razhan/mms-tts-ckb",
    "Kurmanji (Arabic Script)": "facebook/mms-tts-kmr-script_arabic",
    "Kurmanji (Latin Script)": "facebook/mms-tts-kmr-script_latin",
    "Arabic (Habibi - Dialectal)": "SWivid/Habibi-TTS",
    # Kokoro is a bit special, often downloaded via library, but we can cache the repo
    "Multi-Language (Kokoro-82M)": "hexgrad/Kokoro-82M"
}

# Add internal dependencies that aren't in the main map but are used
HIDDEN_MODELS = [
    "charactr/vocos-mel-24khz", # Used by Habibi/Vocos
]

def robust_download(repo_id, max_retries=100):
    """
    Attempts to download a repository with indefinite retries and resume capability.
    This is the 'Final Fix' for unstable internet connections.
    """
    attempt = 1
    while True:
        try:
            print(f"   ⏳ Checking/Downloading: {repo_id} (Attempt {attempt})...")
            snapshot_download(
                repo_id=repo_id, 
                cache_dir=MODEL_CACHE_DIR,
                resume_download=True, # Explicitly request resume (default is usually True but let's be safe)
                local_files_only=False
            )
            print(f"   ✅ Success: {repo_id}")
            return True
        except KeyboardInterrupt:
            print("\n   🛑 User stopped the download.")
            sys.exit(0)
        except Exception as e:
            print(f"   ⚠️ Connection failed ({e}). Retrying in 5 seconds...")
            time.sleep(5)
            attempt += 1
            if attempt > max_retries:
                print(f"   ❌ Failed after {max_retries} attempts.")
                return False

def download_everything():
    print("============================================")
    print("   🐬 Dolphin TTS - Ultimate Model Downloader 🌍")
    print("============================================")
    print(f"Target Cache Directory: {MODEL_CACHE_DIR}")
    print("This script uses a ROBUST RETRY mechanism to fix download interruptions.")
    print("It will keep trying until your models are 100% complete.")
    print("--------------------------------------------")

    # 1. HuggingFace Models
    for name, repo_id in MODELS.items():
        print(f"\n📦 Processing Main Model: {name}")
        robust_download(repo_id)

    # 2. Hidden Dependencies
    for repo_id in HIDDEN_MODELS:
        print(f"\n📦 Processing Dependency: {repo_id}")
        robust_download(repo_id)

    # 3. Spacy Models (for Kokoro/English)
    print("\n📦 Checking Spacy 'en_core_web_sm'...")
    try:
        import spacy
        if not spacy.util.is_package("en_core_web_sm"):
            print("   Downloading en_core_web_sm...")
            from spacy.cli.download import download
            download("en_core_web_sm")
            print("   ✅ Success")
        else:
            print("   ✅ Already installed")
    except ImportError:
        print("   ⚠️ Spacy not installed? Skipping.")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    print("\n============================================")
    print("🎉 ALL DOWNLOADS COMPLETE! 🎉")
    print("Your app is now fully offline-ready.")
    print("You can proceed to run: python package_ready_to_go.py")
    print("============================================")

if __name__ == "__main__":
    download_everything()
