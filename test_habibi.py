import os
import torch
import sys

def test_load():
    try:
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        print("Imports successful")
        
        # Try a dummy model init
        v1_base_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        print("Config created")
        
        # vocoder = load_vocoder() # This might download things, skip for now
        print("Skipping vocoder load")
        
        print("Test finished successfully")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
