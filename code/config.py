"""Shared constants. Every path comes from ROOT so a new Colab session can resume from Drive."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEXSI_ROOT", "/content/drive/MyDrive/lexsi_track3_multilingual_toolcall"))

MODEL = "CohereLabs/tiny-aya-global"
AGENTTUNE_COMMIT = "d4db4c12d52342227c4113a70183d0f64aa2f345"
AUDITKIT_COMMIT = "71316da345784ff220d5ed59337441a59ab31526"

SEED = 42
MAX_NEW_TOKENS = 128

LOCALES = ("en-US", "fr-FR", "es-ES", "zh-CN", "ja-JP", "ar-SA", "hi-IN", "id-ID", "vi-VN", "tr-TR")
