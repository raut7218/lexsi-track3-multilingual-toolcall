"""Shared constants. Every path comes from ROOT so a new Colab session can resume from Drive."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LEXSI_ROOT", "/content/drive/MyDrive/lexsi_track3_multilingual_toolcall"))

MODEL = "CohereLabs/tiny-aya-global"
AGENTTUNE_COMMIT = "d4db4c12d52342227c4113a70183d0f64aa2f345"
AUDITKIT_COMMIT = "71316da345784ff220d5ed59337441a59ab31526"

SEED = 42
MAX_CONTEXT = 8192  # Tiny Aya's max_position_embeddings
MAX_NEW_TOKENS = 128
# Set as tokenizer.eos_token everywhere. The shipped eos (id 6) is not what Tiny Aya emits after a
# response, so TRL never stops and its is_eos mask lets the trailing tokens into the GRPO loss.
STOP_TOKEN = "<|END_RESPONSE|>"

ARG_CUT = 0.02  # gives 211 arguments, the closest of the cutoffs tried to the paper's 200
N_TEST, N_TEST_NO_CALL = 588, 88
N_DEV, N_DEV_NO_CALL = 200, 30
N_SFT, N_GRPO = 3000, 1000
N_CATALOGUE = 100  # call rows per language, every function in every prompt
N_ENGLISH_CALL, N_ENGLISH_NO_CALL = 85, 15  # keeps the test set's no-call share
NO_CALL_RATE, ENGLISH_RATE = 0.15, 0.25

LOCALES = ("en-US", "fr-FR", "es-ES", "zh-CN", "ja-JP", "ar-SA", "hi-IN", "id-ID", "vi-VN", "tr-TR")
# Outside our training, but on Tiny Aya's language list: new to AgentTune, not to the base model.
HELD_OUT = ("sw-KE", "am-ET", "te-IN", "km-KH")
