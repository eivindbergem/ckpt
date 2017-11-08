import os.path

from .misc import load_json, get_ckpt_path

try:
    ckpt_config = load_json(os.path.join(get_ckpt_path(), "config"))
except FileNotFoundError:
    ckpt_config = {}
