import os

import torch as t

if t.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

base_dir_path = "/mnt/ssd-1/mechinterp/koayon/dims_of_tmos/"
if os.path.exists(base_dir_path):
    BASE_DIR = base_dir_path
else:
    BASE_DIR = os.path.join(os.getcwd(), "figures")
    if not os.path.exists(base_dir_path):
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(BASE_DIR, exist_ok=True)
