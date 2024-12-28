import torch as t

if t.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
