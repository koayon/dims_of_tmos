import time

import einops
import numpy as np
import torch as t
from torch.optim import AdamW
from tqdm.notebook import trange

from dims_of_tmos.model import Model


def constant_lr(*_):
    return 1.0


def optimize(
    model: Model,
    render=False,
    batch_size=1024,
    steps=10_000,
    print_freq=100,
    lr=1e-3,
    lr_scale=constant_lr,
    hooks=[],
) -> Model:
    cfg = model.config

    opt = AdamW(list(model.parameters()), lr=lr)

    start = time.time()
    with trange(steps) as t_steps:
        for step in t_steps:
            step_lr = lr * lr_scale(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(batch_size)
            out: t.Tensor = model(batch)
            error = model.importance * (batch.abs() - out) ** 2
            loss = einops.reduce(error, "b i f -> i", "mean").sum()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                t_steps.set_postfix(
                    loss=loss.item() / cfg.num_instances,
                    lr=step_lr,
                )

    return model


def linear_lr(current_step: int, total_steps: int) -> float:
    return 1 - (current_step / total_steps)


def cosine_decay_lr(current_step: int, total_steps: int) -> float:
    return np.cos(0.5 * np.pi * current_step / (total_steps - 1))
