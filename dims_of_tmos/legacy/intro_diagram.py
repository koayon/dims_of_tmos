# ## Introduction Figure from TMOS paper

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from dims_of_tmos import DEVICE
from dims_of_tmos.model import Config, Model
from dims_of_tmos.train import optimize


def plot_intro_diagram(model: Model):
    from matplotlib import collections as mc
    from matplotlib import colors as mcolors

    config = model.config
    WA = model.W.detach()
    # N = len(WA[:, 0])
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels
    plt.rcParams["axes.prop_cycle"] = plt.cycler(  # type: ignore
        "color", plt.cm.viridis(model.importance[0].cpu().numpy())  # type: ignore
    )
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2))
    for i, ax in zip(sel, axs):
        W = WA[i].cpu().detach().numpy()
        colors = [
            mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ]
        ax.scatter(W[:, 0], W[:, 1], c=colors[0 : len(W[:, 0])])
        ax.set_aspect("equal")
        ax.add_collection(
            mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=colors)  # type: ignore
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")
    plt.show()


if __name__ == "__main__":
    config = Config(
        n_features=5,
        n_hidden=2,
        n_instances=10,
    )

    model = Model(
        config=config,
        device=DEVICE,
        # Exponential feature importance curve from 1 to 1/100
        importance=(0.9 ** t.arange(config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        feature_probability=(20 ** -t.linspace(0, 1, config.n_instances))[:, None],
    )

    optimize(model)

    plot_intro_diagram(model)
