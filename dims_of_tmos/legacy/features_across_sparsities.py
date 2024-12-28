# # Visualizing features across varying sparsity

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch as t
from plotly.subplots import make_subplots

from dims_of_tmos import DEVICE
from dims_of_tmos.model import Config, Model
from dims_of_tmos.train import optimize


def render_features(model: Model, which=np.s_[:]) -> go.Figure:
    cfg = model.config
    W = model.W.detach()
    W_norm = W / (1e-5 + t.linalg.norm(W, 2, dim=-1, keepdim=True))

    interference = t.einsum("ifh,igh->ifg", W_norm, W)
    interference[:, t.arange(cfg.n_features), t.arange(cfg.n_features)] = 0

    polysemanticity = t.linalg.norm(interference, dim=-1).cpu()
    net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
    norms = t.linalg.norm(W, 2, dim=-1).cpu()

    WtW = t.einsum("sih,soh->sio", W, W).cpu()

    # width = weights[0].cpu()
    # x = t.cumsum(width+0.1, 0) - width[0]
    x = t.arange(cfg.n_features)
    width = 0.9

    which_instances = np.arange(cfg.n_instances)[which]
    fig = make_subplots(
        rows=len(which_instances),
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.1,
    )
    for row, inst in enumerate(which_instances):
        fig.add_trace(
            go.Bar(
                x=x,
                y=norms[inst],
                marker=dict(color=polysemanticity[inst], cmin=0, cmax=1),
                width=width,
            ),
            row=1 + row,
            col=1,
        )
        data = WtW[inst].numpy()
        fig.add_trace(
            go.Image(
                z=plt.cm.coolwarm((1 + data) / 2, bytes=True),  # type: ignore
                colormodel="rgba256",
                customdata=data,
                hovertemplate="""\
In: %{x}<br>
Out: %{y}<br>
Weight: %{customdata:0.2f}
""",
            ),
            row=1 + row,
            col=2,
        )

    fig.add_vline(
        x=(x[cfg.n_hidden - 1] + x[cfg.n_hidden]) / 2,
        line=dict(width=0.5),
        col=1,  # type: ignore
    )

    # fig.update_traces(marker_size=1)
    fig.update_layout(
        showlegend=False, width=600, height=100 * len(which_instances), margin=dict(t=0, b=0)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


if __name__ == "__main__":
    config = Config(
        n_features=100,
        n_hidden=20,
        n_instances=20,
    )

    model = Model(
        config=config,
        device=DEVICE,
        # Exponential feature importance curve from 1 to 1/100
        importance=(100 ** -t.linspace(0, 1, config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        feature_probability=(20 ** -t.linspace(0, 1, config.n_instances))[:, None],
    )
    optimize(model)

    fig = render_features(model, np.s_[::2])
    fig.update_layout()
