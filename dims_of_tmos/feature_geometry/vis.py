import datetime

import numpy as np
import plotly.express as px
import torch as t
from plotly import graph_objects as go

from dims_of_tmos import BASE_DIR
from dims_of_tmos.feature_geometry.helpers import (
    compute_dimensionality,
    median_dimensionality,
)
from dims_of_tmos.model import Model


def plot_sparsity_vs_features_per_dim(model: Model) -> go.Figure:
    fig = px.line(
        x=1 / model.feature_probability[:, 0].cpu(),
        y=(
            model.config.num_neurons / (t.linalg.matrix_norm(model.W.detach(), "fro") ** 2)
        ).cpu(),
        log_x=True,
        markers=True,
    )
    fig.update_xaxes(title="1/(1-S)")
    fig.update_yaxes(title=f"m/(||W||_F)^2")

    return fig


def get_approx_feature_cluster_sizes_plot(model: Model) -> go.Figure:
    dim_fracs_SF = compute_dimensionality(model.W)

    median_dim_fracs_S = median_dimensionality(dim_fracs_SF)

    fig = go.Figure()

    density = model.feature_probability[:, 0].cpu()
    W_SFN = model.W.detach()

    for a, b in [(1, 2), (2, 3), (2, 5), (2, 6), (2, 7), (4, 9), (6, 13)]:
        val = a / b
        fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

    for a, b in [(1, 1), (3, 4), (3, 7), (3, 8), (3, 12), (3, 20), (5, 11), (5, 13), (4, 11)]:
        val = a / b
        fig.add_hline(
            val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05)
        )

    for i in range(len(W_SFN)):
        fracs_ = dim_fracs_SF[i]
        N = fracs_.shape[0]
        xs = 1 / density
        if i != len(W_SFN) - 1:
            dx = xs[i + 1] - xs[i]
        fig.add_trace(
            go.Scatter(
                x=1 / density[i] * np.ones(N) + dx * np.random.uniform(-0.1, 0.1, N),
                y=fracs_,
                marker=dict(
                    color="black",
                    size=1,
                    opacity=0.5,
                ),
                mode="markers",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=1 / density[i] * np.ones(N) + dx * np.random.uniform(-0.1, 0.1, N),
                y=[median_dim_fracs_S[i].item()],
                marker=dict(
                    color="blue",
                    size=10,
                ),
                mode="markers",
                marker_symbol="diamond-x",
            )
        )

    fig.update_xaxes(
        type="log",
        title="1/(1-S)",
        showgrid=False,
    )
    fig.update_yaxes(showgrid=False, title="hidden dimensions per embedded feature")

    fig.update_layout({"title": "Effective dimensionality per feature vs. sparsity"})
    fig.update_layout(showlegend=False)

    return fig


def get_block_diag_feature_cluster_plot(nfm_FF_np: np.ndarray) -> go.Figure:
    fig = px.imshow(nfm_FF_np, color_continuous_scale=px.colors.diverging.RdBu)
    return fig


if __name__ == "__main__":
    path = "test_model"
    model = Model.from_pretrained(path)

    fig = plot_sparsity_vs_features_per_dim(model)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    fig_file_path = BASE_DIR + f"sparsity_vs_features_per_dim_{current_date}.html"

    fig.write_html(fig_file_path)
