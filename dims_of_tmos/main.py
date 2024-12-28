# %% [markdown]
# # Toy Models of Superposition
#
# This notebook includes the toy model training framework used to generate most of the results in the "Toy Models of Superposition" paper.
#
# The main useful improvement over a basic Pyt tiny autoencoder is the ability to batch train many models with varying sparsity at once, which is much more efficient than training them one at a time.
#
# This notebook is designed to run in Google Colab's Python 3.7 environment.

# %%
# !pip install einops

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from einops import einsum
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from dims_of_tmos import DEVICE
from dims_of_tmos.model import Config, Model
from dims_of_tmos.train import optimize

# %%

# %% [markdown]
# # Visualizing features across varying sparsity

# %%
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

# %%
optimize(model)


# %%
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
                z=plt.cm.coolwarm((1 + data) / 2, bytes=True),
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
        col=1,
    )

    # fig.update_traces(marker_size=1)
    fig.update_layout(
        showlegend=False, width=600, height=100 * len(which_instances), margin=dict(t=0, b=0)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# %%
fig = render_features(model, np.s_[::2])
fig.update_layout()

# %% [markdown]
# # Feature geometry

# %%
config = Config(
    n_features=400,
    n_hidden=30,
    n_instances=100,
)

left_val = 9 / 20
right_val = 11 / 20

model = Model(
    config=config,
    device=DEVICE,
    # For this experiment, use constant importance.
    # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    feature_probability=(20 ** -t.linspace(left_val, right_val, config.n_instances))[:, None],
)

# %%
optimize(model, steps=2_000, n_batch=2**13)  # ideally steps = 50k, batch size = 2^12
# optimize(model, steps=50_000, n_batch=2**12)

# %%
# left_val = math.log(3 / 4)
# right_val = math.log(4 / 5)


# feature_probability = (20 ** -t.linspace(left_val, right_val, config.n_instances))[:, None]
# feature_probability
# 1 / (1 - feature_probability)

# %%
fig = px.line(
    x=1 / model.feature_probability[:, 0].cpu(),
    y=(model.config.n_hidden / (t.linalg.matrix_norm(model.W.detach(), "fro") ** 2)).cpu(),
    log_x=True,
    markers=True,
)
fig.update_xaxes(title="1/(1-S)")
fig.update_yaxes(title=f"m/(||W||_F)^2")

# %%


# %%
@t.no_grad()
def compute_dimensionality(W_SFN: t.Tensor) -> t.Tensor:
    norms_SF = t.linalg.norm(W_SFN, 2, dim=-1)
    W_unit_SFN = W_SFN / t.clamp(norms_SF[:, :, None], 1e-6, float("inf"))

    interferences_SF = (
        einsum(
            W_unit_SFN,
            W_SFN,
            "sparsity num_features1 num_neurons, sparsity num_features2 num_neurons -> sparsity num_features1 num_features2",
        )
        ** 2
    ).sum(-1)

    dim_fracs_SF = norms_SF**2 / interferences_SF
    return dim_fracs_SF.cpu()


# %%
def median_dimensionality(dim_fracs_SF: t.Tensor) -> t.Tensor:
    median_dim_frac: list[t.Tensor] = []
    for features in dim_fracs_SF:
        list_features = features.tolist()
        filtered_values = [features for features in list_features if 0.41 <= features <= 0.48]
        median_value = (
            t.median(t.tensor(filtered_values)) if len(filtered_values) > 0 else t.tensor(0)
        )
        median_dim_frac.append(median_value)

    median_dim_frac_tensor = t.stack(median_dim_frac)

    return median_dim_frac_tensor


# %%
dim_fracs_SF = compute_dimensionality(model.W)

median_dim_fracs_S = median_dimensionality(dim_fracs_SF)

# %%
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

    # fig.add_trace(
    #     go.Scatter(
    #         x=1 / density[i] * np.ones(N) + dx * np.random.uniform(-0.1, 0.1, N),
    #         y=[median_dim_fracs_S[i].item()],
    #         marker=dict(
    #             color="blue",
    #             size=10,
    #         ),
    #         mode="markers",
    #         marker_symbol="diamond-x",
    #     )
    # )

fig.update_xaxes(
    type="log",
    title="1/(1-S)",
    showgrid=False,
)
fig.update_yaxes(showgrid=False, title="hidden dimensions per embedded feature")
fig.update_layout({"title": "Effective dimensionality per feature vs. sparsity"})
fig.update_layout(showlegend=False)

# %%
base_dir = "/mnt/ssd-1/mechinterp/koayon/ml-implementations/superposition/figures/"
fig.write_html(base_dir + "constrained_dimensionality_clusters1.html")

# %%
model.W.shape
GOOD_IDX = 39
# check for the traces in the "effective dim vs sparsity" graph above; use the indices that after you divide the trace by 2

W_SFN[GOOD_IDX]
# can put whichever index here in the brackets?
# the index here might not actually be the weight vectors for model #idx...

# %%


# %%
pca = PCA()
pca.fit(W_SFN[GOOD_IDX].T.cpu().numpy())

# %%
pca_matrix_NF = pca.components_
# fig2 = px.imshow(np.abs(pca_matrix_NF[:]))
# fig2.show()

# Create a collection of heatmaps for each 20 features
fig = make_subplots(rows=5, cols=4, subplot_titles=[f"Feature {i}" for i in range(20)])

for i in range(20):
    fig.add_trace(
        go.Heatmap(
            z=np.abs(pca_matrix_NF[:, i * 20 : (i + 1) * 20]),
            colorscale="Viridis",
        ),
        row=i // 4 + 1,
        col=i % 4 + 1,
    )

fig.update_layout(height=1000, width=1000, title_text="PCA components for each 20 features")
fig.show()

# %%
fig.write_html(base_dir + "separated_pca_matrix.html")


# %%
def reorder_corr_matrix(corr_matrix: t.Tensor):
    # Get diagonal values
    diag_values = t.diagonal(corr_matrix)

    # Get indices that would sort the diagonal values in descending order
    sorted_indices = t.argsort(diag_values, descending=True)

    # Reorder both rows and columns using these indices
    reordered_matrix = corr_matrix[sorted_indices][:, sorted_indices]

    return reordered_matrix


W_GOOD_FN = W_SFN[GOOD_IDX].detach()
nfm_FF = W_GOOD_FN @ W_GOOD_FN.T
undiag_nfm_FF = nfm_FF  # - t.diag(t.diag(nfm_FF))
print(t.max(t.abs(undiag_nfm_FF)))
diagf = nfm_FF.diag()
undiag_nfm_FF = reorder_corr_matrix(undiag_nfm_FF)
px.imshow(undiag_nfm_FF.cpu().numpy(), color_continuous_scale=px.colors.diverging.RdBu)


# %%
threshold = 0.05

# %%
pca_matrix_NF_tensor = t.tensor(pca_matrix_NF)
num_neurons_per_feature_F = t.sum((t.abs(pca_matrix_NF_tensor) > threshold).float(), dim=0)

# %%
for i in range(20):
    print(i, t.sum((num_neurons_per_feature_F == i).float()).item())

# %%
fig2.write_html(base_dir + "constrained_pca_matrix.html")

# %%
pca2 = PCA()
pca2.fit(W_SFN[46].cpu().numpy())

# %%
pca_matrix2 = pca2.components_
fig3 = px.imshow(pca_matrix2)
fig3.show()

# %%
# %%
