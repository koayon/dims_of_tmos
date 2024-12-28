import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from dims_of_tmos import BASE_DIR

# model.W.shape
# CHOSEN_INSTANCE_INDEX = 39

# can put whichever index here in the brackets?
# the index here might not actually be the weight vectors for model #idx...


# # %%
# pca = PCA()
# pca.fit(W_SFN[GOOD_IDX].T.cpu().numpy())

# # %%
# pca_matrix_NF = pca.components_
# # fig2 = px.imshow(np.abs(pca_matrix_NF[:]))
# # fig2.show()

# # Create a collection of heatmaps for each 20 features
# fig = make_subplots(rows=5, cols=4, subplot_titles=[f"Feature {i}" for i in range(20)])

# for i in range(20):
#     fig.add_trace(
#         go.Heatmap(
#             z=np.abs(pca_matrix_NF[:, i * 20 : (i + 1) * 20]),
#             colorscale="Viridis",
#         ),
#         row=i // 4 + 1,
#         col=i % 4 + 1,
#     )

# fig.update_layout(height=1000, width=1000, title_text="PCA components for each 20 features")
# fig.show()

# # %%
# fig.write_html(BASE_DIR + "separated_pca_matrix.html")

threshold = 0.05

# pca_matrix_NF_tensor = t.tensor(pca_matrix_NF)
# num_neurons_per_feature_F = t.sum((t.abs(pca_matrix_NF_tensor) > threshold).float(), dim=0)

# # %%
# for i in range(20):
#     print(i, t.sum((num_neurons_per_feature_F == i).float()).item())

# # %%
# fig2.write_html(BASE_DIR + "constrained_pca_matrix.html")

# # %%
# pca2 = PCA()
# pca2.fit(W_SFN[46].cpu().numpy())

# # %%
# pca_matrix2 = pca2.components_
# fig3 = px.imshow(pca_matrix2)
# fig3.show()
