import plotly.graph_objects as go

from dims_of_tmos.feature_geometry.helpers import reorder_corr_matrix
from dims_of_tmos.feature_geometry.vis import get_block_diag_feature_cluster_plot
from dims_of_tmos.model import Model


def render_block_diag_feature_clusters(model: Model, chosen_trace_index: int) -> go.Figure:
    W_SFN = model.W

    chosen_instance_index = chosen_trace_index // 2
    # check for the traces in the "effective dim vs sparsity" graph above; use the indices that after you divide the trace by 2

    chosen_W_FN = W_SFN[chosen_instance_index].detach()

    nfm_FF = chosen_W_FN @ chosen_W_FN.T  # Gram Matrix
    nfm_FF = reorder_corr_matrix(nfm_FF)

    fig = get_block_diag_feature_cluster_plot(nfm_FF.cpu().numpy())
    return fig


if __name__ == "__main__":
    path = "test_model"
    model = Model.from_pretrained(path)
    fig = render_block_diag_feature_clusters(model, 0)
    fig.show()
