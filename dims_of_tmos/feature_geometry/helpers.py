import torch as t
from einops import einsum


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
    dim_fracs_SF = dim_fracs_SF.cpu()

    return dim_fracs_SF


def median_dimensionality(dim_fracs_SF: t.Tensor) -> t.Tensor:
    median_dim_frac_list_S: list[t.Tensor] = []

    for _instance_num, feature_dimensionality_F in enumerate(dim_fracs_SF):
        feature_dimensionality_list_F = feature_dimensionality_F.tolist()
        filtered_values = [
            feature_dim
            for feature_dim in feature_dimensionality_list_F
            if 0.41 <= feature_dim <= 0.48
        ]
        median_feature_dim = (
            t.median(t.tensor(filtered_values)) if len(filtered_values) > 0 else t.tensor(0)
        )
        median_dim_frac_list_S.append(median_feature_dim)

    median_dim_frac_S = t.stack(median_dim_frac_list_S)

    return median_dim_frac_S


def reorder_corr_matrix(corr_matrix_FF: t.Tensor) -> t.Tensor:
    # Get diagonal values
    diag_values_F = t.diagonal(corr_matrix_FF)

    # Get indices that would sort the diagonal values in descending order
    sorted_indices_F = t.argsort(diag_values_F, descending=True)

    # Reorder both rows and columns using these indices
    reordered_corr_matrix_FF = corr_matrix_FF[sorted_indices_F][:, sorted_indices_F]

    return reordered_corr_matrix_FF
