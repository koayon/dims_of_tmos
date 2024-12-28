import torch as t

from dims_of_tmos import DEVICE
from dims_of_tmos.model import Config, Model
from dims_of_tmos.train import optimize

LEFT_FEATURE_PROB = 9 / 20
RIGHT_FEATURE_PROB = 11 / 20


def train_feature_geom_models() -> Model:
    config = Config(
        n_features=400,
        n_hidden=30,
        n_instances=100,
    )

    model = Model(
        config=config,
        device=DEVICE,
        # For this experiment, use constant importance.
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        feature_probability=(
            20 ** -t.linspace(LEFT_FEATURE_PROB, RIGHT_FEATURE_PROB, config.n_instances)
        )[:, None],
    )

    model = optimize(
        model, steps=2_000, n_batch=2**13
    )  # ideally steps = 50k, batch size = 2^12
    # optimize(model, steps=50_000, n_batch=2**12)

    return model


if __name__ == "__main__":
    model = train_feature_geom_models()
    model.save("feature_geom_model")
