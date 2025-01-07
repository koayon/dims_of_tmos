import torch as t

from dims_of_tmos import DEVICE
from dims_of_tmos.model import Config, Model
from dims_of_tmos.train import optimize

LEFT_FEATURE_PROB = 9 / 20
RIGHT_FEATURE_PROB = 11 / 20


def train_feature_geom_models() -> Model:
    config = Config(
        num_features=num_features,
        num_neurons=num_neurons,
        num_instances=num_instances,
    )

    model = Model(
        config=config,
        device=DEVICE,
        # For this experiment, use constant importance.
        # Sweep feature frequency across the instances
        feature_probability=(
            20 ** -t.linspace(LEFT_FEATURE_PROB, RIGHT_FEATURE_PROB, config.num_instances)
        )[:, None],
    )

    model = optimize(
        model,
        steps=train_steps,
        batch_size=batch_size,
    )

    return model


if __name__ == "__main__":
    model = train_feature_geom_models()
    model.save("feature_geom_model")
