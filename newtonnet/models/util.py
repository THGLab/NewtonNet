from newtonnet.models.output import (
    EnergyOutput,
    GradientForceOutput,
    DirectForceOutput,
    HessianOutput,
    VirialOutput,
    SumAggregator,
    NullAggregator,
)
from newtonnet.models.edge_output import EdgeOutput


def get_output_by_string(key, n_features=None, activation=None, representations=None):
    if key == "energy":
        output_layer = EnergyOutput(n_features, activation)
    elif key == "gradient_force":
        output_layer = GradientForceOutput()
    elif key == "direct_force":
        output_layer = DirectForceOutput(n_features, activation)
    elif key == "hessian":
        output_layer = HessianOutput()
    elif key == "virial":
        output_layer = VirialOutput()
    elif key == "bonds":
        output_layer = EdgeOutput(n_features, activation, representations)
    else:
        raise NotImplementedError(f"Output type {key} is not implemented yet")
    return output_layer


def get_aggregator_by_string(key):
    if key == "energy":
        aggregator = SumAggregator()
    elif key == "gradient_force":
        aggregator = NullAggregator()
    elif key == "direct_force":
        aggregator = NullAggregator()
    elif key == "hessian":
        aggregator = NullAggregator()
    elif key == "virial":
        aggregator = NullAggregator()
    elif key == "bonds":
        aggregator = NullAggregator()
    else:
        raise NotImplementedError(f"Aggregate type {key} is not implemented yet")
    return aggregator
