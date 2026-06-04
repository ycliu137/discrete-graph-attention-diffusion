from dgad.models.encoder import GraphFeatureEncoder
from dgad.models.decoder import InnerProductDecoder

__all__ = ["GraphFeatureEncoder", "InnerProductDecoder", "DGADModel"]


def __getattr__(name):
    if name == "DGADModel":
        from dgad.models.dgad_model import DGADModel
        return DGADModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
