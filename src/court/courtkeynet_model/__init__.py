from .model import CourtKeyNet

COURTKEYNET_MODEL_CONFIG = {
    "model": {
        "ofe": {"channels_per_band": 64, "stem_channels": 64},
        "pta": {"enabled": True, "radial_bins": 64, "angular_bins": 128},
        "num_keypoints": 4,
        "feature_dim": 128,
        "transformer": {
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
        "qcm": {"hidden_dims": [64, 128], "output_dim": 128},
    }
}

__all__ = ["CourtKeyNet", "COURTKEYNET_MODEL_CONFIG"]
