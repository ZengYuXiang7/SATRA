"""Generic time-series forecasting baselines included with SATRA."""

from importlib import import_module


MODEL_MODULES = {
    "LSTNet": "timeseries.models.LSTNet",
    "xLSTMMixer": "timeseries.models.xLSTMMixer",
    "ModernTCN": "timeseries.models.ModernTCN",
    "TimeMixer": "timeseries.models.TimeMixer",
    "WPMixer": "timeseries.models.WPMixer",
    "PatchTST": "timeseries.models.PatchTST",
    "iTransformer": "timeseries.models.iTransformer",
    "Crossformer": "timeseries.models.Crossformer",
    "xPatch": "timeseries.models.xPatch",
    "DUET": "timeseries.models.DUET",
}


def load_model_class(name):
    """Return the selected baseline's ``Model`` class without eager imports."""

    try:
        module_path = MODEL_MODULES[name]
    except KeyError as error:
        choices = ", ".join(MODEL_MODULES)
        raise ValueError(f"Unknown time-series baseline {name!r}; choose from {choices}") from error
    return import_module(module_path).Model


__all__ = ["MODEL_MODULES", "load_model_class"]
