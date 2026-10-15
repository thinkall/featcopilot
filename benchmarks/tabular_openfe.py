"""Optional, spawn-pickleable compatibility adapter for OpenFE 0.0.12."""

import numpy as np
from openfe import OpenFE
from sklearn.metrics import mean_squared_error


class CompatibleOpenFE(OpenFE):
    """Preserve OpenFE's RMSE metric on sklearn versions without ``squared=False``."""

    def get_init_metric(self, pred, label):
        """Compute the original OpenFE metric using the current sklearn API."""
        if self.metric == "rmse":
            return float(np.sqrt(mean_squared_error(label, pred)))
        return super().get_init_metric(pred, label)
