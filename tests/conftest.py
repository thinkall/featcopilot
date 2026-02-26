"""Global test configuration."""

import pandas as pd

# Disable pandas StringDtype inference (default in pandas 3.x) so that
# string columns remain object dtype, which is compatible with
# np.issubdtype and other numpy dtype checks used in featcopilot.
try:
    pd.options.future.infer_string = False  # pandas 2.1-2.x
except Exception:
    pass
try:
    pd.options.mode.infer_string = False  # pandas 3.x
except Exception:
    pass
