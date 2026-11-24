
"""
Shared preprocessing helpers for the water potability thesis.

These functions are used inside sklearn pipelines.
Keep this module available when loading saved joblib models.
"""

import numpy as np
import pandas as pd


def fix_invalid_ph(X):
    """
    Replace physically invalid pH values (<= 0 or > 14) with NaN.

    This function is stateless:
    it does not calculate or learn anything from the dataset.

    Median imputation is performed in the next pipeline step,
    fitted only on the relevant training data.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "fix_invalid_ph expects a pandas DataFrame."
        )

    X = X.copy()

    if "ph" in X.columns:

        invalid_ph = (X["ph"] <= 0) | (X["ph"] > 14)

        X.loc[invalid_ph, "ph"] = np.nan

    return X
