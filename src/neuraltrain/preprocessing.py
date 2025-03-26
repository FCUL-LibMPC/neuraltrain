from sklearn.base import BaseEstimator, TransformerMixin

__original_temp_range = (0, 40)
__normal_range = (0, 1)
__original_radiation_range = (0, 1000)


def normalize_temp(
    temp: float,
    original_range: tuple = __original_temp_range,
    new_range: tuple = __normal_range,
) -> float:
    """Normalize temperature values to a new range. Default behavior is from 0ºC - 40ºC to 0 - 1.

    Args:
        temp (float): Temperature value to normalize
        original_range (tuple, optional): Range for the original values in degrees celsius. Defaults to (0, 40).
        new_range (tuple, optional): Range for the new values. Defaults to (0, 1).

    Returns:
        float: Normalized values in the new range.
    """
    min_temp, max_temp = original_range
    min_val, max_val = new_range
    return (temp - min_temp) / (max_temp - min_temp) * (max_val - min_val) + min_val


def denormalize_temp(
    temp,
    new_range: tuple = __normal_range,
    original_range: tuple = __original_temp_range,
):
    """Denormalize temperature values to a new range. Default behavior is from 0 - 1 to 0ºC - 40ºC.

    Args:
        temp (float | Tensor): Normalized temperature value to denormalize
        new_range (tuple, optional): Range for the new values. Defaults to (0, 1).
        original_range (tuple, optional): Range for the original values in degrees celsius. Defaults to (0, 40).

    Returns:
        float | Tensor: Denormalized values in the original range.
    """
    min_temp, max_temp = original_range
    min_val, max_val = new_range
    return (temp - min_val) / (max_val - min_val) * (max_temp - min_temp) + min_temp


def normalize_radiation(
    radiation: float,
    original_range: tuple = __original_radiation_range,
    new_range: tuple = __normal_range,
) -> float:
    """Normalize direct solar radiation values to a new range. Default behavior is from 0 - 1000 W/m² to 0 - 1.

    Args:
        radiation (float): Radiation value to normalize
        original_range (tuple, optional): Range for the original values in W/m². Defaults to (0, 1000).
        new_range (tuple, optional): Range for the new values. Defaults to (0, 1).

    Returns:
        float: Normalized values in the new range.
    """
    min_rad, max_rad = original_range
    min_val, max_val = new_range
    return (radiation - min_rad) / (max_rad - min_rad) * (max_val - min_val) + min_val


def denormalize_radiation(
    radiation,
    new_range: tuple = __normal_range,
    original_range: tuple = __original_radiation_range,
):
    """Denormalize direct solar radiation values to a new range. Default behavior is from 0 - 1 to 0 - 1000 W/m².

    Args:
        radiation (float | Tensor): Normalized radiation value to denormalize
        new_range (tuple, optional): Range for the new values. Defaults to (0, 1).
        original_range (tuple, optional): Range for the original values in W/m². Defaults to (0, 1000).

    Returns:
        float | Tensor: Denormalized values in the original range.
    """
    min_rad, max_rad = original_range
    min_val, max_val = new_range
    return (radiation - min_val) / (max_val - min_val) * (max_rad - min_rad) + min_rad


def denormalize_mse(normalized_mse, original_range: tuple) -> float:
    """
    Convert MSE from normalized [0, 1] scale back to the original Celsius^2 scale.

    Args:
        normalized_mse (float | Tensor): The MSE in the normalized space (between 0 and 1).
        original_range (tuple): Tuple containing the minimum and maximum values used for normalization.

    Returns:
        float | Tensor: MSE in the original Celsius^2 scale.
    """
    min_val, max_val = original_range
    value_range = max_val - min_val
    return normalized_mse * (value_range**2)


class NormalizeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df["Tz"] = df["Tz"].apply(normalize_temp)
        df["Tz_1"] = df["Tz_1"].apply(normalize_temp)
        df["Tout"] = df["Tout"].apply(normalize_temp)
        df["Thvac"] = df["Thvac"].apply(normalize_temp)
        df["Qsolar"] = df["Qsolar"].apply(normalize_radiation)
        return df
