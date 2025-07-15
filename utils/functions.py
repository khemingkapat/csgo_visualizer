import numpy as np
import pandas as pd


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, degree=True) -> float:
    """
    Calculate angle between two vectors.

    Args:
        v1, v2 (np.array): Input vectors
        degree (bool): If True, return angle in degrees; if False, return in radians

    Returns:
        float: Angle in degrees (default) or radians
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if degree:
        return np.degrees(angle_rad)
    return angle_rad


def eu_dist(x1: int, y1: int, x2: int, y2: int):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


filter_nan = lambda l: [item for item in l if not pd.isna(item)]
