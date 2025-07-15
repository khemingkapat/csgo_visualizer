from utils.symbols import image_dim
import re
import pandas as pd


def transform_coord(
    map_data: dict, player_loc: pd.DataFrame, x_col: str = "x", y_col: str = "y"
) -> pd.DataFrame:
    result = player_loc.copy()

    pos_x = map_data["pos_x"]
    pos_y = map_data["pos_y"]
    scale = map_data["scale"]

    result[x_col] = (result[x_col] - pos_x) / scale
    result[y_col] = image_dim - (pos_y - result[y_col]) / scale

    return result


def transform_coords(
    map_data: dict, player_loc: pd.DataFrame, status: list[str]
) -> pd.DataFrame:
    tf = player_loc.copy()
    for st in status:
        tf = transform_coord(map_data, tf, x_col=f"{st}_x", y_col=f"{st}_y")
    return tf


def convert_to_snake_case(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    # Function to convert a string to snake_case
    def to_snake_case(text, prefix):
        # Convert to lowercase first
        s = text.lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^\w_]", "", s)
        return f"{prefix}{s}"

    new_df = df.copy()
    new_df.index = [to_snake_case(idx, prefix) for idx in df.index]

    return new_df
