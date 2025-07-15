import pandas as pd


def add_action_connection(
    target_df: pd.DataFrame,
    check_df: pd.DataFrame,
    target_combo_key: list[str],
    check_combo_key: list[str],
    connect_prefix="",
) -> pd.DataFrame:
    """
    Add connection column to target_df based on combo key matching with check_df.
    Vectorized version without index loops.
    """
    tdf = target_df.copy()
    cdf = check_df.copy()

    # Create combo keys for both DataFrames
    tdf_combo = _create_combo_key(tdf, target_combo_key)
    cdf_combo = _create_combo_key(cdf, check_combo_key)

    # Create the connection column
    target_col = f"{connect_prefix}_connected"
    tdf[target_col] = tdf_combo.isin(cdf_combo)

    return tdf


def _create_combo_key(df: pd.DataFrame, combo_keys: list[str]) -> pd.Series:
    """
    Create combo key series from specified columns with proper type conversion.
    """
    df_work = df.copy()

    # Convert columns to proper types
    for key in combo_keys:
        if key.endswith("id"):
            df_work[key] = df_work[key].astype("float64")
        df_work[key] = df_work[key].astype(str)

    # Create combo key by concatenating all columns
    if len(combo_keys) == 1:
        return df_work.loc[:, combo_keys[0]]
    else:
        return df_work[combo_keys[0]].str.cat(
            [df_work[key] for key in combo_keys[1:]], sep="_"
        )
