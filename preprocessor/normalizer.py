import pandas as pd


def normalize_tick(
    dfs_dict: dict[str, pd.DataFrame], include: list[str] | None = None
) -> dict[str, pd.DataFrame]:

    if include is None:
        include = [
            "bomb",
            "damages",
            "footsteps",
            "grenades",
            "infernos",
            "kills",
            "rounds",
            "shots",
            "smokes",
            "ticks",
        ]

    processed_dfs = {name: df.copy() for name, df in dfs_dict.items()}

    for df_name in include:
        df = processed_dfs[df_name]
        tick_cols = [col for col in df if "tick" in col]
        if len(tick_cols) < 1:
            continue
        start_tick = dfs_dict["rounds"]["start"].reindex(df.index)

        df.loc[:, tick_cols] = df.loc[:, tick_cols].sub(start_tick, axis=0)
        processed_dfs[df_name] = df

    return processed_dfs


def normalize_steamid_columns(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    processed_dfs = {}

    for name, df in dfs.items():
        df_copy = df.copy()
        steamid_columns = [col for col in df_copy.columns if "steamid" in col.lower()]

        for col in steamid_columns:
            if df_copy[col].dtype == "float64":
                df_copy[col] = df_copy[col].astype("Int64").astype(str)
            else:
                df_copy[col] = df_copy[col].astype(str)

            df_copy[col] = df_copy[col].replace(["nan", "<NA>"], pd.NA)

        processed_dfs[name] = df_copy

    return processed_dfs
