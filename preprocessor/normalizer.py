import pandas as pd


def normalize_tick(
    dfs_dict: dict[str, pd.DataFrame], include: list[str] | None = None
) -> dict[str, pd.DataFrame]:

    if include is None:
        include = [
            "player_frames",
            "flashes",
            "smokes",
            "damages",
            "kills",
            "frames",
            "grenades",
            "weapon_fires",
        ]

    processed_dfs = {name: df.copy() for name, df in dfs_dict.items()}

    for df_name in include:
        df = processed_dfs[df_name]
        tick_cols = [col for col in df if "tick" in col]
        if len(tick_cols) < 1:
            continue
        start_tick = dfs_dict["rounds"]["start_tick"].reindex(df.index)

        df.loc[:, tick_cols] = df.loc[:, tick_cols].sub(start_tick, axis=0)
        processed_dfs[df_name] = df

    return processed_dfs
