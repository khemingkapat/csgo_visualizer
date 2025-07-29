import pandas as pd


def index_dfs(
    dfs_dict: dict[str, pd.DataFrame],
    include: list[str] | None = None,
    index_by: list[str] | None = None,
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
    if index_by is None:
        index_by = ["match_id", "round_num"]

    indexed_dfs = {}

    for df_name, df in dfs_dict.items():
        # print(f"{df_name} => {df.columns}")
        if df_name in include:
            indexed_dfs[df_name] = df.set_index(index_by)
            # print(f"Indexed DataFrame '{df_name}' by {index_by}.")
        else:
            indexed_dfs[df_name] = df.copy()

    return indexed_dfs
