import pandas as pd


def get_player_stat(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    player_stats = dfs["stats"].loc[
        "all", [col for col in dfs["stats"].columns if col not in ["n_rounds"]]
    ]

    kills_df = dfs["kills"]

    # Your existing result_df + add names
    kda_df = (
        pd.concat(
            [
                kills_df["attacker_name"].astype(str).value_counts().rename("kills"),
                kills_df["victim_name"].astype(str).value_counts().rename("deaths"),
                kills_df["assister_name"]
                .dropna()
                .astype(str)
                .value_counts()
                .rename("assists"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "name"})
    )
    stats_df = pd.merge(player_stats, kda_df, on="name", how="inner")
    stats_df = stats_df[
        [
            "name",
            "rating",
            "kills",
            "deaths",
            "assists",
            "kast",
            "dmg",
            "adr",
            "impact",
        ]
    ]
    stats_df = stats_df.sort_values(by=["rating"], ascending=False)

    return stats_df
