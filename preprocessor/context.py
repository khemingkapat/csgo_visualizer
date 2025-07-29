from urllib.parse import non_hierarchical
import pandas as pd


def preprocess_context_data(
    dfs_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    processed_dfs = {name: df.copy() for name, df in dfs_dict.items()}
    if "damages" in processed_dfs:
        damages_df = processed_dfs["damages"]
        damages_df.loc[:, "total_damage"] = (
            damages_df["dmg_health"] + damages_df["dmg_armor"]
        )
        damages_df.loc[:, "total_damage_taken"] = (
            damages_df["dmg_health_real"] + damages_df["dmg_armor"]
        )
        damages_df["attacker_steamid"] = damages_df["attacker_steamid"].astype("Int64")
        damages_df.loc[:, "attacker_steamid"] = damages_df["attacker_steamid"].astype(
            "Int64"
        )
        damages_df["victim_steamid"] = damages_df["victim_steamid"].astype("Int64")
        damages_df.loc[:, "victim_steamid"] = damages_df["victim_steamid"].astype(
            "Int64"
        )
        processed_dfs["damages"] = damages_df
    if "grenades" in processed_dfs:
        grenades_df = processed_dfs["grenades"]
        index_names = grenades_df.index.names
        grenades_df.dropna(subset=["x", "y", "z"], inplace=True)
        grenades_df = (
            grenades_df.reset_index()
            .groupby("entity_id", as_index=False)
            .agg(
                **{name: (name, "first") for name in index_names if name is not None},
                thrower_steamid=("thrower_steamid", "first"),
                thrower=("thrower", "first"),
                grenade_type=("grenade_type", "first"),
                throw_tick=("tick", "first"),
                thrower_x=("x", "first"),
                thrower_y=("y", "first"),
                thrower_z=("z", "first"),
                destroy_tick=("tick", "last"),
                grenade_x=("x", "last"),
                grenade_y=("y", "last"),
                grenade_z=("z", "last"),
            )
        )
        valid_index_names = [
            name
            for name in index_names
            if name is not None and name in grenades_df.columns
        ]

        if valid_index_names:
            grenades_df = grenades_df.set_index(valid_index_names)
        mapping = (
            processed_dfs["ticks"]
            .reset_index()
            .groupby(["round_num", "steamid"])
            .agg(side=("side", "first"))
        ).to_dict()["side"]

        grenades_df["thrower_side"] = pd.Series(
            zip(grenades_df.index, grenades_df["thrower_steamid"])
        ).map(mapping)

        grenades_df.loc[:, "grenade_side"] = grenades_df["thrower_side"]
        processed_dfs["grenades"] = grenades_df
    if "other_data" in processed_dfs:
        tour_name = str(processed_dfs["other_data"]["server_name"][0]).removesuffix(
            " Server"
        )
        processed_dfs["other_data"]["tournament_name"] = tour_name
        if "rounds" in processed_dfs:
            processed_dfs["rounds"]["n_ticks"] = processed_dfs["rounds"][
                "official_end"
            ].sub(processed_dfs["rounds"]["start"])
    return processed_dfs
