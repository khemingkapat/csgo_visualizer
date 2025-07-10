import pandas as pd


def preprocess_context_data(
    dfs_dict: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    processed_dfs = {name: df.copy() for name, df in dfs_dict.items()}

    if "damages" in processed_dfs:
        damages_df = processed_dfs["damages"]

        damages_df.loc[:, "total_damage"] = (
            damages_df["hp_damage"] + damages_df["armor_damage"]
        )

        damages_df.loc[:, "total_damage_taken"] = (
            damages_df["hp_damage_taken"] + damages_df["armor_damage_taken"]
        )

        damages_df.loc[:, "attacker_steam_id"] = damages_df["attacker_steam_id"].astype(
            "Int64"
        )
        damages_df.loc[:, "victim_steam_id"] = damages_df["victim_steam_id"].astype(
            "Int64"
        )

        processed_dfs["damages"] = damages_df

    if "grenades" in processed_dfs:
        grenades_df = processed_dfs["grenades"]

        grenades_df.loc[:, "grenade_side"] = grenades_df["thrower_side"]

        grenades_df = grenades_df.loc[
            grenades_df["throw_tick"] < grenades_df["destroy_tick"]
        ]
        processed_dfs["grenades"] = grenades_df

    if "matches" in processed_dfs:
        processed_dfs["matches"]["match_date"] = pd.to_datetime(
            processed_dfs["matches"]["match_date"]
        )

    if "rounds" in processed_dfs:
        processed_dfs["rounds"]["n_ticks"] = (
            processed_dfs["rounds"]["end_tick"] - processed_dfs["rounds"]["start_tick"]
        )

    return processed_dfs
