import pandas as pd


def get_round_sum(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rounds_sum_df = (
        dfs["rounds"]
        .loc[
            :,
            [
                "n_ticks",
                "bomb_plant_tick",
                "winning_side",
                "round_end_reason",
                "ct_freeze_time_end_eq_val",
                "t_freeze_time_end_eq_val",
            ],
        ]
        .copy()
    )

    default_columns = [
        "ct_flash_opp",
        "t_flash_opp",
        "ct_flashed",
        "t_flashed",
        "ct_avg_flash_duration",
        "t_avg_flash_duration",
        "ct_kills",
        "t_kills",
        "ct_first_kill",
        "t_first_kill",
        "ct_hs_prob",
        "t_hs_prob",
        "ct_trade_prob",
        "t_trade_prob",
        "ct_flash_kill_prob",
        "t_flash_kill_prob",
        "ct_assist_prob",
        "t_assist_prob",
        "ct_kast_prob",
        "t_kast_prob",
        "ct_damage_done",
        "t_damage_done",
        "ct_shot_fires",
        "t_shot_fires",
        "ct_shot_connected_prob",
        "t_shot_connected_prob",
        "ct_grenades_thrown",
        "t_grenades_thrown",
        "ct_grenade_connected_prob",
        "t_grenade_connected_prob",
    ]

    for col in default_columns:
        rounds_sum_df[col] = 0

    if "flashes" in dfs and not dfs["flashes"].empty:

        flashes_df = dfs["flashes"].reset_index()

        flash_agg = flashes_df.groupby("round_num").agg(
            {
                "attacker_side": lambda x: {
                    "ct_flash_opp": (x == "CT").sum(),
                    "t_flash_opp": (x == "T").sum(),
                },
                "player_side": lambda x: {
                    "ct_flashed": (x == "CT").sum(),
                    "t_flashed": (x == "T").sum(),
                },
                "flash_duration": lambda x: {
                    "ct_avg_flash_duration": (
                        x[flashes_df.loc[x.index, "player_side"] == "CT"].mean()
                        if (flashes_df.loc[x.index, "player_side"] == "CT").any()
                        else 0
                    ),
                    "t_avg_flash_duration": (
                        x[flashes_df.loc[x.index, "player_side"] == "T"].mean()
                        if (flashes_df.loc[x.index, "player_side"] == "T").any()
                        else 0
                    ),
                },
            }
        )
        for idx in flash_agg.index:
            if idx in rounds_sum_df.index:
                rounds_sum_df.loc[idx, "ct_flash_opp"] = flash_agg.loc[
                    idx, "attacker_side"
                ]["ct_flash_opp"]
                rounds_sum_df.loc[idx, "t_flash_opp"] = flash_agg.loc[
                    idx, "attacker_side"
                ]["t_flash_opp"]
                rounds_sum_df.loc[idx, "ct_flashed"] = flash_agg.loc[
                    idx, "player_side"
                ]["ct_flashed"]
                rounds_sum_df.loc[idx, "t_flashed"] = flash_agg.loc[idx, "player_side"][
                    "t_flashed"
                ]
                rounds_sum_df.loc[idx, "ct_avg_flash_duration"] = flash_agg.loc[
                    idx, "flash_duration"
                ]["ct_avg_flash_duration"]
                rounds_sum_df.loc[idx, "t_avg_flash_duration"] = flash_agg.loc[
                    idx, "flash_duration"
                ]["t_avg_flash_duration"]

    if "kills" in dfs and not dfs["kills"].empty:
        kills_df = dfs["kills"].reset_index()

        # Create helper columns
        kills_df["ct_attacker"] = kills_df["attacker_side"] == "CT"
        kills_df["t_attacker"] = kills_df["attacker_side"] == "T"
        kills_df["ct_victim"] = kills_df["victim_side"] == "CT"
        kills_df["t_victim"] = kills_df["victim_side"] == "T"

        # Group by round and calculate aggregations
        kill_stats = kills_df.groupby("round_num").agg(
            {
                "ct_attacker": "sum",
                "t_attacker": "sum",
                "ct_victim": "sum",
                "t_victim": "sum",
                "is_first_kill": lambda x: {
                    "ct_first_kill": (x & kills_df.loc[x.index, "ct_attacker"]).any(),
                    "t_first_kill": (x & kills_df.loc[x.index, "t_attacker"]).any(),
                },
                "is_headshot": lambda x: {
                    "ct_hs_prob": (
                        x[kills_df.loc[x.index, "ct_attacker"]].mean()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_hs_prob": (
                        x[kills_df.loc[x.index, "t_attacker"]].mean()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                },
                "is_trade": lambda x: {
                    "ct_trade_prob": (
                        x[kills_df.loc[x.index, "ct_attacker"]].mean()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_trade_prob": (
                        x[kills_df.loc[x.index, "t_attacker"]].mean()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                },
                "victim_blinded": lambda x: {
                    "ct_flash_kill_prob": (
                        x[kills_df.loc[x.index, "ct_attacker"]].mean()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_flash_kill_prob": (
                        x[kills_df.loc[x.index, "t_attacker"]].mean()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                },
                "attacker_steam_id": lambda x: {
                    "ct_unique_killers": (
                        x[kills_df.loc[x.index, "ct_attacker"]].nunique()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_unique_killers": (
                        x[kills_df.loc[x.index, "t_attacker"]].nunique()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                },
                "assister_steam_id": lambda x: {
                    "ct_assist_prob": (
                        x[kills_df.loc[x.index, "ct_attacker"]].notna().mean()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_assist_prob": (
                        x[kills_df.loc[x.index, "t_attacker"]].notna().mean()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                    "ct_assisters": (
                        x[kills_df.loc[x.index, "ct_attacker"]].nunique()
                        if kills_df.loc[x.index, "ct_attacker"].any()
                        else 0
                    ),
                    "t_assisters": (
                        x[kills_df.loc[x.index, "t_attacker"]].nunique()
                        if kills_df.loc[x.index, "t_attacker"].any()
                        else 0
                    ),
                },
            }
        )

        # Update rounds_sum_df with kill statistics
        for idx in kill_stats.index:
            if idx in rounds_sum_df.index:
                rounds_sum_df.loc[idx, "ct_kills"] = kill_stats.loc[idx, "ct_attacker"]
                rounds_sum_df.loc[idx, "t_kills"] = kill_stats.loc[idx, "t_attacker"]
                rounds_sum_df.loc[idx, "ct_first_kill"] = int(
                    kill_stats.loc[idx, "is_first_kill"]["ct_first_kill"]
                )
                rounds_sum_df.loc[idx, "t_first_kill"] = int(
                    kill_stats.loc[idx, "is_first_kill"]["t_first_kill"]
                )
                rounds_sum_df.loc[idx, "ct_hs_prob"] = kill_stats.loc[
                    idx, "is_headshot"
                ]["ct_hs_prob"]
                rounds_sum_df.loc[idx, "t_hs_prob"] = kill_stats.loc[
                    idx, "is_headshot"
                ]["t_hs_prob"]
                rounds_sum_df.loc[idx, "ct_trade_prob"] = kill_stats.loc[
                    idx, "is_trade"
                ]["ct_trade_prob"]
                rounds_sum_df.loc[idx, "t_trade_prob"] = kill_stats.loc[
                    idx, "is_trade"
                ]["t_trade_prob"]
                rounds_sum_df.loc[idx, "ct_flash_kill_prob"] = kill_stats.loc[
                    idx, "victim_blinded"
                ]["ct_flash_kill_prob"]
                rounds_sum_df.loc[idx, "t_flash_kill_prob"] = kill_stats.loc[
                    idx, "victim_blinded"
                ]["t_flash_kill_prob"]
                rounds_sum_df.loc[idx, "ct_assist_prob"] = kill_stats.loc[
                    idx, "assister_steam_id"
                ]["ct_assist_prob"]
                rounds_sum_df.loc[idx, "t_assist_prob"] = kill_stats.loc[
                    idx, "assister_steam_id"
                ]["t_assist_prob"]

                # Calculate KAST (simplified version)
                ct_survivors = 5 - kill_stats.loc[idx, "ct_victim"]
                t_survivors = 5 - kill_stats.loc[idx, "t_victim"]
                ct_kast = max(
                    kill_stats.loc[idx, "attacker_steam_id"]["ct_unique_killers"],
                    kill_stats.loc[idx, "assister_steam_id"]["ct_assisters"],
                    ct_survivors,
                    0,
                )
                t_kast = max(
                    kill_stats.loc[idx, "attacker_steam_id"]["t_unique_killers"],
                    kill_stats.loc[idx, "assister_steam_id"]["t_assisters"],
                    t_survivors,
                    0,
                )

                rounds_sum_df.loc[idx, "ct_kast_prob"] = ct_kast / 5
                rounds_sum_df.loc[idx, "t_kast_prob"] = t_kast / 5

    # Process damages
    if "grenades" in dfs and not dfs["grenades"].empty:

        damages_df = dfs["damages"].reset_index()
        damage_stats = damages_df.groupby("round_num").agg(
            {
                "total_damage": lambda x: {
                    "ct_damage": x[
                        damages_df.loc[x.index, "attacker_side"] == "CT"
                    ].sum(),
                    "t_damage": x[
                        damages_df.loc[x.index, "attacker_side"] == "T"
                    ].sum(),
                }
            }
        )

        for idx in damage_stats.index:
            if idx in rounds_sum_df.index:
                rounds_sum_df.loc[idx, "ct_damage_done"] = int(
                    damage_stats.loc[idx, "total_damage"]["ct_damage"]
                )
                rounds_sum_df.loc[idx, "t_damage_done"] = int(
                    damage_stats.loc[idx, "total_damage"]["t_damage"]
                )

    # Process weapon fires
    if dfs.get("weapon_fires", pd.DataFrame()).empty:
        weapon_fires_df = dfs["weapon_fires"].reset_index()
        weapon_stats = weapon_fires_df.groupby("round_num").agg(
            {
                "player_side": lambda x: {
                    "ct_shots": (x == "CT").sum(),
                    "t_shots": (x == "T").sum(),
                },
                "shot_connected": lambda x: {
                    "ct_connected_prob": (
                        x[weapon_fires_df.loc[x.index, "player_side"] == "CT"].mean()
                        if (weapon_fires_df.loc[x.index, "player_side"] == "CT").any()
                        else 0
                    ),
                    "t_connected_prob": (
                        x[weapon_fires_df.loc[x.index, "player_side"] == "T"].mean()
                        if (weapon_fires_df.loc[x.index, "player_side"] == "T").any()
                        else 0
                    ),
                },
            }
        )

        for idx in weapon_stats.index:
            if idx in rounds_sum_df.index:
                rounds_sum_df.loc[idx, "ct_shot_fires"] = weapon_stats.loc[
                    idx, "player_side"
                ]["ct_shots"]
                rounds_sum_df.loc[idx, "t_shot_fires"] = weapon_stats.loc[
                    idx, "player_side"
                ]["t_shots"]
                rounds_sum_df.loc[idx, "ct_shot_connected_prob"] = weapon_stats.loc[
                    idx, "shot_connected"
                ]["ct_connected_prob"]
                rounds_sum_df.loc[idx, "t_shot_connected_prob"] = weapon_stats.loc[
                    idx, "shot_connected"
                ]["t_connected_prob"]

    # Process grenades
    if dfs.get("grenades", pd.DataFrame()).empty:
        grenades_df = dfs["grenades"].reset_index()

        # Basic grenade counts
        grenade_stats = grenades_df.groupby("round_num").agg(
            {
                "thrower_side": lambda x: {
                    "ct_grenades": (x == "CT").sum(),
                    "t_grenades": (x == "T").sum(),
                },
                "grenade_connected": lambda x: {
                    "ct_connected_prob": (
                        x[grenades_df.loc[x.index, "thrower_side"] == "CT"].mean()
                        if (grenades_df.loc[x.index, "thrower_side"] == "CT").any()
                        else 0
                    ),
                    "t_connected_prob": (
                        x[grenades_df.loc[x.index, "thrower_side"] == "T"].mean()
                        if (grenades_df.loc[x.index, "thrower_side"] == "T").any()
                        else 0
                    ),
                },
            }
        )

        for idx in grenade_stats.index:
            if idx in rounds_sum_df.index:
                rounds_sum_df.loc[idx, "ct_grenades_thrown"] = grenade_stats.loc[
                    idx, "thrower_side"
                ]["ct_grenades"]
                rounds_sum_df.loc[idx, "t_grenades_thrown"] = grenade_stats.loc[
                    idx, "thrower_side"
                ]["t_grenades"]
                rounds_sum_df.loc[idx, "ct_grenade_connected_prob"] = grenade_stats.loc[
                    idx, "grenade_connected"
                ]["ct_connected_prob"]
                rounds_sum_df.loc[idx, "t_grenade_connected_prob"] = grenade_stats.loc[
                    idx, "grenade_connected"
                ]["t_connected_prob"]

        # Process grenade types
        grenade_types_pivot = grenades_df.pivot_table(
            index="round_index",
            columns=["thrower_side", "grenade_type"],
            values="grenade_type",
            aggfunc="count",
            fill_value=0,
        )

        # Flatten column names and add to rounds_sum_df
        for col in grenade_types_pivot.columns:
            side, grenade_type = col
            col_name = f"{side.lower()}_{grenade_type.lower()}"
            if col_name not in rounds_sum_df.columns:
                rounds_sum_df[col_name] = 0
            rounds_sum_df.loc[grenade_types_pivot.index, col_name] = (
                grenade_types_pivot[col].values
            )

    # Final processing
    rounds_sum_df["bomb_plant_tick"] = (
        rounds_sum_df["bomb_plant_tick"].fillna(-1).astype(int)
    )

    # Combine fire grenades
    rounds_sum_df["ct_fire_grenade"] = rounds_sum_df.get(
        "ct_incendiary_grenade", 0
    ) + rounds_sum_df.get("ct_molotov", 0)
    rounds_sum_df["t_fire_grenade"] = rounds_sum_df.get(
        "t_incendiary_grenade", 0
    ) + rounds_sum_df.get("t_molotov", 0)

    # Drop original fire grenade columns
    rounds_sum_df.drop(
        ["ct_incendiary_grenade", "t_incendiary_grenade", "ct_molotov", "t_molotov"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # Convert appropriate columns to int
    count_columns = [
        "ct_flash_opp",
        "t_flash_opp",
        "ct_flashed",
        "t_flashed",
        "ct_kills",
        "t_kills",
        "ct_first_kill",
        "t_first_kill",
        "ct_shot_fires",
        "t_shot_fires",
        "ct_grenades_thrown",
        "t_grenades_thrown",
        "ct_he_grenade",
        "t_he_grenade",
        "ct_smoke_grenade",
        "t_smoke_grenade",
        "ct_flashbang",
        "t_flashbang",
        "ct_decoy_grenade",
        "t_decoy_grenade",
        "ct_fire_grenade",
        "t_fire_grenade",
    ]

    damage_columns = ["ct_damage_done", "t_damage_done"]

    for col in count_columns + damage_columns:
        if col in rounds_sum_df.columns:
            rounds_sum_df[col] = rounds_sum_df[col].fillna(0).astype(int)

    # Fill NaN values in probability columns
    prob_columns = [
        "ct_grenade_connected_prob",
        "t_grenade_connected_prob",
        "ct_shot_connected_prob",
        "t_shot_connected_prob",
        "ct_avg_flash_duration",
        "t_avg_flash_duration",
        "ct_hs_prob",
        "t_hs_prob",
        "ct_trade_prob",
        "t_trade_prob",
        "ct_flash_kill_prob",
        "t_flash_kill_prob",
        "ct_assist_prob",
        "t_assist_prob",
        "ct_kast_prob",
        "t_kast_prob",
    ]

    for col in prob_columns:
        if col in rounds_sum_df.columns:
            rounds_sum_df[col] = rounds_sum_df[col].fillna(0)

    # One-hot encode round_end_reason
    rounds_sum_df = pd.concat(
        [
            rounds_sum_df,
            pd.get_dummies(
                rounds_sum_df["round_end_reason"], prefix="round_end_reason"
            ).astype(int),
        ],
        axis=1,
    )

    # Convert winning_side to binary
    rounds_sum_df["winning_side"] = (
        rounds_sum_df["winning_side"].str.lower() == "ct"
    ).astype(int)

    return rounds_sum_df
