import pandas as pd


def get_player_stat(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:

    player_stat = (
        dfs["player_frames"][
            [
                "steam_id",
                "name",
                "team",
            ]
        ]
        .drop_duplicates()
        .set_index(["name"])
    )
    max_round = dfs["rounds"].index.max()

    kills = dfs["kills"]
    damages = dfs["damages"]

    for name in player_stat.index:
        player_kill_mask = kills.attacker_name == name
        player_death_mask = kills.victim_name == name
        player_assist_mask = kills.assister_name == name
        player_flash_assist_mask = kills.flash_thrower_name == name
        player_traded_mask = kills.player_traded_name == name

        k = player_kill_mask.groupby("round_num").any()
        a = (
            player_assist_mask.groupby("round_num").any()
            | player_flash_assist_mask.groupby("round_num").any()
        )
        s = ~(player_death_mask.groupby("round_num").any())
        t = player_traded_mask.groupby("round_num").any()

        player_kill = kills[player_kill_mask]
        player_death = kills[player_death_mask]
        player_assist = kills[player_assist_mask]
        player_flash_assist = kills[player_flash_assist_mask]

        # display(sel_player_death)

        player_stat.loc[name, "kills"] = len(player_kill)
        player_stat.loc[name, "deaths"] = len(player_death)
        player_stat.loc[name, "assist"] = len(player_assist)
        player_stat.loc[name, "flash_assist"] = len(player_flash_assist)
        player_stat.loc[name, "first_kill"] = player_kill.is_first_kill.sum()
        player_stat.loc[name, "first_death"] = player_death.is_first_kill.sum()
        player_stat.loc[name, "kast"] = (k | a | s | t).sum() / max_round

        sel_player_damage = damages[damages.attacker_name == name]
        sel_player_dmg_round = sel_player_damage.groupby("round_num").agg(
            rtd=("total_damage", "sum")
        )
        player_stat.loc[name, "ADR"] = sel_player_dmg_round.rtd.sum() / max_round
    return player_stat
