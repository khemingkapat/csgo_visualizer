import pandas as pd
from .indexer import index_dfs
from .context import preprocess_context_data
from .connections import add_action_connection
from .normalizer import normalize_tick


class Preprocessor:
    # preprocess = staticmethod(preprocess)
    index_dfs = staticmethod(index_dfs)
    preprocess_context_data = staticmethod(preprocess_context_data)
    add_action_connection = staticmethod(add_action_connection)
    normalize_tick = staticmethod(normalize_tick)

    @staticmethod
    def preprocess_single_match(
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        dfs = index_dfs(dfs, include=None, index_by=["round_num"])

        dfs = preprocess_context_data(dfs)

        dfs["weapon_fires"] = add_action_connection(
            dfs["weapon_fires"],
            dfs["damages"],
            target_combo_key=["tick", "player_steam_id"],
            check_combo_key=["tick", "attacker_steam_id"],
            connect_prefix="shot",
        )

        dfs["grenades"] = add_action_connection(
            dfs["grenades"],
            dfs["damages"],
            target_combo_key=["destroy_tick", "thrower_steam_id"],
            check_combo_key=["tick", "attacker_steam_id"],
            connect_prefix="grenade",
        )

        dfs = normalize_tick(dfs)

        return dfs
