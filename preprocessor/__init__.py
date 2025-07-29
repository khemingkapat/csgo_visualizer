import pandas as pd
from .indexer import index_dfs
from .context import preprocess_context_data
from .connections import add_action_connection
from .normalizer import normalize_tick, normalize_steamid_columns


class Preprocessor:
    # preprocess = staticmethod(preprocess)
    index_dfs = staticmethod(index_dfs)
    preprocess_context_data = staticmethod(preprocess_context_data)
    add_action_connection = staticmethod(add_action_connection)
    normalize_tick = staticmethod(normalize_tick)
    normalize_steamid_columns = staticmethod(normalize_steamid_columns)

    @staticmethod
    def preprocess_single_match(
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        dfs["ticks"] = dfs["ticks"].loc[dfs["ticks"].tick % 32 == 0]
        dfs = normalize_steamid_columns(dfs)
        dfs = index_dfs(dfs, include=None, index_by=["round_num"])
        dfs = preprocess_context_data(dfs)
        dfs["shots"] = Preprocessor.add_action_connection(
            dfs["shots"],
            dfs["damages"],
            target_combo_key=["tick", "player_steamid"],
            check_combo_key=["tick", "attacker_steamid"],
            connect_prefix="shot",
        )
        dfs["grenades"] = Preprocessor.add_action_connection(
            dfs["grenades"],
            dfs["damages"],
            target_combo_key=["destroy_tick", "thrower_steamid"],
            check_combo_key=["tick", "attacker_steamid"],
            connect_prefix="grenade",
        )
        dfs = normalize_tick(dfs)
        if "smokes" in dfs and not dfs["smokes"].empty:
            dfs["smokes"].rename(
                columns={
                    "x": "smoke_x",
                    "y": "smoke_y",
                    "z": "smoke_z",
                },
                inplace=True,
            )
            dfs["smokes"]["smoke_side"] = dfs["smokes"]["thrower_side"]

        if "infernos" in dfs and not dfs["infernos"].empty:
            dfs["infernos"].rename(
                columns={
                    "x": "inferno_x",
                    "y": "inferno_y",
                    "z": "inferno_z",
                },
                inplace=True,
            )
            dfs["infernos"]["inferno_side"] = dfs["infernos"]["thrower_side"]

        return dfs
