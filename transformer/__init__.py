from .coordinate import transform_coords, transform_coord
from .action import transform_action
from .round_sum import get_round_sum
from .player_stat import get_player_stat

import pandas as pd


class Transformer:
    transform_coord = staticmethod(transform_coord)
    transform_coords = staticmethod(transform_coords)
    transform_action = staticmethod(transform_action)
    get_round_sum = staticmethod(get_round_sum)
    get_player_stat = staticmethod(get_player_stat)

    @staticmethod
    def transform_actions(
        dfs: dict[str, pd.DataFrame], all_map_data: dict[str, dict[str, int]]
    ) -> dict[str, pd.DataFrame | str]:

        result: dict[str, pd.DataFrame | str] = dict()
        selected_map = dfs["matches"]["map_name"].iloc[0]
        map_data = all_map_data[selected_map]

        # transforming player location from player_frames
        result["player_locations"] = transform_coord(map_data, dfs["player_frames"])

        # transforming flash data with both flash location and line
        if "flashes" in dfs and not dfs["flashes"].empty:
            flash_lines = transform_coords(
                map_data, dfs["flashes"], ["attacker", "player"]
            )
            flashes = transform_action(
                flash_lines,
                ["attacker", "player"],
                {
                    "player": [
                        "flash_duration",
                    ]
                },
                {"attacker": "flasher", "player": "flashee"},
                check_dup_without_cols=[
                    "status",
                    "flash_duration",
                    "attacker_side",
                    "player_side",
                ],
                assign_dup={"status": "both"},
                common_extra_cols=["tick"],
            )
            result["flashes"] = flashes
            result["flash_lines"] = flash_lines

        # transforming kills data with both kills location and lines
        if "kills" in dfs and not dfs["kills"].empty:
            kill_lines = transform_coords(
                map_data, dfs["kills"], ["attacker", "victim"]
            )
            kills = transform_action(
                kill_lines,
                ["attacker", "victim"],
                check_dup_without_cols=["attacker_side", "victim_side"],
                assign_dup={"status": "suicide"},
                common_extra_cols=["tick"],
            )

            result["kills"] = kills
            result["kill_lines"] = kill_lines

        # transforming grenade data with both grenades location and lines
        if "grenades" in dfs and not dfs["grenades"].empty:
            grenade_lines = transform_coords(
                map_data, dfs["grenades"], ["thrower", "grenade"]
            )
            grenades = transform_action(
                grenade_lines, ["thrower", "grenade"], common_extra_cols=["throw_tick"]
            )
            result["grenades"] = grenades
            result["grenade_lines"] = grenade_lines

        result["map"] = selected_map
        return result

    @staticmethod
    def get_round_result(round_sum: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame()
        if "winning_side" in round_sum:
            result_df["ct_win"] = round_sum["winning_side"]
            result_df["t_win"] = (~result_df["ct_win"].astype(bool)).astype(int)

        if ("ct_kills" in round_sum) and ("t_kills" in round_sum):
            result_df[["ct_kills", "t_kills"]] = round_sum[["ct_kills", "t_kills"]]

        if "round_end_reason" in round_sum:
            result_df["round_end_reason"] = round_sum["round_end_reason"]
        return result_df.reset_index()

    @staticmethod
    def transform_all_data(
        dfs: dict[str, pd.DataFrame], all_map_data: dict[str, dict[str, int]]
    ) -> dict[str, pd.DataFrame | str]:
        act_loc_transformed_dfs = Transformer.transform_actions(dfs, all_map_data)
        round_sum = Transformer.get_round_sum(dfs)
        act_loc_transformed_dfs["rounds_sum"] = round_sum
        act_loc_transformed_dfs["player_stats"] = Transformer.get_player_stat(dfs)
        act_loc_transformed_dfs["round_results"] = Transformer.get_round_result(
            round_sum
        )

        return act_loc_transformed_dfs
