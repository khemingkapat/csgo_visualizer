import pandas as pd


def filter_data_by_tick(
    round_dfs: dict[str, pd.DataFrame], min_tick: int, max_tick: int
) -> dict[str, pd.DataFrame]:
    """Filter all dataframes by tick range"""
    filtered_data = {}

    # Filter locations
    if "player_locations" in round_dfs and len(round_dfs["player_locations"]) > 0:
        loc_mask = (round_dfs["player_locations"]["tick"] >= min_tick) & (
            round_dfs["player_locations"]["tick"] <= max_tick
        )
        filtered_data["locations"] = round_dfs["player_locations"][loc_mask]
    else:
        filtered_data["locations"] = pd.DataFrame()

    # Filter flashes
    if "flashes" in round_dfs and len(round_dfs["flashes"]) > 0:
        flash_mask = (round_dfs["flashes"]["tick"] >= min_tick) & (
            round_dfs["flashes"]["tick"] <= max_tick
        )
        filtered_data["flashes"] = round_dfs["flashes"][flash_mask]
    else:
        filtered_data["flashes"] = pd.DataFrame()

    # Filter kills
    if "kills" in round_dfs and len(round_dfs["kills"]) > 0:
        kill_mask = (round_dfs["kills"]["tick"] >= min_tick) & (
            round_dfs["kills"]["tick"] <= max_tick
        )
        filtered_data["kills"] = round_dfs["kills"][kill_mask]
    else:
        filtered_data["kills"] = pd.DataFrame()

    # Filter grenades
    if "grenades" in round_dfs and len(round_dfs["grenades"]) > 0:
        grenade_mask = (round_dfs["grenades"]["throw_tick"] >= min_tick) & (
            round_dfs["grenades"]["throw_tick"] <= max_tick
        )
        filtered_data["grenades"] = round_dfs["grenades"][grenade_mask]
    else:
        filtered_data["grenades"] = pd.DataFrame()

    # Filter line data
    line_filters = ["flash_lines", "kill_lines", "grenade_lines"]
    tick_columns = ["tick", "tick", "throw_tick"]

    for line_type, tick_col in zip(line_filters, tick_columns):
        if line_type in round_dfs and len(round_dfs[line_type]) > 0:
            line_mask = (round_dfs[line_type][tick_col] >= min_tick) & (
                round_dfs[line_type][tick_col] <= max_tick
            )
            filtered_data[line_type] = round_dfs[line_type][line_mask]
        else:
            filtered_data[line_type] = pd.DataFrame()

    return filtered_data
