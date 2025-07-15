import pandas as pd
import numpy as np


def _get_value_difference(round_sum: pd.DataFrame) -> pd.DataFrame:
    if "winning_side" not in round_sum:
        return pd.DataFrame(columns=np.array(["Feature", "%diff"]))

    ct_win_df = round_sum[round_sum["winning_side"] == 1]
    t_win_df = round_sum[round_sum["winning_side"] == 0]

    # Select relevant features
    features_to_compare = round_sum.columns.to_list()
    for col in ["bomb_plant_tick", "winning_side"]:
        if col in features_to_compare:
            features_to_compare.remove(col)

    features_to_compare = [
        col for col in features_to_compare if not col.startswith("round_end_reason")
    ]

    # Calculate percentage difference
    percentage_df = pd.DataFrame()
    for feature in features_to_compare:
        t_mean = t_win_df[feature].mean()
        ct_mean = ct_win_df[feature].mean()
        if (t_mean > 0) and (ct_mean > 0):
            percentage_df.loc[feature, "%diff"] = ((ct_mean / t_mean) - 1) * 100
        elif (t_mean == 0) and (ct_mean == 0):
            percentage_df.loc[feature, "%diff"] = 0
        else:
            percentage_df.loc[feature, "%diff"] = (
                percentage_df["%diff"].max() * 1.05
                if ct_mean > 0
                else percentage_df["%diff"].min() * 1.05
            )
    # Prepare for plotting
    percentage_df = (
        percentage_df.reset_index()
        .rename(columns={"index": "Feature"})
        .sort_values("%diff", ascending=False)
        .dropna(how="any")
    )

    return percentage_df
