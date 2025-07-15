import plotly.graph_objects as go
import pandas as pd
from utils.symbols import win_reason_colors


def plot_combined_economy_with_reasons(rounds_sum_df: pd.DataFrame) -> go.Figure:

    rounds_sum_df["round"] = range(len(rounds_sum_df))

    fig = go.Figure()

    # Economy lines
    fig.add_trace(
        go.Scatter(
            x=rounds_sum_df["round"],
            y=rounds_sum_df["ct_freeze_time_end_eq_val"],
            name="CT Economy",
            mode="lines+markers",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rounds_sum_df["round"],
            y=rounds_sum_df["t_freeze_time_end_eq_val"],
            name="T Economy",
            mode="lines+markers",
            line=dict(color="red"),
        )
    )

    # Vertical lines + rotated annotations
    for _, row in rounds_sum_df.iterrows():
        reason = str(row["round_end_reason"])
        color = win_reason_colors.get(reason, "#888888")
        round_num = row["round"]

        # Vertical line
        fig.add_vline(
            x=round_num, line=dict(color=color, width=2, dash="dot"), opacity=0.5
        )

        # Rotated annotation (above the top of the y-axis)
        fig.add_annotation(
            x=round_num,
            y=max(
                rounds_sum_df["ct_freeze_time_end_eq_val"].max(),
                rounds_sum_df["t_freeze_time_end_eq_val"].max(),
            )
            * 1.05,  # position above the lines
            text=reason,
            showarrow=False,
            textangle=-90,  # vertical rotation
            font=dict(size=10, color=color),
            yanchor="bottom",
        )

    fig.update_layout(
        title="T and CT Economy with Round End Reasons",
        xaxis_title="Round",
        yaxis_title="Freeze Time End Equipment Value",
        legend_title="Team",
        height=700,
        margin=dict(t=50, b=50),
        showlegend=True,
    )

    return fig
