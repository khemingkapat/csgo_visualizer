import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_location_change_analysis(
    dfs: dict[str, pd.DataFrame], round_num: int
) -> go.Figure:

    # Create subplots with 3 rows and 1 column
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[
            "<b>Location Change with Smoke</b>",
            "<b>Location Change with Inferno</b>",
            "<b>Location Change with Kills</b>",
            "<b>Location Change with Grenade</b>",
        ],
        vertical_spacing=0.15,
        shared_xaxes=True,
    )

    # Preprocessing: Location Change
    sel_loc = dfs["ticks"].loc[round_num][["tick", "side", "x", "y"]].copy()
    curr_x = sel_loc.iloc[1:]["x"].values
    prev_x = sel_loc.iloc[:-1]["x"].values
    curr_y = sel_loc.iloc[1:]["y"].values
    prev_y = sel_loc.iloc[:-1]["y"].values
    loc_change = [0] + list(np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2))
    sel_loc["loc_change"] = loc_change
    sel_team_loc = sel_loc.groupby(["side", "tick"]).agg(
        avg_loc_change=("loc_change", "mean")
    )

    ct_loc_df = sel_team_loc.loc["ct"].reset_index()
    t_loc_df = sel_team_loc.loc["t"].reset_index()

    # Colors - bright colors for dark theme
    ct_color = "rgba(0, 150, 255, 1)"
    t_color = "rgba(255, 80, 80, 1)"

    # Add traces for all three subplots - Team location changes
    for i in range(1, 5):
        # CT Location Change
        fig.add_trace(
            go.Scatter(
                x=ct_loc_df["tick"],
                y=ct_loc_df["avg_loc_change"],
                name="CT Avg. Loc. Change",
                line=dict(color=ct_color, width=2.5),
                hovertemplate="Tick: %{x}<br>Avg. Location Change: %{y:.2f}<extra></extra>",
                legendgroup="ct_loc",
                showlegend=True if i == 1 else False,
            ),
            row=i,
            col=1,
        )

        # T Location Change
        fig.add_trace(
            go.Scatter(
                x=t_loc_df["tick"],
                y=t_loc_df["avg_loc_change"],
                name="T Avg. Loc. Change",
                line=dict(color=t_color, width=2.5),
                hovertemplate="Tick: %{x}<br>Avg. Location Change: %{y:.2f}<extra></extra>",
                legendgroup="t_loc",
                showlegend=True if i == 1 else False,
            ),
            row=i,
            col=1,
        )

    # Get y-axis ranges for proper vertical line scaling
    y_ranges = []
    for i in range(1, 5):
        max_ct = max(ct_loc_df["avg_loc_change"])
        max_t = max(t_loc_df["avg_loc_change"])
        y_max = max(max_ct, max_t) * 1.1  # Add some padding
        y_ranges.append([0, y_max])

        fig.update_yaxes(range=[0, y_max], row=i, col=1)

    # Smokes

    if "smokes" in dfs and not dfs["smokes"].empty:
        sel_s = dfs["smokes"].loc[round_num].reset_index()
        smoke_events = []
        for attacker_side in ["ct", "t"]:
            smoke_ticks = sel_s[sel_s.thrower_side == attacker_side]["start_tick"]
            color = (
                "rgba(0, 150, 255, 0.8)"
                if attacker_side == "ct"
                else "rgba(255, 80, 80, 0.8)"
            )
            if len(smoke_ticks) > 0:
                smoke_events.append(
                    {"side": attacker_side, "ticks": smoke_ticks, "color": color}
                )
                # Add a single legend item for each side's smokes
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        name=f"{attacker_side} Smoke",
                        legendgroup=f"{attacker_side}_smoke",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
        # Add vertical lines for smokes
        for event in smoke_events:
            for tick_value in event["ticks"]:
                fig.add_shape(
                    type="line",
                    x0=tick_value,
                    x1=tick_value,
                    y0=0,
                    y1=y_ranges[0][1],
                    line=dict(color=event["color"], width=1.5, dash="dash"),
                    row=1,
                    col=1,
                )

    # Infernos
    if "infernos" in dfs and not dfs["infernos"].empty:
        sel_i = dfs["infernos"].loc[round_num].reset_index()
        inferno_events = []
        for attacker_side in ["ct", "t"]:
            inferno_ticks = sel_i[sel_i.thrower_side == attacker_side]["start_tick"]
            color = (
                "rgba(0, 150, 255, 0.8)"
                if attacker_side == "ct"
                else "rgba(255, 80, 80, 0.8)"
            )
            if len(inferno_ticks) > 0:
                inferno_events.append(
                    {"side": attacker_side, "ticks": inferno_ticks, "color": color}
                )
                # Add a single legend item for each side's infernos
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        name=f"{attacker_side} Inferno",
                        legendgroup=f"{attacker_side}_inferno",
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )
        # Add vertical lines for infernos
        for event in inferno_events:
            for tick_value in event["ticks"]:
                fig.add_shape(
                    type="line",
                    x0=tick_value,
                    x1=tick_value,
                    y0=0,
                    y1=y_ranges[0][1],
                    line=dict(color=event["color"], width=1.5, dash="dash"),
                    row=2,
                    col=1,
                )

    # Subplot 2: Add Kill events
    if "kills" in dfs and not dfs["kills"].empty:
        sel_k = dfs["kills"].loc[round_num].reset_index()

        kill_events = []
        for attacker_side in ["ct", "t"]:
            kill_ticks = sel_k[sel_k.attacker_side == attacker_side]["tick"]
            color = (
                "rgba(200, 0, 200, 0.8)"
                if attacker_side == "ct"
                else "rgba(255, 200, 0, 0.8)"
            )

            if len(kill_ticks) > 0:
                kill_events.append(
                    {"side": attacker_side, "ticks": kill_ticks, "color": color}
                )

                # Add a single legend item for each side's kills
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        name=f"{attacker_side} Kill",
                        legendgroup=f"{attacker_side}_kill",
                        showlegend=True,
                    ),
                    row=3,
                    col=1,
                )

        # Add vertical lines for kills
        for event in kill_events:
            for tick_value in event["ticks"]:
                fig.add_shape(
                    type="line",
                    x0=tick_value,
                    x1=tick_value,
                    y0=0,
                    y1=y_ranges[1][1],
                    line=dict(color=event["color"], width=1.5, dash="dash"),
                    row=3,
                    col=1,
                )

    # Subplot 3: Add Grenade events
    if "grenades" in dfs and not dfs["grenades"].empty:
        sel_g = dfs["grenades"].loc[round_num].reset_index()

        grenade_events = []
        for thrower_side in ["ct", "t"]:
            grenade_ticks = sel_g[sel_g.thrower_side == thrower_side]["throw_tick"]
            color = (
                "rgba(255, 105, 180, 0.8)"
                if thrower_side == "ct"
                else "rgba(0, 255, 255, 0.8)"
            )

            if len(grenade_ticks) > 0:
                grenade_events.append(
                    {"side": thrower_side, "ticks": grenade_ticks, "color": color}
                )

                # Add a single legend item for each side's grenades
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        name=f"{thrower_side} Grenade",
                        legendgroup=f"{thrower_side}_grenade",
                        showlegend=True,
                    ),
                    row=4,
                    col=1,
                )

        # Add vertical lines for grenades
        for event in grenade_events:
            for tick_value in event["ticks"]:
                fig.add_shape(
                    type="line",
                    x0=tick_value,
                    x1=tick_value,
                    y0=0,
                    y1=y_ranges[2][1],
                    line=dict(color=event["color"], width=1.5, dash="dash"),
                    row=4,
                    col=1,
                )

    # Update layout with dark theme
    fig.update_layout(
        height=1000,
        width=900,
        title=dict(
            text=f"Location Change Analysis - Round {round_num}",
            font=dict(size=22, color="white"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",  # Changed from "bottom" to "top"
            y=-0.15,  # Changed from 1.05 to -0.15
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="white"),
            bgcolor="rgba(30, 30, 30, 0.8)",
            bordercolor="rgba(100, 100, 100, 0.8)",
            borderwidth=1,
            itemsizing="constant",
            itemwidth=40,
            tracegroupgap=5,
        ),
        hovermode="closest",
        plot_bgcolor="rgb(17, 17, 17)",
        paper_bgcolor="rgb(17, 17, 17)",
        font=dict(family="Arial, sans-serif", size=14, color="white"),
        margin=dict(
            l=80, r=50, t=120, b=120
        ),  # Increased bottom margin to accommodate legend
    )

    # Update axes
    fig.update_xaxes(
        title=dict(text="Tick", font=dict(size=16, color="white")),
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(70, 70, 70, 1)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(100, 100, 100, 1)",
        showline=True,
        linewidth=1,
        linecolor="rgba(150, 150, 150, 1)",
        row=3,
        col=1,  # Only show x-axis title on bottom plot
    )

    # Update y-axes for each subplot
    for i in range(1, 4):
        fig.update_yaxes(
            title=dict(
                text="Average Location Change", font=dict(size=16, color="white")
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(70, 70, 70, 1)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(100, 100, 100, 1)",
            showline=True,
            linewidth=1,
            linecolor="rgba(150, 150, 150, 1)",
            row=i,
            col=1,
        )

    return fig
