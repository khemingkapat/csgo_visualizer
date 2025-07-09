from matplotlib import legend
import streamlit as st
import matplotlib.pyplot as plt
import json
import pandas as pd
from .symbols import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image


def upload_and_parse_json(preview_limit=10):
    """
    Streamlit widget to upload a JSON file and return the parsed content.

    Args:
        preview_limit (int): Number of items to preview in the UI.
                             If 0, no preview is shown.

    Returns:
        dict or list or None: Parsed JSON data if uploaded successfully, else None.
    """
    uploaded_file = st.file_uploader("Upload your JSON file", type="json")

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.success("✅ JSON file loaded successfully!")

            # Show preview only if preview_limit > 0
            if preview_limit > 0:
                if isinstance(data, dict):
                    preview = {
                        k: data[k]
                        for i, k in enumerate(data.keys())
                        if i < preview_limit
                    }
                elif isinstance(data, list):
                    preview = data[:preview_limit]
                else:
                    preview = str(data)

                st.write(f"📄 Preview (first {preview_limit} items or keys):")
                st.json(preview)

            return data

        except Exception as e:
            st.error(f"❌ Error loading JSON: {e}")
            return None

    else:
        st.info("📂 Please upload a JSON file.")
        return None


def plot_map(map_name, default_img_dim=1024):
    map_image_path = f".awpy/maps/{map_name}.png"

    fig = go.Figure()
    image_height, image_width = default_img_dim, default_img_dim

    image = Image.open(map_image_path)
    image_width, image_height = image.size  # Get actual pixel dimensions

    fig.add_layout_image(
        dict(
            source=image,  # Directly pass the PIL Image object
            xref="x",  # Reference x-axis data coordinates
            yref="y",  # Reference y-axis data coordinates
            x=0,  # X-coordinate of the image's left edge (assuming map starts at 0)
            y=image_height,  # Y-coordinate of the image's top edge (max y-value for top)
            sizex=image_width,  # Width of the image in data units (use actual image width)
            sizey=image_height,  # Height of the image in data units (use actual image height)
            sizing="stretch",  # Stretch image to fit the defined size
            opacity=1.0,  # Full opacity for the map
            layer="below",  # Crucial: Ensures the image is drawn behind data traces
        )
    )

    return fig


def create_plotly_actions_plot(
    round_dfs,
    max_tick,
    show_loc=True,
    show_flash=True,
    show_kills=True,
    show_grenades=True,
    flash_alpha=0.7,
    kill_alpha=0.7,
    grenade_alpha=0.7,
    flash_size=10,
    kill_size=10,
    grenade_size=10,
    show_lines=True,
    map_name="de_dust2",
    fig_height=800,
):
    """
    Create a Plotly figure with CS:GO game actions filtered by max tick.
    This function creates a dynamic plot that can be updated without full reload.

    Parameters:
    -----------
    round_dfs : dict
        Dictionary containing DataFrames for different game actions
    max_tick : int
        Maximum tick value to filter actions
    show_* : bool
        Flags to control visibility of different action types
    *_alpha : float
        Transparency values for different action types
    *_size : int
        Size values for different action types
    show_lines : bool
        Whether to show connection lines between actions
    transformed_data : dict
        Additional data including map information
    map_name : str
        Name of the map for background

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """

    # Create the main figure
    fig = go.Figure()

    # Set up the plot dimensions (CS:GO maps are typically 1024x1024)
    plot_width, plot_height = 1024, 1024

    # Add map background if available
    fig = plot_map(map_name)

    # Filter data by tick range
    min_tick = 0
    filtered_data = filter_data_by_tick(round_dfs, min_tick, max_tick)

    # Add player locations
    if show_loc and len(filtered_data["locations"]) > 0:
        add_player_actions(
            fig,
            filtered_data["locations"],
            gradient_by="tick",
            color_by="side",
            color_dict=side_color,
            legendgroup="Locations",
        )

    # Add flashes
    if show_flash and len(filtered_data["flashes"]) > 0:
        legendgroup = "Flashes"
        add_player_actions(
            fig,
            filtered_data["flashes"],
            size=flash_size,
            alpha=flash_alpha,
            gradient_by="tick",
            color_by="side",
            color_dict=side_color,
            marker_by="status",
            marker_dict=flash_marker,
            legendgroup=legendgroup,
        )

        if show_lines and len(filtered_data["flash_lines"]) > 0:
            add_connection_lines(
                fig,
                filtered_data["flash_lines"],
                st1="attacker",
                st2="player",
                gradient_by="tick",
                legendgroup=legendgroup,
            )
    # Add kills
    if show_kills and len(filtered_data["kills"]) > 0:
        legendgroup = "Kills"
        add_player_actions(
            fig,
            filtered_data["kills"],
            size=kill_size,
            alpha=kill_alpha,
            gradient_by="tick",
            color_by="side",
            color_dict=side_color,
            marker_by="status",
            marker_dict=kill_marker,
            legendgroup=legendgroup,
        )

        if show_lines and len(filtered_data["kill_lines"]) > 0:
            add_connection_lines(
                fig,
                filtered_data["kill_lines"],
                st1="attacker",
                st2="victim",
                gradient_by="tick",
                legendgroup=legendgroup,
            )

    # Add grenades
    if show_grenades and len(filtered_data["grenades"]) > 0:
        legendgroup = "Grenades"
        add_player_actions(
            fig,
            filtered_data["grenades"],
            size=grenade_size,
            alpha=grenade_alpha,
            gradient_by="throw_tick",
            color_by="side",
            color_dict=side_color,
            marker_by="status",
            marker_dict=grenade_marker,
            legendgroup=legendgroup,
        )

        if show_lines and len(filtered_data["grenade_lines"]) > 0:
            add_connection_lines(
                fig,
                filtered_data["grenade_lines"],
                st1="thrower",
                st2="grenade",
                gradient_by="throw_tick",
                legendgroup=legendgroup,
            )

    # Configure layout
    fig.update_layout(
        title=f"CS:GO Game Actions (Tick: 0 - {max_tick})",
        dragmode="pan",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[{"dragmode": "pan"}], label="Pan", method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "zoom"}], label="Zoom", method="relayout"
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top",
            ),
        ],
        xaxis=dict(
            range=[0, plot_width],
            showgrid=False,
            zeroline=True,
            showticklabels=True,
            title="",
            scaleratio=1,
            autorange=False,
            fixedrange=False,
            constrain="domain",
        ),
        yaxis=dict(
            range=[0, plot_height],
            showgrid=False,
            zeroline=True,
            showticklabels=True,
            title="",
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        height=fig_height,
        margin=dict(l=0, r=0, t=50, b=50),
    )
    fig.update_xaxes(range=[0, plot_width])
    fig.update_yaxes(range=[0, plot_height])

    # Add event count annotation
    event_counts = get_event_counts(filtered_data)
    fig.add_annotation(
        x=0.5,
        y=-0.1,
        xref="paper",
        yref="paper",
        text=f"Events: {event_counts['flashes']} flashes, {event_counts['kills']} kills, {event_counts['grenades']} grenades",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    )

    return fig


def filter_data_by_tick(round_dfs, min_tick, max_tick):
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


def add_player_actions(
    fig,
    df,
    size=10,
    gradient_by="tick",
    color_by=None,
    color_dict=None,
    default_color="viridis",  # Default colormap when color_by is None
    alpha=0.5,
    marker_by=None,
    marker_dict=None,
    default_marker="\u2B24",  # Default marker when marker_by is None
    legendgroup=None,
):
    if df.empty:
        return

    min_grad = df[gradient_by].min()
    max_grad = df[gradient_by].max()

    normalized_grad = (df[gradient_by] - min_grad) / (max_grad - min_grad)

    symbols = [default_marker] * len(df)
    if marker_by is not None:
        symbols = df[marker_by].map(marker_dict)

    cmaps = [default_color] * len(df)
    if color_by is not None:
        cmaps = df[color_by].map(color_dict).to_list()

    mapped_color = [
        plt.get_cmap(cmap)(grad) for (cmap, grad) in zip(cmaps, normalized_grad)
    ]

    colors = [f"rgba({mc[0]}, {mc[1]}, {mc[2]}, {alpha})" for mc in mapped_color]

    if marker_by is not None:
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"] + 1,
                mode="markers",
                marker=dict(
                    color=colors,
                    size=size * 1.2,  # Larger background
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
                name=f"{color_by} Background",
                showlegend=False,  # Don't show in legend
                hoverinfo="skip",  # Don't show hover for background
                legendgroup=legendgroup,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="text",
            text=symbols,
            textfont=dict(
                color=colors,
                size=size,
            ),
            opacity=alpha,
            hovertemplate=f"<br>"
            + "X: %{x}<br>"
            + "Y: %{y}<br>"
            + "Tick: %{customdata}<extra></extra>",
            customdata=df[gradient_by],
            showlegend=True,
            legendgroup=legendgroup,
            name=legendgroup,
        )
    )


def add_connection_lines(
    fig,
    df,
    st1,
    st2,
    gradient_by,
    gradient="viridis",
    linewidth=1,
    alpha=1,
    legendgroup=None,
):
    """Add connection lines between related events"""
    if df.empty:
        return

    min_grad = df[gradient_by].min()
    max_grad = df[gradient_by].max()

    for _, row in df.iterrows():
        cmap = plt.get_cmap(gradient)

        color = cmap((row[gradient_by] - min_grad) / (max_grad - min_grad))
        str_color = f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"

        fig.add_trace(
            go.Scatter(
                x=[row[f"{st1}_x"], row[f"{st2}_x"]],
                y=[row[f"{st1}_y"], row[f"{st2}_y"]],
                mode="lines",
                line=dict(color=str_color, width=linewidth),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=legendgroup,
            )
        )


def get_event_counts(filtered_data):
    """Get counts of different event types"""
    counts = {
        "flashes": len(filtered_data.get("flashes", [])),
        "kills": len(filtered_data.get("kills", [])),
        "grenades": len(filtered_data.get("grenades", [])),
    }
    return counts


def plot_actions_by_max_tick(
    round_dfs,
    max_tick,
    show_loc,
    show_flash,
    show_kills,
    show_grenades,
    flash_alpha,
    kill_alpha,
    grenade_alpha,
    flash_size,
    kill_size,
    grenade_size,
    show_lines,
    transformed_data,
    fig_height,
):
    """Plotly version of the original function"""

    fig = create_plotly_actions_plot(
        round_dfs=round_dfs,
        max_tick=max_tick,
        show_loc=show_loc,
        show_flash=show_flash,
        show_kills=show_kills,
        show_grenades=show_grenades,
        flash_alpha=flash_alpha,
        kill_alpha=kill_alpha,
        grenade_alpha=grenade_alpha,
        flash_size=flash_size,
        kill_size=kill_size,
        grenade_size=grenade_size,
        show_lines=show_lines,
        map_name=transformed_data.get("map", "de_dust2"),
        fig_height=fig_height,
    )

    return fig


def plot_round_timeline_plotly(df):
    colors = [
        "skyblue" if row["ct_win"] == 1 else "lightcoral" for _, row in df.iterrows()
    ]
    labels = df["round_end_reason"]
    rounds = df["round_num"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=rounds,
            y=[1] * len(rounds),
            marker_color=colors,
            text=labels,
            hovertext=labels,
            hoverinfo="text+x",
            textposition="inside",
            insidetextanchor="middle",
            width=0.9,
        )
    )

    fig.update_layout(
        title="Round Timeline",
        xaxis_title="Round Number",
        yaxis=dict(visible=False),
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def plot_scaled_feature_difference(rounds_sum_df: pd.DataFrame):
    """
    Plots percentage difference between CT and T round wins.

    Returns:
        percentage_df (pd.DataFrame): DataFrame of percentage differences.
        fig (plotly.graph_objects.Figure): Plotly bar chart figure.
    """
    # Split by winning side
    ct_win_df = rounds_sum_df[rounds_sum_df["winning_side"] == 1]
    t_win_df = rounds_sum_df[rounds_sum_df["winning_side"] == 0]

    # Select relevant features
    features_to_compare = rounds_sum_df.columns.to_list()
    for col in ["bomb_plant_tick", "winning_side"]:
        if col in features_to_compare:
            features_to_compare.remove(col)

    features_to_compare = [
        col for col in features_to_compare if not col.startswith("round_end_reason_")
    ]

    # Calculate percentage difference
    percentage_df = pd.DataFrame()
    for feature in features_to_compare:
        t_mean = t_win_df[feature].mean()
        ct_mean = ct_win_df[feature].mean()
        if t_mean != 0:
            pct_change = ((ct_mean / t_mean) - 1) * 100
        else:
            pct_change = float("inf") if ct_mean > 0 else float("-inf")
        percentage_df.loc[feature, "%diff"] = pct_change

    # Prepare for plotting
    percentage_df.reset_index(inplace=True)
    percentage_df.rename(columns={"index": "Feature"}, inplace=True)
    percentage_df.sort_values("%diff", ascending=False, inplace=True)
    percentage_df.dropna(how="any", inplace=True)

    # Plot
    fig = px.bar(
        percentage_df,
        x="Feature",
        y="%diff",
        color=percentage_df["%diff"].apply(lambda x: "CT Win" if x > 0 else "T Win"),
        color_discrete_map={"CT Win": "green", "T Win": "red"},
        title="Percentage Difference Between CT and T Win Rounds",
        labels={"%diff": "Percentage Difference (%)"},
    )

    fig.update_layout(
        xaxis_tickangle=45,
        yaxis=dict(gridcolor="lightgray"),
        height=600,
        showlegend=True,
        legend_title="Winning Side",
    )

    return percentage_df, fig


def plot_location_change_analysis(clean_dfs, round_num):

    # Create subplots with 3 rows and 1 column
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "<b>Location Change with Flash</b>",
            "<b>Location Change with Kills</b>",
            "<b>Location Change with Grenade</b>",
        ],
        vertical_spacing=0.15,
        shared_xaxes=True,
    )

    # Preprocessing: Location Change
    sel_loc = (
        clean_dfs["player_frames"].loc[round_num][["tick", "side", "x", "y"]].copy()
    )
    curr_x = sel_loc.iloc[1:]["x"].values
    prev_x = sel_loc.iloc[:-1]["x"].values
    curr_y = sel_loc.iloc[1:]["y"].values
    prev_y = sel_loc.iloc[:-1]["y"].values
    loc_change = [0] + list(np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2))
    sel_loc["loc_change"] = loc_change
    sel_team_loc = sel_loc.groupby(["side", "tick"]).agg(
        avg_loc_change=("loc_change", "mean")
    )

    ct_loc_df = sel_team_loc.loc["CT"].reset_index()
    t_loc_df = sel_team_loc.loc["T"].reset_index()

    # Colors - bright colors for dark theme
    ct_color = "rgba(0, 150, 255, 1)"
    t_color = "rgba(255, 80, 80, 1)"

    # Add traces for all three subplots - Team location changes
    for i in range(1, 4):
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
    for i in range(1, 4):
        max_ct = max(ct_loc_df["avg_loc_change"])
        max_t = max(t_loc_df["avg_loc_change"])
        y_max = max(max_ct, max_t) * 1.1  # Add some padding
        y_ranges.append([0, y_max])

        fig.update_yaxes(range=[0, y_max], row=i, col=1)

    # Subplot 1: Add Flash events - each to its specific subplot
    sel_f = clean_dfs["flashes"].loc[round_num].reset_index()

    flash_events = []
    for attacker_side in ["CT", "T"]:
        flash_ticks = sel_f[sel_f.attacker_side == attacker_side]["tick"]
        color = (
            "rgba(0, 150, 255, 0.8)"
            if attacker_side == "CT"
            else "rgba(255, 80, 80, 0.8)"
        )

        if len(flash_ticks) > 0:
            flash_events.append(
                {"side": attacker_side, "ticks": flash_ticks, "color": color}
            )

            # Add a single legend item for each side's flashes
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{attacker_side} Flash",
                    legendgroup=f"{attacker_side}_flash",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # Add vertical lines for flashes
    for event in flash_events:
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

    # Subplot 2: Add Kill events
    sel_k = clean_dfs["kills"].loc[round_num].reset_index()

    kill_events = []
    for attacker_side in ["CT", "T"]:
        kill_ticks = sel_k[sel_k.attacker_side == attacker_side]["tick"]
        color = (
            "rgba(200, 0, 200, 0.8)"
            if attacker_side == "CT"
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
                row=2,
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
                row=2,
                col=1,
            )

    # Subplot 3: Add Grenade events
    sel_g = clean_dfs["grenades"].loc[round_num].reset_index()

    grenade_events = []
    for thrower_side in ["CT", "T"]:
        grenade_ticks = sel_g[sel_g.thrower_side == thrower_side]["throw_tick"]
        color = (
            "rgba(255, 105, 180, 0.8)"
            if thrower_side == "CT"
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
                row=3,
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
                row=3,
                col=1,
            )

    # Update layout with dark theme
    fig.update_layout(
        height=800,
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


def plot_combined_economy_with_reasons(rounds_sum_df: pd.DataFrame):
    # Reconstruct `round_end_reason` from one-hot
    reason_cols = [
        col for col in rounds_sum_df.columns if col.startswith("round_end_reason_")
    ]
    rounds_sum_df = rounds_sum_df.copy()
    rounds_sum_df["round_end_reason"] = rounds_sum_df[reason_cols].idxmax(axis=1)
    rounds_sum_df["round_end_reason"] = rounds_sum_df["round_end_reason"].str.replace(
        "round_end_reason_", ""
    )

    # Color mapping
    win_reason_colors = {
        "TerroristsWin": "#e74c3c",
        "TargetBombed": "#f39c12",
        "TargetSaved": "#27ae60",
        "CTWin": "#3498db",
        "BombDefused": "#9b59b6",
    }

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
        reason = row["round_end_reason"]
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
