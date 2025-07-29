import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from utils.symbols import *
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt


def plot_map(fig: Figure, map_name: str, default_img_dim: int = 1024):
    map_image_path = f".awpy/maps/{map_name}.png"

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


def create_plotly_actions_plot(
    filtered_data: dict[str, pd.DataFrame],
    map_name: str,
    show_loc: bool = True,
    show_smoke: bool = True,
    show_inferno: bool = True,
    show_kills: bool = True,
    show_grenades: bool = True,
    loc_alpha=0.5,
    smoke_alpha: float = 0.7,
    inferno_alpha: float = 0.7,
    kill_alpha: float = 0.7,
    grenade_alpha: float = 0.7,
    smoke_size: int = 10,
    inferno_size: int = 10,
    kill_size: int = 10,
    grenade_size: int = 10,
    show_lines: bool = True,
    fig_height: int = 800,
):
    """
    Create a Plotly figure with CS:GO game actions filtered by max tick.
    This function creates a dynamic plot that can be updated without full reload.

    Parameters:
    -----------
    filtered_data : dict
        Dictionary containing DataFrames for different game actions
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
    plot_map(fig, map_name)

    # Filter data by tick range

    # Add player locations
    if show_loc and len(filtered_data["locations"]) > 0:
        add_player_action(
            fig,
            filtered_data["locations"],
            alpha=loc_alpha,
            gradient_by="tick",
            color_by="side",
            color_dict=side_color,
            legendgroup="Locations",
        )

    # Add flashes
    if show_smoke and len(filtered_data["smokes"]) > 0:
        legendgroup = "Smoke"
        add_player_action(
            fig,
            filtered_data["smokes"],
            size=smoke_size,
            alpha=smoke_alpha,
            gradient_by="start_tick",
            color_by="side",
            color_dict=side_color,
            marker_by="status",
            marker_dict=smoke_marker,
            legendgroup=legendgroup,
        )

        if show_lines and len(filtered_data["smoke_lines"]) > 0:
            add_action_connection(
                fig,
                filtered_data["smoke_lines"],
                st1="thrower",
                st2="smoke",
                gradient_by="start_tick",
                legendgroup=legendgroup,
            )
    if show_inferno and len(filtered_data["infernos"]) > 0:
        legendgroup = "Infernos"
        add_player_action(
            fig,
            filtered_data["infernos"],
            size=inferno_size,
            alpha=inferno_alpha,
            gradient_by="start_tick",
            color_by="side",
            color_dict=side_color,
            marker_by="status",
            marker_dict=inferno_marker,
            legendgroup=legendgroup,
        )

        if show_lines and len(filtered_data["inferno_lines"]) > 0:
            add_action_connection(
                fig,
                filtered_data["inferno_lines"],
                st1="thrower",
                st2="inferno",
                gradient_by="start_tick",
                legendgroup=legendgroup,
            )
    # Add kills
    if show_kills and len(filtered_data["kills"]) > 0:
        legendgroup = "Kills"
        add_player_action(
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
            additional_data="name",
        )

        if show_lines and len(filtered_data["kill_lines"]) > 0:
            add_action_connection(
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
        add_player_action(
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
            additional_data="type",
        )

        if show_lines and len(filtered_data["grenade_lines"]) > 0:
            add_action_connection(
                fig,
                filtered_data["grenade_lines"],
                st1="thrower",
                st2="grenade",
                gradient_by="throw_tick",
                legendgroup=legendgroup,
            )

    # Configure layout
    fig.update_layout(
        title="CS:GO Game Actions",
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
            bgcolor="rgba(255,255,255,0.0)",
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


def add_player_action(
    fig: Figure,
    df: pd.DataFrame,
    size: int = 10,
    gradient_by: str = "tick",
    color_by: str | None = None,
    color_dict: dict[str, str] = {},
    default_color: str = "viridis",  # Default colormap when color_by is None
    alpha: float = 0.5,
    marker_by: str | None = None,
    marker_dict: dict[str, str] = {},
    default_marker: str = "\u2b24",  # Default marker when marker_by is None
    legendgroup: str | None = None,
    additional_data: str = "",
):
    if df.empty:
        return

    min_grad = df[gradient_by].min()
    max_grad = df[gradient_by].max()

    normalized_grad = (df[gradient_by] - min_grad) / (max_grad - min_grad)

    symbols = [default_marker] * len(df)
    if marker_by:
        symbols = df[marker_by].map(lambda x: marker_dict[x])

    cmaps = [default_color] * len(df)
    if color_by:
        cmaps = df[color_by].map(lambda x: color_dict[x]).to_list()

    mapped_color = [
        plt.get_cmap(cmap)(grad) for (cmap, grad) in zip(cmaps, normalized_grad)
    ]

    colors = [f"rgba({mc[0]}, {mc[1]}, {mc[2]}, {1})" for mc in mapped_color]

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
                opacity=alpha,
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
            hovertemplate=(
                f"<br>"
                + "X: %{x}<br>"
                + "Y: %{y}<br>"
                + "Tick: %{customdata[0]}<br>"
                + ("Info: %{customdata[1]}" if additional_data else "")
                + "<extra></extra>"
            ),
            customdata=df[[gradient_by] + [additional_data] * bool(additional_data)],
            showlegend=True,
            legendgroup=legendgroup,
            name=legendgroup,
        )
    )


def add_action_connection(
    fig: Figure,
    df: pd.DataFrame,
    st1: str,
    st2: str,
    gradient_by: str,
    gradient: str = "viridis",
    linewidth: int = 1,
    alpha: float = 1,
    legendgroup: str | None = None,
):
    """Add connection lines between related events"""
    if df.empty:
        return

    min_grad = df[gradient_by].min()
    max_grad = df[gradient_by].max()

    for _, row in df.iterrows():
        cmap = plt.get_cmap(gradient)

        val = row.get(gradient_by, 0)
        if (pd.notna(val)) and (max_grad != min_grad):
            normalized = (val - min_grad) / (max_grad - min_grad)
        else:
            normalized = 0.5
        color = cmap(normalized)

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


def get_event_counts(filtered_data: dict[str, pd.DataFrame]):
    """Get counts of different event types"""
    counts = {
        "flashes": len(filtered_data.get("flashes", [])),
        "kills": len(filtered_data.get("kills", [])),
        "grenades": len(filtered_data.get("grenades", [])),
    }
    return counts
