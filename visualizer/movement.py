import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from utils.symbols import side_color
from .logic.cluster import get_cluster_lineage


def plot_chained_vector(
    fig: go.Figure, chain_vecs: list[dict], alpha: float = 0.5
) -> None:
    """
    Plot chained vectors as arrows with labels on a Plotly figure.

    Args:
        fig: Plotly figure object to add the vectors to
        chain_vecs: List of dictionaries containing vector information
                   Each dict should have: 'start_pos', 'end_pos', 'tick', 'name'
    """
    if not chain_vecs:
        return

    # Create colormap normalization based on tick values
    tick_values = [vector["tick"] for vector in chain_vecs]

    norm = lambda val: (
        (val - min(tick_values)) / (max(tick_values) - min(tick_values))
        if (max(tick_values) - min(tick_values)) != 0
        else 1
    )
    colormap = plt.get_cmap("viridis")

    for vector in chain_vecs:
        # Get color from colormap
        color_rgba = colormap(norm(vector["tick"]))
        color_hex = f"rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alpha})"

        start_pos = vector["start_pos"]
        end_pos = vector["end_pos"]

        # Calculate vector components
        vec_dx = end_pos[0] - start_pos[0]
        vec_dy = end_pos[1] - start_pos[1]

        # Add arrowhead using annotation
        fig.add_annotation(
            x=end_pos[0],
            y=end_pos[1],
            ax=start_pos[0],
            ay=start_pos[1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=4,
            arrowcolor=color_hex,
            showarrow=True,
            text="",
            opacity=0.8,
        )

        # Calculate text position with offset
        text_offset_factor = 15

        # Adjust offset based on direction of the vector
        text_x_offset = text_offset_factor * (1 if vec_dx >= 0 else -1)
        text_y_offset = text_offset_factor * (1 if vec_dy >= 0 else -1)

        # If the vector is mostly horizontal, adjust y-offset more. If vertical, adjust x-offset more.
        if abs(vec_dx) > abs(vec_dy):  # More horizontal
            text_y_offset = (
                text_offset_factor * (1 if vec_dy >= 0 else -1) * 0.5
            )  # Less y-offset
        else:  # More vertical
            text_x_offset = (
                text_offset_factor * (1 if vec_dx >= 0 else -1) * 0.5
            )  # Less x-offset

        # Add text label
        fig.add_annotation(
            x=end_pos[0] + text_x_offset,
            y=end_pos[1] + text_y_offset,
            text=vector["name"],
            showarrow=False,
            font=dict(color=color_hex, size=10, family="Arial Black"),
            xanchor="center",
            yanchor="middle",
        )


def plot_community(fig: go.Figure, df: pd.DataFrame, alpha: float = 0.5):
    """
    Convert matplotlib scatter plot with connections to Plotly graph_objects

    Parameters:
    fig: go.Figure - Plotly figure object
    df: DataFrame - renamed from new_result, should have MultiIndex with (tick, side, c)
    """

    # Define color mappings (you may need to adjust these based on your actual color scheme)

    # Define marker styles mapping (adjust symbols as needed)
    marker_styles = ["circle", "x", "triangle-up", "square", "star"]

    # Create colorscale normalization (you'll need to define your norm function)
    # This assumes you have a way to normalize tick values
    tick_values = df.index.get_level_values("tick").unique().to_list()

    norm = lambda val: (val - min(tick_values)) / (max(tick_values) - min(tick_values))

    # Process each row in the dataframe
    for (tick, side, c), row in df.iterrows():
        x, y, size = row.iloc[0], row.iloc[1], row.iloc[2]
        show = row.iloc[-1]  # assuming 'show' is the last column

        if show:
            # Get color for this point
            color_rgba = plt.get_cmap(side_color[side])(norm(tick))
            color_hex = f"rgba({int(color_rgba[0]*255)},{int(color_rgba[1]*255)},{int(color_rgba[2]*255)},{alpha})"

            # Get marker style
            marker_symbol = (
                marker_styles[int(size) - 1]
                if int(size) - 1 < len(marker_styles)
                else "circle"
            )

            # Calculate marker size
            marker_size = 20 + size

            # Add scatter point
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        color=color_hex,
                        symbol=marker_symbol,
                        size=marker_size,
                        line=dict(width=1),
                    ),
                    name=f"{side}_{tick}_{c}",
                    showlegend=False,
                    hovertemplate=f"Tick: {tick}<br>Side: {side}<br>Cluster: {c}<br>Size: {size}<extra></extra>",
                )
            )

            # Get lineage for connections (you'll need to implement get_cluster_lineage)
            lineage = get_cluster_lineage(df, tick, side, c)
            if lineage:
                lineage = lineage[0]

                for prev_tick, prev_side, prev_c in list(lineage):
                    prev_data = df.loc[(prev_tick, prev_side, prev_c)]
                    prev_x, prev_y = prev_data.iloc[0], prev_data.iloc[1]

                    # Add arrow annotation
                    fig.add_annotation(
                        x=x,
                        y=y,
                        ax=prev_x,
                        ay=prev_y,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=color_hex,
                        opacity=0.5,
                        standoff=5,
                    )
