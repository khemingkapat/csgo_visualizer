import plotly.graph_objects as go
import pandas as pd


def plot_round_timeline_plotly(df: pd.DataFrame) -> go.Figure:
    colors = [
        "skyblue" if row["ct_win"] == 1 else "lightcoral" for _, row in df.iterrows()
    ]
    labels = df["reason"]
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


def _plot_scaled_feature_difference(percentage_df: pd.DataFrame) -> go.Figure:
    """
    Plots percentage difference between CT and T round wins.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly bar chart figure.
    """

    fig = go.Figure()

    colors = ["green" if val > 0 else "red" for val in percentage_df["%diff"]]
    labels = ["CT Win" if val > 0 else "T Win" for val in percentage_df["%diff"]]

    fig.add_trace(
        go.Bar(
            x=percentage_df["Feature"],
            y=percentage_df["%diff"],
            marker_color=colors,
            name="",
            customdata=labels,
            hovertemplate="<b>%{x}</b><br>%{customdata}<br>%{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_tickangle=45,
        yaxis=dict(gridcolor="lightgray"),
        height=600,
        showlegend=True,
        legend_title="Winning Side",
    )

    return fig
