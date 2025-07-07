import plotly.graph_objects as go
from PIL import Image


def plot_map(map_name, fig_height, default_img_dim=1024):
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

    fig.update_layout(
        height=fig_height * 1.2,
        xaxis=dict(
            range=[0, image_width],
            showgrid=False,  # Hide grid for cleaner map background
            zeroline=True,
            showticklabels=True,  # Hide tick labels
            title="X Coordinate",  # Optional title
        ),
        yaxis=dict(
            range=[0, image_height],
            showgrid=False,  # Hide grid
            zeroline=True,
            showticklabels=True,  # Hide tick labels
            scaleanchor="x",  # Ensures aspect ratio is maintained (1:1)
            scaleratio=1,  # Ensures 1:1 aspect ratio
            title="Y Coordinate",  # Optional title
        ),
        # Ensure plot and paper backgrounds are transparent if image is used,
        # otherwise, they might obscure the image.
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the plot area
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background for the paper area
        margin=dict(l=0, r=0, t=40, b=0),  # Adjust margins as needed
    )
    return fig
