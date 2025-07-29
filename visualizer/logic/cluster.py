import numpy as np
import igraph as ig
import leidenalg as la
import pandas as pd
from utils.functions import filter_nan, eu_dist


def cluster_player_communities(
    tf_loc: pd.DataFrame,
    distance_threshold: float = 100,
    epsilon: float = 1e-6,
    sampling_rate: int = 20,
):
    """
    Detect player communities for each tick using proximity-based graph clustering.

    Parameters:
    -----------
    tf_loc : pandas.DataFrame
    DataFrame containing player location data with columns:
        - 'tick': time tick
        - 'steam_id': unique player identifier
        - 'x', 'y': player coordinates
        - 'side': player team/side
    distance_threshold : float, default=100
        Maximum distance between players to create an edge
    epsilon : float, default=1e-6
        Small value to avoid division by zero in weight calculation
    leiden_resolution : float, default=1.0
        Resolution parameter for Leiden algorithm (currently unused)
    sampling_rate : int, default=50
        Sample every nth tick to reduce computation

    Returns:
    --------
    DataFrame as a copy of `tf_loc` but with column "c" as cluster
    """

    tick_partitions = {}
    all_ticks = sorted(tf_loc["tick"].unique())
    all_ticks = all_ticks[::sampling_rate]

    # --- Loop through each tick ---
    for tick in all_ticks:
        tick_df = tf_loc[tf_loc["tick"] == tick].copy()
        players_in_tick = np.unique(tick_df["steamid"])

        # --- Prepare edge data for igraph.Graph.TupleList ---
        # Store (source_steam_id, target_steam_id, weight) tuples
        edges_and_weights_for_igraph = []

        pos_data = (
            tick_df.set_index("steamid").loc[:, ["x", "y", "side"]].to_dict("index")
        )

        # Calculate distances between all pairs of players
        for i in range(len(players_in_tick)):
            player_a_id = players_in_tick[i]

            # Ensure player_a is present in pos_data for this tick
            if player_a_id not in pos_data:
                continue

            for j in range(i + 1, len(players_in_tick)):
                player_b_id = players_in_tick[j]

                # Ensure player_b is present and on the same side as player_a
                if (
                    player_b_id not in pos_data
                    or pos_data[player_a_id]["side"] != pos_data[player_b_id]["side"]
                ):
                    continue

                # Calculate Euclidean distance
                coords_a = np.array(
                    [pos_data[player_a_id]["x"], pos_data[player_a_id]["y"]]
                )
                coords_b = np.array(
                    [pos_data[player_b_id]["x"], pos_data[player_b_id]["y"]]
                )
                dist = np.linalg.norm(coords_a - coords_b)

                # Add edge only if within distance_threshold
                if dist < distance_threshold:
                    weight = 1.0 / (dist + epsilon)  # Inverse distance as weight
                    edges_and_weights_for_igraph.append(
                        (player_a_id, player_b_id, weight)
                    )

        # --- Create igraph graph for the current tick ---
        current_tick_partition_dict = {}

        # Handle case with no edges (isolated players)
        if not edges_and_weights_for_igraph:
            g_ig_tick = ig.Graph(directed=False)
            g_ig_tick.add_vertices(
                list(players_in_tick)
            )  # Add players as isolated vertices
        else:
            # Create graph from edge list with weights
            g_ig_tick = ig.Graph.TupleList(
                edges_and_weights_for_igraph, directed=False, weights=True
            )

            # Ensure all players are in the graph, even those with no connections
            existing_names = set(v["name"] for v in g_ig_tick.vs)
            for player_id in players_in_tick:
                if player_id not in existing_names:
                    g_ig_tick.add_vertex(name=player_id)

        # --- Apply Leiden algorithm ---
        if g_ig_tick.vcount() > 0:  # Ensure there are players in the graph
            if (
                g_ig_tick.ecount() > 0
            ):  # Ensure there are edges for meaningful communities
                partition_ig = la.find_partition(
                    g_ig_tick,
                    la.ModularityVertexPartition,
                    weights="weight",
                )
            else:
                # If no edges, each player is their own community
                partition_ig = ig.VertexClustering(
                    g_ig_tick, membership=list(range(g_ig_tick.vcount()))
                )

            # Convert igraph's partition membership back to steam_id dictionary
            current_tick_partition_dict = {
                g_ig_tick.vs[i]["name"]: partition_ig.membership[i]
                for i in range(g_ig_tick.vcount())
            }

        tick_partitions[tick] = current_tick_partition_dict

    result = tf_loc.reset_index().drop(["round_num"], axis=1).set_index("tick")
    for tick in all_ticks:
        result.loc[tick, "c"] = result.loc[tick, "steamid"].map(tick_partitions[tick])

    return result


def transform_to_cluster(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        df.reset_index()
        .groupby(["tick", "side", "c"], observed=True)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            size=("steamid", "count"),
            start=("tick", "mean"),
            end=("tick", "mean"),
        )
    )

    result.loc[:, ["p1", "p2", "p3", "p4", "p5"]] = pd.NA

    for tick, side, c in result.index:
        c_players = df[(df.index == tick) & (df.side == side) & (df.c == c)]["steamid"]
        result.loc[(tick, side, c), [f"p{i}" for i in range(1, len(c_players) + 1)]] = (
            np.array(c_players)
        )

    result["show"] = True

    return result


def get_cluster_lineage(
    df: pd.DataFrame,
    tick: int,
    side: str,
    cluster: int,
    time_points: list[int] = [],
    distance_threshold: float = 100,
) -> list:
    """
    Trace the lineage of a cluster back through time based on exact player match
    """
    result = []
    if not time_points:
        time_points = df.index.get_level_values(0).unique().to_list()

    if tick == time_points[0]:
        return result

    current_data = df.loc[(tick, side, cluster)]
    current_players = set(filter_nan(current_data[5:-1]))  # Players are in positions 3+

    if current_data["start"] != current_data["end"]:
        prev_tick = current_data["start"]
    else:
        prev_tick = time_points[time_points.index(tick) - 1]
    prev_clusters = df.loc[(prev_tick, side)]

    prev_points = []
    # Find previous cluster with exact player match
    for prev_cluster_id, prev_data in prev_clusters.iterrows():
        prev_players = set(filter_nan(prev_data[5:-1]))

        if prev_players == current_players:
            prev_points.append((prev_tick, side, prev_cluster_id))
            if eu_dist(*prev_data[:2], *current_data[:2]) < distance_threshold:
                result.extend(
                    get_cluster_lineage(
                        df, prev_tick, side, prev_cluster_id, time_points=time_points
                    )
                )
        elif current_players.intersection(prev_players):
            prev_points.append((prev_tick, side, prev_cluster_id))

    if prev_points:
        result.append(prev_points)
    return result
