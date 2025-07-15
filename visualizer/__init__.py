import pandas as pd
from .action_location import (
    plot_chained_vector,
    plot_map,
    create_plotly_actions_plot,
    get_event_counts,
    add_action_connection,
    add_player_action,
    plot_community,
)
from .overview import plot_round_timeline_plotly, _plot_scaled_feature_difference
from .action_timeline import plot_location_change_analysis
from .round_economy import plot_combined_economy_with_reasons
from .logic.value_diff import _get_value_difference
from .logic.filter import filter_data_by_tick
from .logic.pca import chain_pc_vectors, get_side_df, temporal_pca
from .logic.cluster import (
    cluster_player_communities,
    transform_to_cluster,
    get_cluster_lineage,
)


class Visualizer:

    # overview
    plot_round_timeline_plotly = staticmethod(plot_round_timeline_plotly)
    plot_scaled_feature_difference = staticmethod(_plot_scaled_feature_difference)
    get_value_difference = staticmethod(_get_value_difference)

    # action location
    plot_map = staticmethod(plot_map)
    add_player_action = staticmethod(add_player_action)
    add_action_connection = staticmethod(add_action_connection)
    create_plotly_actions_plot = staticmethod(create_plotly_actions_plot)
    get_event_counts = staticmethod(get_event_counts)
    filter_data_by_tick = staticmethod(filter_data_by_tick)

    # action timeline
    plot_location_change_analysis = staticmethod(plot_location_change_analysis)

    # round economy
    plot_combined_economy_with_reasons = staticmethod(
        plot_combined_economy_with_reasons
    )

    # pca result
    get_side_df = staticmethod(get_side_df)
    temporal_pca = staticmethod(temporal_pca)
    chain_pc_vectors = staticmethod(chain_pc_vectors)
    plot_chained_vector = staticmethod(plot_chained_vector)

    # clustering
    cluster_player_communities = staticmethod(cluster_player_communities)
    transform_to_cluster = staticmethod(transform_to_cluster)
    get_cluster_lineage = staticmethod(get_cluster_lineage)

    @staticmethod
    def apply_cluster_community(transformed_loc: pd.DataFrame) -> pd.DataFrame:
        tf_loc = transformed_loc.loc[:, ["tick", "side", "steam_id", "x", "y"]]
        clustered_players_df = Visualizer.cluster_player_communities(tf_loc)
        cluster_df = Visualizer.transform_to_cluster(clustered_players_df)

        to_delete_indices = set()

        for tick, side, c in cluster_df.index[::-1]:
            idx = (tick, side, c)
            lineage = get_cluster_lineage(cluster_df, tick, side, c)

            if lineage and len(lineage) > 1:
                first_group_cluster = cluster_df.loc[lineage[0][0]]
                cluster_df.loc[idx, "start"] = first_group_cluster.start

                to_delete_indices.update([i[0] for i in lineage[1:]])

        mask = ~cluster_df.index.isin(to_delete_indices)
        cluster_df["show"] = mask
        return cluster_df

    plot_community = staticmethod(plot_community)
