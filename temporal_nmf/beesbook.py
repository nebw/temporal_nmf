import datetime
import numba
import numpy as np
import pandas as pd
import psycopg2

from temporal_nmf import evaluation

DEFAULT_QUERY_PREFIX = """
SET geqo_effort to 10;
"""


def combine_data(factor_df, basics_path, circadian_path):
    circadian_df = evaluation.load_circadian_df(circadian_path)
    basics_df = evaluation.load_basics_df(basics_path)

    combined = pd.merge(factor_df, circadian_df, how="inner", on=("bee_id", "age"))
    combined = pd.merge(combined, basics_df, how="inner", on=("bee_id", "age"))

    return combined


def get_proximity_interactions(
    ts_start, ts_end, connect_str, table_name, max_dist=150.0, query_prefix=DEFAULT_QUERY_PREFIX
):
    query = """
    {query_prefix}

    SELECT * FROM (
    SELECT
        A.bee_id, B.bee_id,
        (|/((A.x_pos - B.x_pos) ^ 2 + (A.y_pos - B.y_pos) ^ 2)) as distance,
        A.cam_id, A.timestamp
    FROM {table_name} A
    INNER JOIN {table_name} B ON A.timestamp = B.timestamp
    WHERE
        A.timestamp >= %s
        AND A.timestamp <= %s
        AND B.timestamp >= %s
        AND B.timestamp <= %s
        AND A.bee_id < B.bee_id
    ) _subquery
    WHERE distance < %s
    ORDER BY timestamp ASC; """

    with psycopg2.connect(connect_str) as conn:
        interaction_df = pd.read_sql_query(
            query, conn, params=(ts_start, ts_end, ts_start, ts_end, max_dist), coerce_float=False,
        )

    return interaction_df


@numba.njit(parallel=False)
def crosstab(ids, adj):
    for i in range(len(ids)):
        a = ids[i, 0]
        b = ids[i, 1]

        adj[min(a, b), max(a, b)] = 1

    return adj


def create_interaction_list(interaction_df, num_individuals=8829, fps=6, ringbuffer_size=10):
    ts = interaction_df.timestamp.min()

    rbs = ringbuffer_size
    rbs_hp = (rbs // 2) + 1

    # cumulative interactions over whole period
    previous_interactions = np.zeros((num_individuals, num_individuals), dtype=np.bool)
    # frame sliding window
    interaction_ringbuffer = np.zeros((rbs, num_individuals, num_individuals), dtype=np.bool)
    # cumulative interactions for all cameras within a 1/fps frame period
    # == 1 frame combined for all cameras
    current_interactions = np.zeros((num_individuals, num_individuals), dtype=np.bool)

    interval_counter = 0

    events = []

    for timestamp, group in interaction_df.groupby("timestamp"):
        # still within current time interval
        if (timestamp - ts) < datetime.timedelta(milliseconds=int(900 / fps)):
            pass
        # end of current time interval
        else:
            # count as interaction if more than half of rbs consecutive frames had interactions
            # == median filter over temporal dimension with kernel size rbs
            if interval_counter >= rbs:
                new_interactions = np.sum(interaction_ringbuffer, axis=0) > rbs_hp
                stopped_interactions = (
                    previous_interactions.astype(np.int) - new_interactions.astype(np.int)
                ) == 1

                if stopped_interactions.sum() > 0:
                    for bee_id_a, bee_id_b in np.argwhere(stopped_interactions):
                        events.append((timestamp, bee_id_a, bee_id_b))

                previous_interactions = new_interactions

            interaction_ringbuffer[interval_counter % rbs] = current_interactions.copy()

            # new time interval => reset adjacency matrix and timestamp
            current_interactions = np.zeros_like(current_interactions)
            ts = group.timestamp.min()

            interval_counter += 1

        # interaction adjacency matrix
        adj = np.zeros((num_individuals, num_individuals), dtype=np.uint32)
        adj = crosstab(group["bee_id"].values, adj)

        # logical or => accumulate interactions from different cameras
        # for current time interval (~1/fps of a second)
        current_interactions |= adj > 0

    return events
