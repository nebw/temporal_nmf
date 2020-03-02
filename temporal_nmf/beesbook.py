import pandas as pd

from temporal_nmf import evaluation


def combine_data(factor_df, basics_path, circadian_path):
    circadian_df = evaluation.load_circadian_df(circadian_path)
    basics_df = evaluation.load_basics_df(basics_path)

    combined = pd.merge(factor_df, circadian_df, how='inner', on=('bee_id', 'age'))
    combined = pd.merge(combined, basics_df, how='inner', on=('bee_id', 'age'))

    return combined
