import os
import datetime
import numpy as np
import pandas
import pandas as pd
import pytz
from bb_utils.meta import BeeMetaInfo
import bb_utils
from tqdm import tqdm

def load_circadian_df(curta_scratch_path):
    import slurmhelper

    META = BeeMetaInfo()
    data = slurmhelper.SLURMJob("circadiansine5", os.path.join(curta_scratch_path, 'dormagen', 'slurm'))

    circadian_df = []

    for kwargs, results in tqdm(data.items(ignore_open_jobs=True)):
        if results is None:
            continue

        for (subsample, date, bee_id), bee_data in results.items():
            bee_row = dict()
            def add_dict(d, prefix=""):
                for key, val in d.items():
                    if type(val) is not dict:
                        bee_row[prefix + key] = val
                        
                        if key == "parameters":
                            if len(val) == 3:
                                amplitude, phase, offset = val
                                bee_row[prefix + "amplitude"] = amplitude
                                bee_row[prefix + "phase"] = phase
                                bee_row[prefix + "offset"] = offset
                                bee_row[prefix + "base_activity"] = offset - abs(amplitude)
                            elif len(val) == 2:
                                amplitude, phase = val
                                bee_row[prefix + "amplitude"] = amplitude
                                bee_row[prefix + "phase"] = phase
                        if key == "constant_parameters":
                            mean = val[0]
                            bee_row[prefix + "mean"] = mean
                        continue
                    else:
                        add_dict(val, prefix=prefix+key + "_")
            add_dict(bee_data)
            circadian_df.append(bee_row)

    circadian_df = pandas.DataFrame(circadian_df)
    circadian_df.describe()
    circadian_df.subsample.fillna(0, inplace=True)
    circadian_df = circadian_df[circadian_df.date < datetime.datetime(2016, 9, 1, tzinfo=pytz.UTC)]
    circadian_df["is_good_fit"] = (circadian_df.goodness_of_fit > 0.1).astype(np.float)
    circadian_df["is_circadian"] = (circadian_df.p_value < 0.05).astype(np.float)
    circadian_df["well_tested_circadianess"] = (circadian_df.is_circadian * circadian_df.is_good_fit)
    circadian_df = circadian_df[~pandas.isnull(circadian_df.amplitude)]
    circadian_df = circadian_df[~pandas.isnull(circadian_df.r_squared)]
    circadian_df.bee_id = circadian_df.bee_id.astype(np.uint32)
    circadian_df.amplitude = circadian_df.amplitude.astype(np.float32)
    circadian_df.r_squared = circadian_df.r_squared.astype(np.float32)
    circadian_df = circadian_df[circadian_df.n_data_points > np.expm1(9.5)]

    def get_bee_age(bee_id, date):
        date = date.to_pydatetime().replace(tzinfo=None)
        return META.get_age(bb_utils.ids.BeesbookID.from_ferwar(bee_id), date).days
    circadian_df["age"] = circadian_df[["bee_id", "date"]].apply(
        lambda f: get_bee_age(*f), axis=1)

    circadian_df = circadian_df[circadian_df.age < 55]
    circadian_df = circadian_df[circadian_df.subsample == 0]

    return circadian_df


def load_basics_df(path):
    import dill

    basics = dill.load(open(path, 'rb'))

    basics = pd.DataFrame([[b0] + list(b1) for b0, b1 in basics if b1 is not None],
                        columns=['datetime', 'age', 'bee_id', 'mean_movement_speed', 
                                'total_movement_distance', 'proportion_active', 
                                'mean_angular_speed', 'mean_exit_distance', 'mean_turtosity'])

    basics.age = basics.age.apply(lambda d: d.days).astype(np.float)

    return basics


def evaluate_regression(combined, model_name, target_vars=None, id_subset_score=None, day_subset_score=None, n_models=1):
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn import metrics
    import statsmodels.api as sm

    if target_vars is None:
        target_vars = ('power', 'amplitude', 'age', 'mean_movement_speed', 
                       'total_movement_distance', 'proportion_active', 
                       'mean_angular_speed', 'mean_exit_distance', 'mean_turtosity',
                       'is_circadian', 'well_tested_circadianess')

    results = {}
    results['model'] = model_name

    if id_subset_score is None:
        use_rows = [True for _ in range(len(combined))]
    else:
        use_rows = [bee_id in id_subset_score for bee_id in combined.bee_id]

    if day_subset_score is not None:
        use_rows = [(use and (day in day_subset_score)) for day, use in zip(combined.day, use_rows)]

    for target_var in tqdm(target_vars):
        day_onehot = pd.DataFrame(OneHotEncoder(sparse=False, categories='auto').fit_transform(combined.day[:, None]),
                                  columns=['day_{}'.format(d) for d in combined.day.unique()])
        
        factor_names = [c for c in combined.columns if c.startswith('f_')]
        emb_names = [c for c in combined.columns if c.startswith('e_')]

        combined_ = np.concatenate((
            combined[factor_names], 
            combined[emb_names], 
            day_onehot), axis=1)

        feature_names = factor_names + emb_names + list(day_onehot.columns)

        features_scaled = pd.DataFrame(
            sm.add_constant(RobustScaler().fit_transform(combined_)), columns=['const'] + feature_names)

        if 'circadian' in target_var:
            scores = []
            for _ in range(n_models):
                model = sm.Logit(combined.reset_index()[target_var], features_scaled).fit_regularized(
                    alpha=1, disp=False, trim_mode='off', qc_tol=1)
                scores.append(metrics.roc_auc_score(combined.loc[use_rows][target_var], model.predict(features_scaled[use_rows])))

            results[target_var + '_roc_auc_mean'] = np.mean(scores)
            if n_models > 1:
                results[target_var + '_roc_auc_std'] = np.std(scores)
        else:
            scores = []
            for _ in range(n_models):
                model = sm.OLS(combined.reset_index()[target_var], features_scaled).fit()
                scores.append(metrics.r2_score(combined.loc[use_rows][target_var], model.predict(features_scaled[use_rows])))

            results[target_var + '_r2_mean'] = np.mean(scores)
            if n_models > 1:
                results[target_var + '_r2_std'] = np.std(scores)

        
    results['data_subset'] = [combined[['bee_id', 'date']]]

    return pd.DataFrame(results)


def evaluate_auc_over_time(combined, model_name, target_var='is_circadian'):
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn import metrics
    import statsmodels.api as sm

    day_onehot = pd.DataFrame(OneHotEncoder(sparse=False, categories='auto').fit_transform(combined.day[:, None]),
                                columns=['day_{}'.format(d) for d in combined.day.unique()])
    
    factor_names = [c for c in combined.columns if c.startswith('f_')]
    emb_names = [c for c in combined.columns if c.startswith('e_')]

    combined_ = np.concatenate((
        combined[factor_names], 
        combined[emb_names], 
        day_onehot), axis=1)

    feature_names = factor_names + emb_names + list(day_onehot.columns)

    features_scaled = pd.DataFrame(
        sm.add_constant(RobustScaler().fit_transform(combined_)), columns=['const'] + feature_names)

    model = sm.Logit(combined.reset_index()['is_circadian'], features_scaled).fit_regularized(
        alpha=1, disp=False, trim_mode='off', qc_tol=1)

    results_by_day = []

    for day, group_indices in combined.groupby('day').indices.items():
        auc_score = metrics.roc_auc_score(combined.iloc[group_indices]['is_circadian'], 
                                        model.predict(features_scaled.iloc[group_indices]))
        
        results_by_day.append((day, auc_score))
        
    return np.array(results_by_day)
