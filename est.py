import numpy as np
import pandas as pd
import scipy.stats
import sys

from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold


# i: index of dataset
# k: number of folds
# S: number of repeat of DML
def estimate(i, k, S, robust=False):
    # load and reorder the data
    path1 = "./data/practice/acic_practice_"
    path2 = "./data/practice_year/acic_practice_year_"
    fnm1 = path1 + str(i).zfill(4) + ".csv"
    fnm2 = path2 + str(i).zfill(4) + ".csv"
    # data_x = pd.read_csv("./data/practice/acic_practice_0001.csv")
    # data_y = pd.read_csv("./data/practice_year/acic_practice_year_0001.csv")
    data_x = pd.read_csv(fnm1)
    data_y = pd.read_csv(fnm2)
    data_y = pd.concat([data_y[data_y['year'] == j] for j in range(1, 5)])
    # data_y = data_y.reset_index(drop=True)

    # preprocessing for tree models
    ct = ['X2', 'X4']
    for c in ct:
        data_x[c] = data_x[c].astype('category')

    X = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
    V = ['V1_avg', 'V2_avg', 'V3_avg', 'V4_avg', 'V5_A_avg', 'V5_B_avg', 'V5_C_avg']
    XV = V + X
    Y = ['Y']
    T = ['Z']

    # defince ML model objects, to estimate e(X) and g_0(V,X)
    prob_m = LGBMClassifier(objective='binary',
                            is_unbalance=True,
                            # metric = 'log_loss',
                            metric='binary_logloss,auc',
                            max_depth=8,
                            num_leaves=20,
                            learning_rate=0.1,
                            # feature_fraction = 0.7,
                            min_child_samples=20,
                            min_child_weight=0.001,
                            # bagging = 1,
                            # subsample_freq = 2,
                            reg_alpha=0.002,
                            reg_lambda=10,
                            cat_smooth=0,
                            n_estimators=200,
                            )

    mean_m = LGBMRegressor(is_unbalance=True,
                           # metric = 'log_loss',
                           metric='l2',
                           max_depth=8,
                           num_leaves=20,
                           learning_rate=0.1,
                           # feature_fraction = 0.7,
                           min_child_samples=20,
                           min_child_weight=0.001,
                           # bagging = 1,
                           # subsample_freq = 2,
                           reg_alpha=0.002,
                           reg_lambda=10,
                           cat_smooth=0,
                           n_estimators=200,
                           )

    # a list of ndarrays to store results
    tmp = np.zeros((15, 2, S))
    layer = 0

    # record lengths of sub-dataframes
    df_lens = [1000, 500, 500]

    for _ in range(S):
        # create k-fold indices
        kf = KFold(n_splits=k, shuffle=True)
        indices = []
        for train_index, test_index in kf.split(data_x):
            indices.append((train_index, test_index))

        # propensity score
        data_x = data_x.assign(ps=0)
        for train_index, test_index in indices:
            prob_m.fit(X=data_x[X].iloc[train_index], y=data_y['Z'].iloc[train_index].values.ravel())
            data_x.loc[test_index, 'ps'] = prob_m.predict_proba(X=data_x[X].iloc[test_index])[:, 1]

        df = pd.merge(data_x, data_y, on='id.practice')
        df = pd.concat([df[df['year'] == j] for j in range(1, 5)])
        df = df.reset_index(drop=True)

        # year average, cross-fitting
        df = df.assign(ft=0)
        for train_index, test_index in indices:
            for inc in [0, 500, 1000, 1500]:
                train = train_index + inc
                test = test_index + inc
                if inc >= 1000:
                    tr = df['Z'].iloc[train]
                    train = train[tr == 0]
                df.loc[test, 'ft'] = np.average(df['Y'].iloc[train], weights=df['n.patients'].iloc[train])
        # print(df[['ft','year']])
        # print(df.loc[[1,2,3]])
        # print(df.iloc[[1,2,3]])

        # g_0(X,V)
        df = df.assign(g0=0)
        for train_index, test_index in indices:
            train = np.concatenate((train_index, train_index + 500, train_index + 1000, train_index + 1500))
            test = np.concatenate((test_index, test_index + 500, test_index + 1000, test_index + 1500))
            tr = df['Z'].iloc[train]
            tr = df['Z']
            xx = df[XV].iloc[train]
            yy = (df['Y'] - df['ft']).iloc[train]
            sw = df['n.patients'].iloc[
                np.concatenate((train_index, train_index + 500, train_index + 1000, train_index + 1500))]

            # use all data of year 1,2; untreated groups of year 3,4
            xx = xx.loc[tr == 0]
            yy = yy.loc[tr == 0]
            sw = sw.loc[tr == 0]
            mean_m.fit(X=xx, y=yy, sample_weight=sw)
            df.loc[test, 'g0'] = mean_m.predict(df[XV].iloc[test])
            # print(df['g0'])

        # ATTEs
        onerep = np.zeros((15, 2))
        qt = scipy.stats.norm.ppf(0.95)
        row = 0

        # all, year 3, year 4
        if robust:
            df = df[2 * df['Z'] * df['ps'] - df['Z'] -df['ps'] + 1 > 0.1]
        df_3 = df[df['year'] == 3]
        df_4 = df[df['year'] == 4]
        df_34 = df[df['year'] >= 3]

        psi_3 = df_3['Z'] * (df_3['Y'] - df_3['g0'] - df_3['ft']) - df_3['ps'] / (1 - df_3['ps']) * (1 - df_3['Z']) * (
                    df_3['Y'] - df_3['g0'] - df_3['ft'])
        psi_4 = df_4['Z'] * (df_4['Y'] - df_4['g0'] - df_4['ft']) - df_4['ps'] / (1 - df_4['ps']) * (1 - df_4['Z']) * (
                    df_4['Y'] - df_4['g0'] - df_4['ft'])
        psi_34 = df_34['Z'] * (df_34['Y'] - df_34['g0'] - df_34['ft']) - df_34['ps'] / (1 - df_34['ps']) * (
                    1 - df_34['Z']) * (df_34['Y'] - df_34['g0'] - df_34['ft'])

        tau_3 = np.average(psi_3, weights=df_3['n.patients']) / np.average(df_3['Z'], weights=df_3['n.patients'])
        tau_4 = np.average(psi_4, weights=df_4['n.patients']) / np.average(df_4['Z'], weights=df_4['n.patients'])
        tau_34 = np.average(psi_34, weights=df_34['n.patients']) / np.average(df_34['Z'], weights=df_34['n.patients'])

        # confidence intervals
        p3 = np.average(df_3['ps'], weights=df_3['n.patients'])
        p4 = np.average(df_4['ps'], weights=df_4['n.patients'])
        p34 = np.average(df_34['ps'], weights=df_34['n.patients'])

        sigma_3 = psi_3 - df_3['Z'] * tau_3 / p3
        sigma_4 = psi_4 - df_4['Z'] * tau_4 / p4
        sigma_34 = psi_34 - df_34['Z'] * tau_34 / p34

        sigma_3 = (np.average(sigma_3 ** 2, weights=df_3['n.patients'])) ** 0.5
        sigma_4 = (np.average(sigma_4 ** 2, weights=df_4['n.patients'])) ** 0.5
        sigma_34 = (np.average(sigma_34 ** 2, weights=df_34['n.patients'])) ** 0.5

        # ci_3 = (tau_3 - qt * sigma_3 / 500, tau_3 + qt * sigma_3 / 500)
        # ci_4 = (tau_4 - qt * sigma_4 / 500, tau_4 + qt * sigma_4 / 500)
        # ci_34 = (tau_34 - qt * sigma_34 / 500, tau_34 + qt * sigma_34 / 500)
        # print(ci_3,ci_4,ci_34)
        onerep[0] = [tau_34, sigma_34]
        onerep[1] = [tau_3, sigma_3]
        onerep[2] = [tau_4, sigma_4]

        # 12 subgroups
        row = 3
        for ind, var in enumerate(['X1', 'X2', 'X3', 'X4', 'X5']):
            if ind % 2 == 0:
                val_list = [0, 1]
            else:
                val_list = ['A', 'B', 'C']

            for val in val_list:
                df_var = df[(df['year'] >= 3) & (df[var] == val)].copy()

                # record df_len only at s=1
                if len(df_lens) < 15:
                    df_lens.append(len(df_var))

                # normalize m(x|var=val)
                # df_var['ps'] = df_var['ps'] / sum(df_var['ps']) * 2

                # mean
                score = df_var['Z'] * (df_var['Y'] - df_var['g0'] - df_var['ft']) - df_var['ps'] / (
                            1 - df_var['ps']) * (1 - df_var['Z']) * (df_var['Y'] - df_var['g0'] - df_var['ft'])
                tau = np.average(score, weights=df_var['n.patients']) / np.average(df_var['Z'],
                                                                                   weights=df_var['n.patients'])

                # variance
                p_t = np.average(df_var['ps'], weights=df_var['n.patients'])
                sigma = score - df_var['Z'] * tau / p_t
                sigma = (np.average(sigma ** 2, weights=df_var['n.patients'])) ** 0.5
                # ci = (tau - qt * sigma / len(df_var), tau + qt * sigma / len(df_var))

                # assign to onerep
                onerep[row] = [tau, sigma]
                row += 1

        tmp[:, :, layer] = onerep
        layer += 1

    # post processing, median version of the final estimator
    res = np.zeros((15, 2))
    res[:, 0] = np.median(tmp[:, 0, :], axis=1)
    res[:, 1] = np.median(tmp[:, 1, :] + (tmp[:, 0, :] - res[:, 0].reshape(15, 1)) ** 2, axis=1)

    # change every row in res to a dict, add to row_lists
    for ind, year in enumerate(['NA', 3, 4]):
        dict1 = {'dataset.num': str(i).zfill(4), 'variable': 'Overall', 'level': 'NA', 'year': year}
        dict1.update({'satt': res[ind, 0], 'lower90': res[ind, 0] - qt * res[ind, 1] / df_lens[ind],
                      'upper90': res[ind, 0] + qt * res[ind, 1] / df_lens[ind]})
        rows_list.append(dict1)

    for j, var in enumerate(['X1', 'X2', 'X3', 'X4', 'X5']):
        if j % 2 == 0:
            val_list = [0, 1]
        else:
            val_list = ['A', 'B', 'C']
        for val in val_list:
            ind += 1
            dict1 = {'dataset.num': str(i).zfill(4), 'variable': var, 'level': val, 'year': 'NA'}
            dict1.update({'satt': res[ind, 0], 'lower90': res[ind, 0] - qt * res[ind, 1] / df_lens[ind],
                          'upper90': res[ind, 0] + qt * res[ind, 1] / df_lens[ind]})
            rows_list.append(dict1)


start = int(sys.argv[1])
end = int(sys.argv[2])

rows_list = []
for i in range(start, end+1):
    estimate(i, 5, 50)
    if i%10 == 0:
        print(i)

df0 = pd.DataFrame(rows_list)
name = str(start).zfill(4) + '-' + str(end).zfill(4)
df0.to_csv(path_or_buf='./res/'+name+'.csv', index=False)