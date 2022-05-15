import numpy as np
import pandas as pd
import scipy.stats
import sys

from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold


# i: index of dataset
# k: number of folds
# S: number of repeat of DML
def estimate(i, k, S):
    # load and reorder the data
    path1 = "./data/practice/acic_practice_"
    path2 = "./data/practice_year/acic_practice_year_"
    fnm1 = path1 + str(i).zfill(4) + ".csv"
    fnm2 = path2 + str(i).zfill(4) + ".csv"
    #data_x = pd.read_csv("./data/practice/acic_practice_0001.csv")
    #data_y = pd.read_csv("./data/practice_year/acic_practice_year_0001.csv")
    data_x = pd.read_csv(fnm1)
    data_y = pd.read_csv(fnm2)
    
    data_y = pd.concat([data_y[data_y['year'] == j] for j in range(1,5)])
    #data_y = data_y.reset_index(drop=True)
    
    # preprocessing for tree models
    ct = ['X2','X4']
    for c in ct:
        data_x[c] = data_x[c].astype('category')
    
    X = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','pre_level','trend']
    V = ['V1_avg', 'V2_avg', 'V3_avg', 'V4_avg', 'V5_A_avg', 'V5_B_avg', 'V5_C_avg']
    XV = V + X
    Y = ['Y']
    T = ['Z']
    
    #preprocessing, y_new = (Y_3+Y_4)/2-(Y_1+Y_2)/2, Vk_new = avg(Vk)
    names = globals()
    for year in range(1,5):
        names['data_' + str(year)] = data_y.loc[data_y['year'] == year].reset_index(drop=True)
    data = data_1[data_y.columns[[0,2,3] + list(range(6,13))]]
    
    for col in data_y.columns[6:]:
        #data.col
        #data.loc[:, col] = (data_4[col] + data_3[col] + data_2[col] + data_1[col]) / 4 # warnings
        data[:][col] = (data_4[col] + data_3[col] + data_2[col] + data_1[col]) / 4
        
    # add (Y_1+Y_2)/2, Y_2-Y_1 and (Y_3+Y_4)/2-(Y_1+Y_2)/2
    data = data.assign(pre_level=(data_1[Y] + data_2[Y]) / 2)
    data = data.assign(trend=data_2[Y] - data_1[Y])
    data = data.assign(outcome=(data_3[Y] + data_4[Y] - data_2[Y] - data_1[Y]) / 2)
    
    # dataframe to use
    df = pd.merge(data_x, data, on='id.practice')
        
    # defince ML model objects, to estimate e(X) and g_0(V,X)
    prob_m = LGBMClassifier(objective = 'binary',
                            is_unbalance = True,
                            #metric = 'log_loss',
                            metric = 'binary_logloss,auc',
                            max_depth = 8,
                            num_leaves = 20,
                            learning_rate = 0.1,
                            #feature_fraction = 0.7,
                            min_child_samples=20,
                            min_child_weight=0.001,
                            #bagging = 1,
                            #subsample_freq = 2,
                            reg_alpha = 0.002,
                            reg_lambda = 10,
                            cat_smooth = 0,
                            n_estimators = 200,   
                            )

    mean_m = LGBMRegressor(is_unbalance = True,
                           #metric = 'log_loss',
                           metric = 'l2',
                           max_depth = 8,
                           num_leaves = 20,
                           learning_rate = 0.1,
                           #feature_fraction = 0.7,
                           min_child_samples=20,
                           min_child_weight=0.001,
                           #bagging = 1,
                           #subsample_freq = 2,
                           reg_alpha = 0.002,
                           reg_lambda = 10,
                           cat_smooth = 0,
                           n_estimators = 200,   
                           )
    
    # a list of ndarrays to store results
    tmp = np.zeros((13,2,S))
    layer = 0
    
    # record lengths of sub-dataframes
    df_lens = [500]
    
    for _ in range(S):
        # create k-fold indices
        kf = KFold(n_splits=k, shuffle=True)
        indices = []
        for train_index, test_index in kf.split(data_x):
            indices.append((train_index, test_index))
            
        # propensity score
        df = df.assign(ps=0)
        for train_index, test_index in indices:
            prob_m.fit(X=df[X].iloc[train_index], y=df['Z'].iloc[train_index].values.ravel())
            df.loc[test_index, 'ps'] = prob_m.predict_proba(X=df[X].iloc[test_index])[:,1]
          
        # g_0(X,V)
        df = df.assign(g0=0)
        for train_index, test_index in indices:    
            tr = df['Z'].iloc[train_index]
            xx = df[XV].iloc[train_index]
            yy = df['outcome'].iloc[train_index]
            mean_m.fit(X=xx, y=yy)    
            df.loc[test_index, 'g0'] = mean_m.predict(df[XV].iloc[test_index])       
        #print(df['g0'])
        
        # ATTEs
        onerep = np.zeros((13,2))
        qt = scipy.stats.norm.ppf(0.95)
        row = 0
        
        # all, year 3, year 4
        #df = df[2 * df['Z'] * df['ps'] - df['Z'] -df['ps'] + 1 > 0.1]

        psi = df['Z'] * (df['outcome'] - df['g0']) - df['ps'] / (1 - df['ps']) * (1 - df['Z']) * (df['outcome'] - df['g0'])
        

        tau = np.average(psi) / np.average(df['Z'])
        
        # confidence intervals
        p = np.average(df['ps'])
        sigma = psi - df['Z'] * tau / p
        sigma = (np.average(sigma ** 2)) ** 0.5

        #ci = (tau - qt * sigma / 500, tau + qt * sigma / 500)
        #print(ci)
        onerep[0] = [tau, sigma]
        
        # 12 subgroups
        row = 1
        for ind, var in enumerate(['X1','X2','X3','X4','X5']):
            if ind %2 == 0:
                val_list = [0, 1]
            else:
                val_list = ['A', 'B', 'C']
            
            for val in val_list:
                df_var = df[df[var] == val].copy()
                
                # record df_len only at s=1
                if len(df_lens) < 13:
                    df_lens.append(len(df_var))
                
                # normalize m(x|var=val)
                #df_var['ps'] = df_var['ps'] / sum(df_var['ps']) * 2
                
                # mean
                score = df_var['Z'] * (df_var['outcome'] - df_var['g0']) - df_var['ps'] / (1 - df_var['ps']) * (1 - df_var['Z']) * (df_var['outcome'] - df_var['g0'])
                tau = np.average(score) / np.average(df_var['Z'])
                
                # variance
                p_t = np.average(df_var['ps'])
                sigma = score - df_var['Z'] * tau / p_t
                sigma = (np.average(sigma ** 2)) ** 0.5
                #ci = (tau - qt * sigma / len(df_var), tau + qt * sigma / len(df_var))
                
                # assign to onerep
                onerep[row] = [tau, sigma]
                row += 1
                
        tmp[:,:,layer] = onerep
        layer += 1

    # post processing, median version of the final estimator
    res = np.zeros((13,2))
    res[:,0] = np.median(tmp[:,0,:], axis=1)
    res[:,1] = np.median(tmp[:,1,:] + (tmp[:,0,:]-res[:,0].reshape(13,1)) ** 2, axis=1)
        
    # change every row in res to a dict, add to row_lists
    dict1 = {'dataset.num':str(i).zfill(4), 'variable':'Overall', 'level':'NA', 'year':'NA'}
    dict1.update({'satt':res[0,0], 'lower90':res[0,0] - qt * res[0,1]/df_lens[0], 'upper90':res[0,0] + qt * res[0,1]/df_lens[0]})
    rows_list.append(dict1)
    
    #for ind, year in enumerate(['NA',3,4]):
        #dict1 = {'dataset.num':str(i).zfill(4), 'variable':'Overall', 'level':'NA', 'year':year}
        #dict1.update({'satt':res[ind,0], 'lower90':res[ind,0] - qt * res[ind,1]/df_lens[ind], 'upper90':res[ind,0] + qt * res[ind,1]/df_lens[ind]})
        #rows_list.append(dict1)
    
    ind = 0
    for j, var in enumerate(['X1','X2','X3','X4','X5']):
        if j%2 == 0:
            val_list = [0, 1]
        else:
            val_list = ['A', 'B', 'C']
        for val in val_list:
            ind += 1
            dict1 = {'dataset.num':str(i).zfill(4), 'variable':var, 'level':val, 'year':'NA'}
            dict1.update({'satt':res[ind,0], 'lower90':res[ind,0] - qt * res[ind,1]/df_lens[ind], 'upper90':res[ind,0] + qt * res[ind,1]/df_lens[ind]})
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
df0.to_csv(path_or_buf='./post/'+name+'.csv', index=False)