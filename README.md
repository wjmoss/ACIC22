# ACIC22


This method uses double machine learning framework to estimate average treatement for treated, with the correction of average year effects. The outcome function g_0 for untreated, the year average for untreated t_i's and the propensity scores are estimated via 5-fold cross-fitting, then the point estimate and variance estimate are obtained by standard results of DML. Finally, the procedure is repeated 50 times for each practice, and the median version of ensembling is used, to relieve the uncertainty caused by sample splitting. For more robust results (robust=True), large discrepancies between actual treatment value and propensity score are omitted, where the threshold value is chosen to be 0.1. (Z=1 and ps<0.1, Z=0 and ps>0.9)