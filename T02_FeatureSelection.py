import numpy as np
import pandas as pd
import statsmodels.api as sm
import toad
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from toad.metrics import MSE, AIC, BIC, KS, AUC
from toad.utils import split_target, unpack_tuple, to_ndarray
import scipy.stats as scs

		
def feature_selection_1(df, eda, target="target", missing_rate=0.9, unique_num=1, cate_unique_num = 20, iv_lower_limit=0.05, iv_upper_limit=9999, exclude = None):
    '''
    usage:
    给定缺失率、单值率、离散特征属性个数阈值、iv值阈值，返回删除不符合要求的特征后的宽表，以及被删除的特征列表。
    parameters:
    1.df：包含目标y和特征x的宽表
    2.eda：toad.detect函数输出的dataframe，index为特征名，column为统计量
    3.target：df中目标变量的列名
    4.missing_rate：缺失率阈值，高于此阈值的特征删除
    5.unique_num：单一值个数阈值，低于此阈值的特征删除
    6.cate_unique_num: 字符型特征唯一值个数阈值，超出此个数的特征删除（应在本步骤之前进行处理）
    7.iv_lower_limit：iv值阈值下限，低于此阈值的特征删除
    8.iv_upper_limit：iv值阈值上限，高于此阈值的特征删除
    9.exclude：list，不进行筛选的特征名称的列表
    returns:
    1.df：删除不符合要求的特征后的宽表
    2.drop_list：因缺失率、单值率、iv值删除的特征列表
    3.drop_list1：因缺失率删除的特征列表
    4.drop_list2：因单值率删除的特征列表
    5.drop_list3：因字符属性值过多删除的特征列表
    6.drop_list4: 因iv值删除的特征列表
    '''
    new_df = df.copy()
    new_eda = eda.copy()
    if exclude is not None:
        new_df.drop(exclude, axis=1, inplace=True)
        new_eda.drop(exclude, axis=0, inplace=True)
    #步骤1：缺失率筛选
    drop_list1 = list(new_eda[new_eda["missing"] >= missing_rate].index)
    print("缺失率大于等于 %s 的特征数量有 %s 个" % (missing_rate, len(drop_list1)))
    #步骤2：单值率筛选
    drop_list2 = list(new_eda[new_eda["unique"] <= unique_num].index)
    print("单一值小于等于 %s 个的特征数量有 %s 个" % (unique_num, len(drop_list2)))
    #步骤3：字符属性值筛选
    drop_list3 = list(new_eda[[True if (new_eda['type'][i].kind in 'bmMOSUV') & (new_eda['unique'][i] > cate_unique_num) else False for i in new_eda.index]].index)
    print("字符型特征类别个数大于 %s 个的特征数量有 %s 个" % (cate_unique_num,len(drop_list3)))
    #步骤4：iv值初筛（使用toad的quality函数做iv值初筛）
    #参数：变量宽表df，目标变量名称target，是否只输出iv值True
    #输出：由每个特征的iv、gini、entropy、unique构成的dataframe
    iv_list = toad.quality(dataframe=new_df, target=target, iv_only=True)
    drop_list4 = list(iv_list.loc[(iv_list["iv"]<iv_lower_limit) | (iv_list["iv"]>iv_upper_limit)].index)
    print("IV值小于 %s 或大于 %s 的特征数量有 %s 个" % (iv_lower_limit,iv_upper_limit,len(drop_list4)))
    #缺失率、单值率、类别属性值、iv值去重后剔除特征
    drop_list = list(set(drop_list1 + drop_list2 + drop_list3 + drop_list4))
    print("因缺失率、单值率、类别属性值、iv值剔除的特征数量有 %s 个" % (len(drop_list)))
    #剔除特征
    df.drop(drop_list, axis=1, inplace=True)
    print("Shape：",df.shape)
    return df, drop_list, drop_list1, drop_list2, drop_list3, drop_list4
    
 

def feature_selection_corr(df_train_woe, df_test_woe, target, threshold=0.7, by="IV",
                           return_drop=True, exclude=None):
    '''
    usage: 使用相关系数对特征进行筛选
    parameters:
    1.df_train_woe：dataframe，原特征训练集的woe值
    2.df_test_woe：dataframe，原特征测试集的woe值
    3.target：str，目标变量的名称
    4.threshold：float，相关系数筛选的阈值
    5.by：str，默认剔除IV值较低的特征
    6.return_drop：bool，是否返回剔除特征的列表
    7.exclude：list，不进行筛选的特征名称的列表
    returns:
    1.df_train_woe_new：dataframe，经筛选得到的新训练集的woe值
    2.df_test_woe_new：dataframe，经筛选得到的新测试集的woe值
    3.corr_dropped：list，剔除特征的列表
    '''
    df_train_woe_new, corr_dropped = toad.selection.drop_corr(df_train_woe, target=target, threshold=threshold,
                                                              by=by, return_drop=return_drop, exclude=exclude)
    df_test_woe_new = df_test_woe[list(df_train_woe_new.columns)]
    print("经过相关系数检验剔除的特征数：%s" % len(corr_dropped))
    return df_train_woe_new, df_test_woe_new, corr_dropped
    
    
    
def feature_selection_vif(df_train_woe, df_test_woe, threshold=10, return_drop=True, exclude=None):
    '''
    usage: 使用方差膨胀系数VIF对特征进行筛选
    parameters:
    1.df_train_woe：dataframe，原特征训练集的woe值
    2.df_test_woe：dataframe，原特征测试集的woe值
    3.threshold：float，VIF筛选的阈值
    4.return_drop：bool，是否返回剔除特征的列表
    5.exclude：list，不进行筛选的特征名称的列表
    returns:
    1.df_train_woe_new：dataframe，经筛选得到的新训练集的woe值
    2.df_test_woe_new：dataframe，经筛选得到的新测试集的woe值
    3.vif_dropped：list，剔除特征的列表
    '''
    df_train_woe_new, vif_dropped = toad.selection.drop_vif(df_train_woe, threshold=threshold, return_drop=return_drop, exclude=exclude)
    df_test_woe_new = df_test_woe[list(df_train_woe_new.columns)]
    print("经过VIF检验剔除的特征数：%s" % len(vif_dropped))
    return df_train_woe_new, df_test_woe_new, vif_dropped



def feature_selection_psi(df_train_woe, df_test_woe, threshold=0.1, exclude=None):
    '''
    usage: 使用群体稳定性系数PSI对特征进行筛选
    parameters:
    1.df_train_woe：dataframe，原特征训练集的woe值
    2.df_test_woe：dataframe，原特征测试集的woe值
    3.threshold：float，PSI筛选的阈值
    4.exclude：list，不进行筛选的特征名称的列表
    returns:
    1.df_train_woe_new：dataframe，经筛选得到的新训练集的woe值
    2.df_test_woe_new：dataframe，经筛选得到的新测试集的woe值
    3.psi_dropped：list，剔除特征的列表
    '''
    var_psi = pd.DataFrame(toad.metrics.PSI(df_test_woe, df_train_woe, return_frame=False).reset_index())
    var_psi.columns = ["x", "psi"]
    if exclude is not None:
        psi_dropped = var_psi.loc[(var_psi["psi"]>threshold) & (~var_psi["x"].isin(exclude))]["x"].to_list()
    else:
        psi_dropped = var_psi.loc[var_psi["psi"]>threshold]["x"].to_list()
    df_train_woe_new = df_train_woe.drop(psi_dropped, axis=1, inplace=False)
    df_test_woe_new = df_test_woe.drop(psi_dropped, axis=1, inplace=False)
    print("经过PSI检验剔除的特征数：%s" % len(psi_dropped))
    return df_train_woe_new, df_test_woe_new, psi_dropped



def bins_chi_test(bins, var_name, alpha=0.05):
    '''
    usage:
    给定显著性水平，对变量的分箱结果（除空箱外）进行卡方检验。
    parameters:
    1.bins：dict，scorecardpy.woebin函数输出的分箱结果
    2.var_name：string，进行卡方检验的变量名称
    3.alpha：float，进行卡方检验的显著性水平
    returns:
    chi_result: int，0说明通过卡方检验；1说明未通过卡方检验；2说明分箱数量太少不足以进行卡方检验
    '''
    from scipy.stats import chi2
    #根据两个分箱中好坏样本数量进行卡方检验，自由度为(2-1)*(2-1)=1
    critical_value = chi2.ppf(1-alpha, 1)  #显著性水平为alpha、自由度为1的卡方检验临界值
    var_bin_chi = bins[var_name].query(' bin != "missing" ')[["bad", "good", "count", "badprob"]]
    if len(var_bin_chi) <= 1:
        print("变量 %s 的分箱数量太少，无法进行卡方检验！" % var_name)
        return 2
    else:
        break2 = False
        chi_result = 0
        #利用双层循环对分箱进行两两卡方检验
        for i in range(len(var_bin_chi)-1):
            for j in range(i+1, len(var_bin_chi)):
                df_chi = var_bin_chi.iloc[[i,j], :]
                exp_bad_rate = df_chi["bad"].sum() / df_chi["count"].sum()  #理论坏样本比例
                df_chi["exp_bad"] = df_chi["count"] * exp_bad_rate          #理论坏样本数量
                df_chi["exp_good"] = df_chi["count"] * (1 - exp_bad_rate)   #理论好样本数量
                df_chi["chi_bad"] = (df_chi["bad"] - df_chi["exp_bad"]) ** 2 / df_chi["exp_bad"]
                df_chi["chi_good"] = (df_chi["good"] - df_chi["exp_good"]) ** 2 / df_chi["exp_good"]
                chi_value = df_chi["chi_bad"].sum() + df_chi["chi_good"].sum()  #卡方值计算Σ(Act-Exp)^2/Exp
                if chi_value <= critical_value:
                    print("变量 %s 的分箱结果未通过卡方检验" % var_name)
                    chi_result += 1
                    break2 = True
                    break  #跳出内循环
                else:
                    continue
            if break2:
                break  #跳出外循环       
        if not break2:
            print("变量 %s 的分箱结果通过了卡方检验" % var_name)
        return chi_result



def feature_selection_chi(df_train_woe, df_test_woe, bins, target, alpha=0.05, exclude = []):
    '''
    usage: 对分箱结果进行卡方检验进行特征筛选
    parameters:
    1.df_train_woe：dataframe，原特征训练集的woe值，特征名称带有"_woe"后缀
    2.df_test_woe：dataframe，原特征测试集的woe值
    3.bins：dict，scorecarpy.woebin的分箱结果
    4.target：str，目标变量的名称
    5.alpha：float进行卡方检验的显著性水平
    6.exclude: list,不参与卡方检验的字段列表
    returns:
    1.df_train_woe_new：dataframe，经筛选得到的新训练集的woe值
    2.df_test_woe_new：dataframe，经筛选得到的新测试集的woe值
    3.chi_dropped：list，剔除特征的列表
    '''
    chi_dropped = []
    var_list = list(df_train_woe.columns)
    for var in var_list:
        var = var.replace("_woe", "")
        if var != target and var not in exclude:
            chi_result = bins_chi_test(bins, var, alpha=alpha)
            if chi_result != 0:
                var += "_woe"
                chi_dropped.append(var)
            else:
                continue
    df_train_woe_new = df_train_woe.drop(chi_dropped, axis=1, inplace=False)
    df_test_woe_new = df_test_woe.drop(chi_dropped, axis=1, inplace=False)
    print("经过卡方检验剔除的特征数：%s" % len(chi_dropped))
    return df_train_woe_new, df_test_woe_new, chi_dropped
 

 
def feature_selection_coef_p(df_train_woe,df_test_woe,target = "default",by_aic = True,p_limit = 0.05):
    '''
    usage: 对于所有系数为负或不显著的特征，逐一尝试剔除
    parameters：
    1.df_train_woe：dataframe，原特征训练集的woe值+目标变量，特征名称带有"_woe"后缀
    2.df_test_woe：dataframe，原特征测试集的woe值+目标变量
    3.target：str，目标变量的名称
    4.by_aic：bool，是否选择使得模型AIC最小的特征进行剔除，否的话不考虑AIC是否最优
    5.p_limit：显著性水平阈值
    returns：
    筛选后的训练集X，测试集X,训练集y，测试集y，statsmodels的逻辑回归结果lr
    '''
    X_train = df_train_woe.loc[:, df_train_woe.columns!=target]
    X_test = df_test_woe.loc[:, df_test_woe.columns!=target]
    y_train = df_train_woe.loc[:, target]
    y_test = df_test_woe.loc[:, target]
    #剔除系数为负或不显著的特征
    if by_aic:
        X_train_sm, X_test_sm, lr = logit_drop_by_aic(X_train, X_test, y_train, p_limit = p_limit)
    else:
        X_train_sm, X_test_sm, lr = logit_drop_all(X_train, X_test, y_train, p_limit = p_limit)
    display(lr.summary())
    #提取剔除的变量
    #目标变量不一定位于第一列
    #vars_start = list(df_train_woe.columns[1:])
    vars_start = [var for var in df_train_woe.columns if var != target]
    vars_end = list(X_train_sm.columns[1:])
    coef_p_dropped = list(set(vars_start).difference(set(vars_end)))  
    print("经过系数及P值剔除的特征数：%s" % len(coef_p_dropped))    
    return X_train_sm, X_test_sm, y_train, y_test, coef_p_dropped


def logit_drop_by_aic(X_train, X_test, y_train, p_limit=0.05):
    '''
    usage: 对于所有系数为负或不显著的特征，逐一尝试剔除，选择使得模型AIC最小的特征进行剔除
    parameters：训练集X，测试集X，训练集y，显著性水平阈值p_limit
    returns：筛选后的训练集X，测试集X，statsmodels的逻辑回归结果lr
    '''
    X_train_sm = sm.add_constant(X_train)  #新增第一列const
    X_test_sm = sm.add_constant(X_test)
    cols = list(X_train_sm.columns)  #所有列名
    while True:
        log_reg = sm.Logit(y_train, X_train_sm)
        #lr = log_reg.fit_regularized(method="l1", maxiter=100, alpha=1.0, trim_mode="off")
        lr = log_reg.fit(method="ncg", max_iter=100)
        params, p_values = lr.params[1:], lr.pvalues[1:]  #Series，不包含const
        drop_list1 = list(params[params<0].index)  #系数为负的特征
        drop_list2 = list(p_values[p_values>=p_limit].index)  #系数不显著的特征
        drop_list = list(set(drop_list1 + drop_list2))
        if len(drop_list) == 0:
            print(list(X_train_sm.columns))
            break
        else:
            var_aic = []
            for var in drop_list:
                #将需要剔除的特征逐一剔除后进行回归查看AIC，选择回归后最小的AIC对应的特征进行剔除
                X_train_sm_sub = X_train_sm.drop(var, axis=1, inplace=False)
                #aic = sm.Logit(y_train, X_train_sm_sub).fit_regularized(method="l2", maxiter=100, alpha=1.0, trim_mode="off").aic
                aic = sm.Logit(y_train, X_train_sm_sub).fit(method="ncg", max_iter=100).aic
                var_aic.append((aic, var))
            var_aic.sort(reverse=True)  #对AIC降序排序，选最小
            best_aic, best_var = var_aic[-1]
            X_train_sm.drop(best_var, axis=1, inplace=True)
            X_test_sm.drop(best_var, axis=1, inplace=True)
            print("Variable %s dropped!" % best_var)
    return X_train_sm, X_test_sm, lr



def logit_drop_all(X_train, X_test, y_train, p_limit=0.05):
    '''
    usage: 剔除逻辑回归中所有系数为负或不显著的特征
    parameters：训练集X，测试集X，训练集y，显著性水平阈值p_limit
    returns：筛选后的训练集X，测试集X，statsmodels的逻辑回归结果lr
    '''
    X_train_sm = sm.add_constant(X_train)  #新增第一列const
    X_test_sm = sm.add_constant(X_test)
    cols = list(X_train_sm.columns)  #所有列名
    while True:
        log_reg = sm.Logit(y_train, X_train_sm)
        #lr = log_reg.fit_regularized(method="l1", maxiter=100, alpha=1.0, trim_mode="off")
        lr = log_reg.fit(method="ncg", max_iter=100)
        params, p_values = lr.params[1:], lr.pvalues[1:]  #Series
        drop_list1 = list(params[params<0].index)
        drop_list2 = list(p_values[p_values>=p_limit].index)
        drop_list = list(set(drop_list1 + drop_list2))
        if len(drop_list) == 0:
            print(list(X_train_sm.columns))
            break
        else:
            X_train_sm.drop(drop_list, axis=1, inplace=True)
            X_test_sm.drop(drop_list, axis=1, inplace=True)
    return X_train_sm, X_test_sm, lr



class StatsModel:
    def __init__(self, estimator = 'ols', criterion = 'aic', intercept = False):
        if isinstance(estimator, str):
            Est = self.get_estimator(estimator)
            estimator = Est(fit_intercept = intercept,)

        self.estimator = estimator
        self.intercept = intercept
        self.criterion = criterion


    def get_estimator(self, name):
        ests = {
            'ols': LinearRegression,
            'lr': LogisticRegression,
            'lasso': Lasso,
            'ridge': Ridge,
        }

        if name in ests:
            return ests[name]

        raise Exception('estimator {name} is not supported'.format(name = name))



    def stats(self, X, y):
        X = X.copy()

        if isinstance(X, pd.Series):
            X = X.to_frame()

        self.estimator.fit(X, y)

        if hasattr(self.estimator, 'predict_proba'):
            pre = self.estimator.predict_proba(X)[:, 1]
        else:
            pre = self.estimator.predict(X)

        coef = self.estimator.coef_.reshape(-1)

        if self.intercept:
            coef = np.append(coef, self.estimator.intercept_)
            X['intercept'] = np.ones(X.shape[0])

        n, k = X.shape

        t_value = self.t_value(pre, y, X, coef)
        p_value = self.p_value(t_value, n)
        c = self.get_criterion(pre, y, k)

        return {
            't_value': pd.Series(t_value, index = X.columns),
            'p_value': pd.Series(p_value, index = X.columns),
            'criterion': c
        }

    def get_criterion(self, pre, y, k):
        if self.criterion == 'aic':
            llf = self.loglikelihood(pre, y, k)
            return AIC(pre, y, k, llf = llf)

        if self.criterion == 'bic':
            llf = self.loglikelihood(pre, y, k)
            return BIC(pre, y, k, llf = llf)

        if self.criterion == 'ks':
            return KS(pre, y)

        if self.criterion == 'auc':
            return AUC(pre, y)

    def t_value(self, pre, y, X, coef):
        n, k = X.shape
        mse = sum((y - pre) ** 2) / float(n - k)
        nx = np.dot(X.T, X)

        if np.linalg.det(nx) == 0:
            return np.nan
        else:
            std_e = np.sqrt(mse * (np.linalg.inv(nx).diagonal()))
            return coef / std_e

    def p_value(self, t, n):
        return scs.t.sf(np.abs(t), n - 1) * 2

    def loglikelihood(self, pre, y, k):
        n = len(y)
        mse = MSE(pre, y)
        return (-n / 2) * np.log(2 * np.pi * mse * np.e)


def stepwise(frame, target = 'target', estimator = 'ols', direction = 'both', criterion = 'aic',
            p_enter = 0.01, p_remove = 0.01, p_value_enter = 0.2, intercept = False,
            max_iter = None, return_drop = False, exclude = None):
    """stepwise to select features

    Args:
        frame (DataFrame): dataframe that will be use to select
        target (str): target name in frame
        estimator (str): model to use for stats
        direction (str): direction of stepwise, support 'forward', 'backward' and 'both', suggest 'both'
        criterion (str): criterion to statistic model, support 'aic', 'bic', 'ks', 'auc'
        p_enter (float): threshold that will be used in 'forward' and 'both' to keep features
        p_remove (float): threshold that will be used in 'backward' to remove features
        intercept (bool): if have intercept
        p_value_enter (float): threshold that will be used in 'both' to remove features
        max_iter (int): maximum number of iterate
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df, y = split_target(frame, target)

    if exclude is not None:
        df = df.drop(columns = exclude)

    drop_list = []  #经过逐步回归删除的变量列表
    remaining = df.columns.tolist()  #经过逐步回归剩余的变量列表

    selected = []

    sm = StatsModel(estimator = estimator, criterion = criterion, intercept = intercept)

    order = -1 if criterion in ['aic', 'bic'] else 1  #aic/bic指标越小越好，ks/auc越大越好

    best_score = -np.inf * order

    iter = -1
    while remaining:
        iter += 1
        if max_iter and iter > max_iter:
            break

        l = len(remaining)
        test_score = np.zeros(l)
        test_res = np.empty(l, dtype = np.object)

        if direction == 'backward':
            for i in range(l):
            	#每次剔除一个变量，与y进行ols回归
            	#sm.stats返回dict格式对象，包括回归得到的t值、p值、criterion值(aic/bic/ks/auc)
                test_res[i] = sm.stats(
                    df[ remaining[:i] + remaining[i+1:] ],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            if (curr_score - best_score) * order < p_remove:
            	#若剔除该变量，模型criterion无法提升，则停止逐步回归
                break

            name = remaining.pop(curr_ix)
            print('变量 %s 被删除' % name)
            drop_list.append(name)

            best_score = curr_score

        # forward and both
        else:
            for i in range(l):
            	#每次加入一个变量，与y进行ols回归
                test_res[i] = sm.stats(
                    df[ selected + [remaining[i]] ],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)
            if (curr_score - best_score) * order < p_enter:
            	#若加入该变量，模型criterion无法提升，则删除该变量
                print('变量 %s 被删除' % name)
                drop_list.append(name)

                # early stop
                if selected:
                	#若此时selected列表不为空，则停止逐步回归
                    drop_list += remaining
                    break

                continue

            print('变量 %s 被加入' % name)
            selected.append(name)
            best_score = curr_score

            if direction == 'both':
                p_values = test_res[curr_ix]['p_value']
                #ols回归后p值大于阈值(默认0.2)的变量会被删除
                drop_names = p_values[p_values > p_value_enter].index

                for name in drop_names:
                    selected.remove(name)
                    print('变量 %s 因超过p值阈值 %s 被删除' % (name, p_value_enter))
                    drop_list.append(name)

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def feature_selection_stepwise(df_train_woe, df_test_woe, target, estimator='ols', direction='both', criterion='aic',
                               p_enter=0.01, p_remove=0.01, p_value_enter=0.2, intercept=False, max_iter=None,
                               return_drop=True, exclude=None):
    '''
    usage: 使用逐步回归方法对特征进行筛选
    parameters:
    1.df_train_woe：dataframe，原特征训练集的woe值
    2.df_test_woe：dataframe，原特征测试集的woe值
    3.target：str，目标变量的名称
    4.estimator：str，逐步回归使用的模型，默认ols
    5.direction：str，支持forward，backward，both
    6.criterion：str，支持aic，bic，ks，auc
    7.p_enter：float，在forward和both中用于保留特征的阈值
    8.p_remove：float，在backward中用于剔除特征的阈值
    9.p_value_enter：float，在both中用于剔除特征的阈值
    10.intercept：bool，回归时是否带有截距项
    11.max_iter：int，最大迭代次数
    12.return_drop：bool，是否返回剔除特征的列表
    13.exclude：list，不进行筛选的特征名称的列表
    returns:
    1.df_train_woe_new：dataframe，经筛选得到的新训练集的woe值
    2.df_test_woe_new：dataframe，经筛选得到的新测试集的woe值
    3.stepwise_dropped：list，剔除特征的列表
    '''
    df_train_woe_new, stepwise_dropped = stepwise(df_train_woe, target=target, estimator=estimator,
                                                  direction=direction, criterion=criterion, p_enter=p_enter,
                                                  p_remove=p_remove, p_value_enter=p_value_enter, intercept=intercept,
                                                  max_iter=max_iter, return_drop=return_drop, exclude=exclude)
    df_test_woe_new = df_test_woe[list(df_train_woe_new.columns)]
    print("经过逐步回归剔除的特征数：%s" % len(stepwise_dropped))
    return df_train_woe_new, df_test_woe_new, stepwise_dropped
    


