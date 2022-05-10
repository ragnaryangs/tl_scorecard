import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_lr_result(X_train_sm,X_test_sm,y_train):
    '''
    usage: 使用最终选择的特征，获取逻辑回归结果
    parameters：statsmodel特征筛选后的训练集，测试集，数据为woe列+常数项
    returns：逻辑回归结果lr
    '''    
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(penalty="l2", tol=0.0001, C=1.0, fit_intercept=True, solver="saga", max_iter=100, n_jobs=-1)
    X_train = X_train_sm.iloc[:, 1:]
    X_test = X_test_sm.iloc[:, 1:]
    LR.fit(X_train, y_train)
    #输出模型参数
    var = ["const"] + list(X_train.columns)
    coef = list(LR.intercept_) + list(LR.coef_[0])
    LR_result = pd.DataFrame({"var":var, "coef":coef})
    display(LR_result)
    return LR,LR_result

def get_model_vars_bins(df,bins):
    '''
    parameters：
    1.df：dataframe，最终筛选后的特征训练集的woe值，特征名称带有"_woe"后缀
    2.bins：dict, 最终的分箱结果
    returns:
    1.vars_final：list，模型最终使用的特征集
    2.bins_final：dict,最终入模特征的分箱结果
    '''    
    vars_final = list(df.columns[1:])
    vars_final = [var.replace("_woe","") for var in vars_final]
    bins_final = {}
    for var in vars_final:
        bins_final[var] = bins[var]
    return vars_final,bins_final

