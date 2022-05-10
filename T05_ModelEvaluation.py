import pandas as pd
import toad
import scorecardpy as sc


def get_ks_roc(LR,X_train_sm,X_test_sm,y_train,y_test,df_train,df_test,target = "default"):
    '''
    usage: 绘制ks、roc图，并输出KS分箱表，和加了预测概率列的原始训练集、测试集
    parameters：
    1.LR：sklearn训练得到的模型
    2.X_train_sm：dataframe，全部特征筛选后的训练集，数据为woe列+常数项，无target列
    3.X_test_sm：dataframe，全部特征筛选后的测试集，数据为woe列+常数项，无target列   
    4.y_train：dataframe，全部特征筛选后的训练集标签，仅有target列
    5.y_test：dataframe，全部特征筛选后的测试集标签，仅有target列
    6.df_train：dataframe，原始训练数据集（训练集测试集刚划分）
    7.df_test：dataframe，原始测试数据集（训练集测试集刚划分）    
    8.target：str，目标变量的名称
    returns：
    加了预测概率列的原始训练集、测试集；以及训练集测试集的KS分箱结果
    '''
    X_train = X_train_sm.iloc[:, 1:]
    X_test = X_test_sm.iloc[:, 1:]
    train_pred = LR.predict_proba(X_train)[:, 1]
    test_pred = LR.predict_proba(X_test)[:, 1]
    #将预测结果添加到原数据集中
    df_train["train_pred"] = train_pred
    df_test["test_pred"] = test_pred
    train_perf = sc.perf_eva(label=y_train, pred=train_pred, plot_type=["ks", "roc"], title="train", show_plot=True)
    test_perf = sc.perf_eva(label=y_test, pred=test_pred, plot_type=["ks", "roc"], title="test", show_plot=True)
    print("train KS: ", train_perf["KS"])
    print("test KS: ", test_perf["KS"])
    print("train AUC: ",  train_perf["AUC"])
    print("test AUC: ", test_perf["AUC"])    
    #分箱展示KS结果
    KS_bucket_train = toad.metrics.KS_bucket(score=df_train["train_pred"], target=df_train[target],
                                             bucket=10, method="quantile", return_splits=False)
    KS_bucket_test = toad.metrics.KS_bucket(score=df_test["test_pred"], target=df_test[target],
                                            bucket=10, method="quantile", return_splits=False)
    display(KS_bucket_train)
    display(KS_bucket_test)
    return df_train,df_test,KS_bucket_train,KS_bucket_test



def get_psi_eval(df_train, df_test, df_train_woe, df_test_woe, card, target="default", var_skip=[], return_var_psi=True, 
                 x_limits=None, x_tick_break=50, show_plot=True, seed=186, return_distr_dat=True):
    '''
    usage：计算变量和模型分数的psi
    parameters：
    1.df_train,df_test：dataframe，包含原始变量的dataframe
    2.df_train_woe,df_test_woe：dataframe，包含变量woe值的dataframe
    3.card：dict，scorecardpy生成的评分卡对象
    4.target：string，目标变量名称
    5.var_skip：list，不需要进行psi计算的变量列表
    6.return_var_psi：boolean，是否同时计算变量的psi
    7.x_limits：list，x轴的最小、最大值
    8.x_tick_break：int，x轴刻度的间隔，适用于评分psi的计算，变量直接根据woe分箱切分
    9.show_plot：boolean，是否画psi图
    10.seed：int，随机数种子
    11.return_distr_dat：boolean，是否返回按分数分箱的dataframe
    returns：
    1.psi_all_value：dict，包含分数或变量psi值的字典
    2.psi_all_df：dataframe，包含分数或变量分箱计算psi的dataframe
    '''
    psi_all_value = {}
    psi_all_df = pd.DataFrame()
    train_score = sc.scorecard_ply(dt=df_train, card=card, only_total_score=True,
                                   print_step=0, replace_blank_na=True, var_kp=None)
    test_score = sc.scorecard_ply(dt=df_test, card=card, only_total_score=True,
                                   print_step=0, replace_blank_na=True, var_kp=None)
    y_train = df_train[target]
    y_test = df_test[target]
    df_train_woe_withscore = pd.concat([df_train_woe, train_score], axis=1)
    df_test_woe_withscore = pd.concat([df_test_woe, test_score], axis=1)
    if return_var_psi:
        for col in df_train_woe_withscore.columns:
            if col in var_skip:
                print("变量 %s 不需要进行PSI评估" % col)
            else:
                psi_result = sc.perf_psi(
                            score={"train":df_train_woe_withscore[col].to_frame(),"test":df_test_woe_withscore[col].to_frame()},
                            label={"train":y_train,"test":y_test},
                            title=None, x_limits=x_limits, x_tick_break=x_tick_break, show_plot=show_plot,
                            seed=seed, return_distr_dat=return_distr_dat)
                psi_value = psi_result["psi"].iloc[0,1]
                psi_all_value[col] = psi_value
                if return_distr_dat:
                    psi_df = psi_result["dat"][col]
                    psi_df.index = [col] * len(psi_df)
                    psi_all_df = pd.concat([psi_all_df, psi_df])
        return psi_all_value, psi_all_df
    else:
        score_psi = sc.perf_psi(
                            score={"train":train_score,"test":test_score},
                            label={"train":y_test,"test":y_test},
                            title=None, x_limits=x_limits, x_tick_break=x_tick_break, show_plot=show_plot,
                            seed=seed, return_distr_dat=return_distr_dat)
        psi_all_value["score"] = score_psi["psi"].iloc[0,1]
        psi_all_df = pd.concat([psi_all_df, score_psi["dat"]["score"]])
        return psi_all_value, psi_all_df
        












