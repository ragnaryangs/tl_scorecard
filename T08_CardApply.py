import scorecardpy as sc


def get_score_of_new_data(df,card,index = None,only_total_score = False, var_kp = None):
    '''
    usage:使用评分卡，对新数据集进行评分
    params:
    1.df: dataframe，新的数据集（如模型上线运行后的监控回溯数据）
    2.card: get_score_outcome输出的card文件，或通过7.2 / 7.3重写写入已保存的card文件
    3.only_total_score: logical，默认为True，即只输出样本总得分。如果为False，则同时输出特征得分和总得分
    4.print_step: 非负整数，是否打印变量名，默认值即可
    5.replace_blank_na: logical，默认值即可
    6.var_kp: list,跳过的评分列，如id列
    return: 新数据的评分
    '''
    df.set_index([index],inplace = True)
    score_of_new_data = sc.scorecard_ply(df, card,only_total_score = only_total_score,var_kp = var_kp)
    score_of_new_data.reset_index()
    return score_of_new_data