import pandas as pd
import scorecardpy as sc


def get_score_outcome(bins,model,xcolumns,base_point,odds,pdo,df_train,df_test):
    '''
    parameters：
    1.bins：分箱结果
    2.model：评分卡模型结果
    3.xcolumns:list,最终入模变量清单
    4.base_point:int,基准分
    5.odds:decimal,好坏比，好/坏
    6.pdo:int,好坏比翻倍时得分增加值
    7.df_train:原训练集
    8.df_test:原测试集
    returns:
    1.card:scorecardpy生成的卡文件    
    2.card_df：dataframe格式的评分卡
    3.df_train_withscore:dataframe,带得分列的训练集
    4.df_test_withscore:dataframe,带得分列的测试集
    '''        
    card = sc.scorecard(bins=bins, model=model, xcolumns=xcolumns,
                    points0=base_point, odds0=1/odds, pdo=pdo, basepoints_eq0=True)
    min_score, max_score, card_df = get_score_range(card, bins)
    train_score = sc.scorecard_ply(dt=df_train, card=card, only_total_score=True,
                               print_step=0, replace_blank_na=True, var_kp=None)
    df_train_withscore = pd.concat([df_train, train_score], axis=1)
    test_score = sc.scorecard_ply(dt=df_test, card=card, only_total_score=True,
                               print_step=0, replace_blank_na=True, var_kp=None)
    df_test_withscore = pd.concat([df_test, test_score], axis=1)
    return card,card_df,df_train_withscore,df_test_withscore


def get_score_range(scorecard, vars_final):
    '''
    parameters：
    1.scorecard：scorecardpy生成的评分卡对象
    2.vars_final：模型最终的变量列表
    returns:
    1._min：评分卡的最低分
    2._max：评分卡的最高分
    3.card_df：dataframe格式的评分卡
    '''
    card_df = pd.DataFrame()
    basepoints = scorecard["basepoints"]["points"].sum()  #基础分
    _min, _max = basepoints, basepoints
    for var in vars_final:
        var_df = scorecard[var]
        card_df = pd.concat([card_df, var_df], axis=0)
        _min += var_df["points"].min()
        _max += var_df["points"].max()
    card_df.index = range(len(card_df))
    print("Scorecard score range: ", _min, "~", _max)
    return _min, _max, card_df
    
    
    
    
    