import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scorecardpy as sc


def bins_analysis(bins,eda_df,iv_limit=0.05,display = True):
    '''
    usage:
    根据分箱结果将变量分为5类，分别是低iv变量、未成功分箱变量、高IV且字符型变量，高iv且woe单调递减变量、高iv且woe单调递增变量、高iv但woe不单调变量；
    parameters:
    1.bins：dict，通过scorecardpy.woebin函数得到的分箱结果
    2.eda_df：dataframe，即探索性数据分析结果，根据该结果判断变量是否为字符型
    3.iv_limit：decimal，iv阈值，低于此阈值的变量被分类为低iv变量
    4.display：bool，是否对bins的分箱分析结果进行展示
    returns:
    1.vars_low_iv：list，低iv变量的列表
    2.vars_unsplited：list，未成功分箱变量的列表
    3.vars_cate：dict,高IV字符型变量的字典，包括分箱结果
    4.vars_monotonic_des：dict，高iv且woe单调递减变量的字典，包括分箱结果
    5.vars_monotonic_inc:dict, 高iv且woe单调递增变量的字典，包括分箱结果
    6.vars_nonmonotonic：dict，高iv但woe不单调变量的字典，包括分箱结果
    '''
    vars_unsplited = []  #未成功分箱变量
    vars_cate = {} #高iv的字符型变量
    vars_monotonic_des = {}  #高iv且woe单调递减变量
    vars_monotonic_inc = {}  #高iv且woe单调递增变量
    vars_nonmonotonic = {}  #高iv但woe不单调变量
    vars_low_iv = []  #筛选出低iv变量
    for var in bins.keys():
        var_type = eda_df.loc[var,'type']
        var_bin_woe = bins[var].query('bin != "missing"')["woe"]  #除去缺失值一箱后的变量woe值
        if len(var_bin_woe)  == 1:
            vars_unsplited.append(var)
        elif var_type in ['object', 'str', 'bool']:
            if bins[var]["bin_iv"].sum() >= iv_limit:
                vars_cate[var] = bins[var]
            else:
                vars_low_iv.append(var)        
        elif var_bin_woe.is_monotonic_decreasing:
            if bins[var]["bin_iv"].sum() >= iv_limit:
                vars_monotonic_des[var] = bins[var]
            else:
                vars_low_iv.append(var)
        elif var_bin_woe.is_monotonic_increasing:
            if bins[var]["bin_iv"].sum() >= iv_limit:
                vars_monotonic_inc[var] = bins[var]
            else:
                vars_low_iv.append(var)                   
        else:
            if bins[var]["bin_iv"].sum() >= iv_limit:
                vars_nonmonotonic[var] = bins[var]
            else:
                vars_low_iv.append(var)
    if display:                      
        print("未成功分箱的特征数量：", len(vars_unsplited))
        print("高iv字符型特征数量：",len(vars_cate))
        print("高iv且woe单调递减的特征数量：", len(vars_monotonic_des)) 
        print("高iv且woe单调递增的特征数量：", len(vars_monotonic_inc)) 
        print("高iv但woe不单调的特征数量：", len(vars_nonmonotonic))
        print("低iv特征数量：", len(vars_low_iv))                                               
    return vars_unsplited,vars_cate, vars_monotonic_des,vars_monotonic_inc, vars_nonmonotonic,vars_low_iv


def bins_auto_adjust(df, eda_df, bins, target ,auto_adj_method, exclude = [], special_values = None, bin_method="chimerge", iv_limit=0.05, cate_adj = True, display = True):
    '''
    usage:
    根据分箱结果将变量分为4类，分别是低iv变量、未成功分箱变量、高iv且woe单调变量、高iv但woe不单调变量；
    通过减少分箱数量对高iv但woe不单调变量进行分箱调整，输出最终结果。
    parameters:
    1.df：dataframe，进行分箱的原始数据，一般是包含x和y的训练集
    2.eda_df：dataframe，即探索性数据分析结果，根据该结果判断变量是否为字符型
    3.bins：dict，通过scorecardpy.woebin函数得到的分箱结果
    4.target：string，目标变量名称
    5.auto_adj_method:int,即数值型特征自动分箱方法选择，目前只有1
    6.exclude: list,不需要进行单调性处理的变量，如年龄等
    7.special_values: dict,需要单独分为一箱的特殊值字典。如：special_values = {'credit_amount': [2600, 9960, "6850%,%missing"],'purpose': ["education", "others%,%missing"]}
    8.bin_method：string，分箱方法，和scorecardpy.woebin函数的分箱方法保持一致
    9.iv_limit：decimal，iv阈值，低于此阈值的变量被分类为低iv变量
    10.cate_adj: bool,是否需要对字符型变量进行手动交互式分箱，默认需要
    11.display：bool，是否对自动调整前后的分箱分析结果进行展示
    returns:
    1.bins_adj：dict，留下来的变量分箱结果，包含三部分，一是高iv字符型特征；二是高iv单调变量；三是高iv不单调但不需转换的变量
    '''
    vars_unsplited,vars_cate, vars_monotonic_des,vars_monotonic_inc, vars_nonmonotonic,vars_low_iv = bins_analysis(bins,eda_df,iv_limit=iv_limit,display = display)
    #将所有变量分为6类后，交互式对高iv字符型变量进行分箱调整
    if cate_adj and len(vars_cate)!=0:
        cate_breaks_adj = woebin_adj(
                    dt = df,y = target,
                    adj_all_var = True,
                    count_distr_limit = 0.05,
                    bins = vars_cate,   ##选择非单调特征进行调整
                    method = bin_method)        
        bins_adj_for_cate = sc.woebin(df,y = target,breaks_list = cate_breaks_adj,method = bin_method,special_values = special_values,
                                        var_skip = exclude,count_distr_limit = 0.05,bin_num_limit = 8,check_cate_num=True,ignore_const_cols=True)
        cate_bins_adj = {key: value for key, value in bins_adj_for_cate.items() if key in vars_cate.keys()}
    else:
        cate_bins_adj = vars_cate
    
    if auto_adj_method == 1:
        #将所有变量分为6类后，调用自动调整函数1对高iv但woe不单调变量进行分箱自动调整（减少分箱数量） 
        bins_nonmono, bins_mono_inc, bins_mono_des = num_bins_auto_adj_1(df,target,bin_method,vars_nonmonotonic,vars_monotonic_inc,vars_monotonic_des,exclude)  
    elif auto_adj_method == 2:
        pass
    else:
        pass
    
    #将处理后的字典合并，并再次进行分箱结果分析
    dictMerged = cate_bins_adj.copy()
    dictMerged.update(bins_nonmono)
    dictMerged.update(bins_mono_inc)
    dictMerged.update(bins_mono_des)    
    vars_unsplited_new,vars_cate_new, vars_monotonic_des_new,vars_monotonic_inc_new, vars_nonmonotonic_new,vars_low_iv_new = bins_analysis(dictMerged,eda_df,iv_limit=iv_limit,display = display)
    #对分箱分析结果进行合并，留下来的用于后续的建模                              
    bins_adj = vars_cate_new.copy()
    bins_adj.update(vars_nonmonotonic_new)
    bins_adj.update(vars_monotonic_des_new)      
    bins_adj.update(vars_monotonic_inc_new)          
    return bins_adj


def num_bins_auto_adj_1(df,target,bin_method,bins_nonmono,bins_mono_inc,bins_mono_des,exclude=[]):
    '''
    usage:
    通过循环减少默认分箱数量的方法，将数值型且woe分箱不单调的特征调整为单调
    parameters:
    1.df：dataframe，进行分箱的原始数据，一般是包含x和y的训练集
    2.target：string，目标变量名称
    3.bin_method：string，分箱方法
    4.bins_nonmono：dict，将原始分箱结果，根据bins_analysis分箱分析函数分析后，输出的非单调数值型变量分箱子集
    5.bins_nonmono：dict，将原始分箱结果，根据bins_analysis分箱分析函数分析后，输出的非单调数值型变量分箱子集
    6.bins_nonmono：dict，将原始分箱结果，根据bins_analysis分箱分析函数分析后，输出的非单调数值型变量分箱子集
    7.exclude: list,不需要进行单调性处理的变量，如年龄等
    returns:
    1.bins_adj：dict，调整后的非单调分箱子集
    2.bins_mono_inc: dict, 调整后的单调递增分箱子集
    3.bins_mono_des: dict, 调整后的单调递减分箱子集
    '''
    nonmonotonic_var_list = list(bins_nonmono.keys())
    if len(nonmonotonic_var_list) == 0:
        print("不存在高iv但woe不单调的变量，无需调整！")
    else:
        for var in nonmonotonic_var_list:
            if var in exclude:
                print("不需要对变量 %s 进行单调处理！" % var)
            else:    
                var_bin_woe = bins_nonmono[var].query('bin != "missing"')["woe"]
                while True:
                    if var_bin_woe.is_monotonic_decreasing:
                        bins_mono_des[var] = bins_nonmono[var]
                        del bins_nonmono[var]  #不单调的特征字典需要删除，否则有重复
                        print("变量 %s 分箱后的woe已调整为单调递减" % var)
                        break  #满足woe单调性时停止分箱
                    elif var_bin_woe.is_monotonic_increasing:
                        bins_mono_inc[var] = bins_nonmono[var]
                        del bins_nonmono[var]  #不单调的特征字典需要删除，否则有重复
                        print("变量 %s 分箱后的woe已调整为单调递增" % var)
                        break  #满足woe单调性时停止分箱                        
                    else:
                        #减少最大分箱数后分箱，更新分箱结果bins
                        bins_nonmono[var] = sc.woebin(df[[var, target]], y=target,
                                              bin_num_limit=len(var_bin_woe)-1, method=bin_method)[var]
                        var_bin_woe = bins_nonmono[var].query('bin != "missing"')["woe"]
    return bins_nonmono,bins_mono_inc,bins_mono_des


def adjust_nan_bin(bins, var_skip=[], zero_woe=True):
    '''
    usage：原始分箱结果中有空值箱时，对空值箱的woe进行调整,没有空值箱时，增加空值箱
    parameters：
    1.bins：dict，scorecarpy.woebin的分箱结果
    2.var_skip：list，不参与空值箱调整的变量，通过是空值箱有业务含义的变量
    3.zero_woe：bool，True时将空值箱的woe赋为0，即认为空值组好坏分布与训练整体样本一致；否则作为最坏箱
    returns：
    1.new_bins：dict，经过调整的分箱结果
    '''
    new_bins = {}
    var_list = list(bins.keys())
    for var in var_list:
        var_bin = bins[var].copy()  #变量var的分箱结果，dataframe格式
        var_iv = var_bin["bin_iv"].sum()
        if len(var_skip)!=0 and var in var_skip:
            print("变量 %s 不参与空值箱调整" % var)
        else:
            if "missing" in var_bin["bin"].to_list():
                print("变量 %s 存在空值分箱，进行空值箱woe调整" % var)
                index = var_bin["bin"].to_list().index("missing")
                if zero_woe:
                    #将空值箱的woe和iv值赋为0
                    var_bin.loc[index,"woe"] = 0
                    #var_bin.loc[index,"bin_iv"] = 0
                    #var_bin["total_iv"] = var_bin["bin_iv"].sum()
                else:
                    #将空值箱的woe和iv值赋为woe值最大的箱的woe和iv值
                    #max_woe_index = var_bin["woe"].values.argmax()
                    var_bin.loc[index,"woe"] = var_bin["woe"].max()
                    #var_bin.loc[index,"bin_iv"] = var_bin.loc[max_woe_index,"bin_iv"]
                    #var_bin["total_iv"] = var_bin["bin_iv"].sum()
            else:
                print("变量 %s 没有空值分箱，新增一箱空值箱" % var)
                if zero_woe:
                    nan_bin = pd.DataFrame([var, 'missing', 0, 0.0, 0, 0, 0.0, 0.0, 0.0, var_iv, 'missing', True],
                                     index=['variable','bin','count','count_distr','good','bad','badprob','woe','bin_iv','total_iv','breaks','is_special_values']).T
                else:
                    max_woe = var_bin["woe"].max()
                    nan_bin = pd.DataFrame([var, 'missing', 0, 0.0, 0, 0, 0.0, max_woe, 0.0, var_iv, 'missing', True],
                                     index=['variable','bin','count','count_distr','good','bad','badprob','woe','bin_iv','total_iv','breaks','is_special_values']).T
                for col in nan_bin.columns:
                        nan_bin[col] = nan_bin[col].astype(var_bin.dtypes[col])
                var_bin = pd.concat([nan_bin, var_bin], axis=0)
                var_bin.index = range(len(var_bin))
        new_bins[var] = var_bin
    return new_bins


def adjust_bins_decimal(df, bins, decimal=3):
    '''
    usage：调整分箱分割点精度至小数点后三位
    parameters：
    1.df：dataframe，原始数据集
    2.bins：dict，scorecarpy.woebin的分箱结果
    3.decimal：int，需要调整的分箱分割点精度
    returns：
    1.new_bins：dict，经过精度调整的分箱结果
    '''
    new_bins = {}
    for var in bins.keys():
        var_bin = bins[var]  #变量原始分箱结果
        if df[var].dtype.kind != 'f':
            new_bins[var] = var_bin
        else:
            #仅对float类型变量进行调整
            for i in range(len(var_bin)):
                #先判断分割点是否为missing特殊值
                if not var_bin.loc[i, "is_special_values"]:
                    var_bin.loc[i, 'breaks'] = str(round(float(var_bin.loc[i, 'breaks']), decimal))
            new_bins[var] = var_bin
    return new_bins


def get_df_with_woe(df_train,df_test,final_bins,eda_df,target,iv_limit = 0.05,display = True,plot_woe = True):
    '''
    usage：将原始值的数据框，根据分箱结果，替换为woe值组成的数据框
    parameters：
    1.df_train：dataframe，训练集
    
    2.df_test：dataframe，测试集    
    3.final_bins：dict，自动或手动调整后生成的新分箱文件，且已经经过woe人工变更
    4.eda_df：dataframe，即探索性数据分析结果，根据该结果判断变量是否为字符型
    5.iv_limit：decimal，iv阈值，低于此阈值的变量被分类为低iv变量
    6.display：bool，是否对final_bins的分箱分析结果进行展示
    7.plot_woe：bool，是否对筛选后用于woe替换的分箱结果进行绘图展示    
    returns：
    1.df_train_woe：dataframe，woe替换后的训练集文件
    2.df_test_woe：dataframe，woe替换后的测试集文件    
    '''
    #对最终分箱结果进行分析,由于调整分箱后，iv会降低，这一步可以将分箱后iv不满足要求的特征再次剔除
    vars_unsplited,vars_cate, vars_monotonic_des,vars_monotonic_inc, vars_nonmonotonic,vars_low_iv = bins_analysis(final_bins,eda_df,iv_limit=iv_limit,display = display)
    #剔除低iv、未成功分箱的特征
    drop_list = list(set(vars_low_iv + vars_unsplited))
    df_train.drop(drop_list, axis=1, inplace=True)
    df_test.drop(drop_list, axis=1, inplace=True)
    #将分箱调整后的vars_monotonic_des\vars_monotonic_inc和可以接受的非单调分箱vars_nonmonotonic,以及离散型分箱vars_cate进行合并，用于woe转换
    bins_for_ply = vars_cate.copy()
    bins_for_ply.update(vars_monotonic_des)
    bins_for_ply.update(vars_monotonic_inc)
    bins_for_ply.update(vars_nonmonotonic)
    #scorecardpy中的woebin_ply函数，根据分箱结果(bins)将原始数据(dt)转换为woe
    df_train_woe = sc.woebin_ply(dt=df_train, bins=bins_for_ply)
    df_test_woe = sc.woebin_ply(dt=df_test, bins=bins_for_ply)
    if plot_woe:
        sc.woebin_plot(bins=bins_for_ply, x=None, title=None, show_iv=True); 
    bins_var_list = list(bins_for_ply.keys())
    bins_var_list1 = [i + "_woe" for i in bins_var_list]
    bins_var_list1.append(target) 
    df_train_woe_new = df_train_woe[bins_var_list1]
    df_test_woe_new = df_test_woe[bins_var_list1]
    return df_train_woe_new,df_test_woe_new


##针对woebin_adj函数的定制化修改
from pandas.api.types import is_numeric_dtype
import re

def check_special_values(special_values, xs):
    if special_values is not None:
        # # is string
        # if isinstance(special_values, str):
        #     special_values = eval(special_values)
        if isinstance(special_values, list):
            warnings.warn("The special_values should be a dict. Make sure special values are exactly the same in all variables if special_values is a list.")
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict): 
            raise Exception("Incorrect inputs; special_values should be a list or dict.")
    return special_values


def bins_to_breaks(bins, dt, to_string=False, save_string=None):
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)

    # x variables
    xs_all = bins['variable'].unique()
    # dtypes of  variables
    vars_class = pd.DataFrame({
      'variable': xs_all,
      'not_numeric': [not is_numeric_dtype(dt[i]) for i in xs_all]
    })
    
    # breakslist of bins
    bins_breakslist = bins[~bins['breaks'].isin(["-inf","inf","missing"]) & ~bins['is_special_values']]
    bins_breakslist = pd.merge(bins_breakslist[['variable', 'breaks']], vars_class, how='left', on='variable')
    bins_breakslist.loc[bins_breakslist['not_numeric'], 'breaks'] = '\''+bins_breakslist.loc[bins_breakslist['not_numeric'], 'breaks']+'\''
    bins_breakslist = bins_breakslist.groupby('variable')['breaks'].agg(lambda x: ','.join(x))
    
    if to_string:
        bins_breakslist = "breaks_list={\n"+', \n'.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
        if save_string is not None:
            brk_lst_name = '{}_{}.py'.format(save_string, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
            with open(brk_lst_name, 'w') as f:
                f.write(bins_breakslist)
            print('[INFO] The breaks_list is saved as {}'.format(brk_lst_name))
            return 
    return bins_breakslist

# print basic information in woebin_adj
def woebin_adj_print_basic_info(i, xs, bins, dt, bins_breakslist):
    '''
    print basic information of woebinnig in adjusting process
    
    Params
    ------
    
    Returns
    ------
    
    
    '''
    x_i = xs[i-1]
    xs_len = len(xs)
    binx = bins.loc[bins['variable']==x_i]
    print("--------", str(i)+"/"+str(xs_len), x_i, "--------")
    # print(">>> dt["+x_i+"].dtypes: ")
    # print(str(dt[x_i].dtypes), '\n')
    # 
    print(">>> dt["+x_i+"].describe(): ")
    print(dt[x_i].describe(), '\n')
    
    if len(dt[x_i].unique()) < 10 or not is_numeric_dtype(dt[x_i]):
        print(">>> dt["+x_i+"].value_counts(): ")
        print(dt[x_i].value_counts(), '\n')
    else:
        dt[x_i].hist()
        plt.title(x_i)
        plt.show()
        
    ## current breaks
    print(">>> Current breaks:")
    print(bins_breakslist[x_i], '\n')
    ## woebin plotting
    plt.show(sc.woebin_plot(binx)[x_i])

# plot adjusted binning in woebin_adj
def woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method):
    '''
    update breaks and provies a binning plot
    
    Params
    ------
    
    Returns
    ------
    
    '''
    if breaks == '':
        breaks = None
    breaks_list = None if breaks is None else {x_i: eval('['+breaks+']')}
    special_values = None if sv_i is None else {x_i: sv_i}
    # binx update
    bins_adj = sc.woebin(dt[[x_i,y]], y, breaks_list=breaks_list, special_values=special_values, stop_limit = stop_limit, method=method)
    
    ## print adjust breaks
    breaks_bin = set(bins_adj[x_i]['breaks']) - set(["-inf","inf","missing"])
    breaks_bin = ', '.join(breaks_bin) if is_numeric_dtype(dt[x_i]) else ', '.join(['\''+ i+'\'' for i in breaks_bin])
    print(">>> Current breaks:")
    print(breaks_bin, '\n')
    # print bin_adj
    plt.show(sc.woebin_plot(bins_adj))
    # return breaks 
    if breaks == '' or breaks is None: breaks = breaks_bin
    return breaks

def woebin_adj(dt, y, bins, adj_all_var=False, special_values=None, method="tree", save_breaks_list=None, count_distr_limit=0.05):
    '''
    WOE Binning Adjustment
    ------
    `woebin_adj` interactively adjust the binning breaks.
    
    Params
    ------
    dt: A data frame.
    y: Name of y variable.
    bins: A list or data frame. Binning information generated from woebin.
    adj_all_var: Logical, whether to show monotonic woe variables. Default
      is True
    special_values: the values specified in special_values will in separate 
      bins. Default is None.
    method: optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    save_breaks_list: The file name to save breaks_list. Default is None.
    count_distr_limit: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.
    
    Returns
    ------
    dict
        dictionary of breaks
        
    '''
    # bins concat 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_all = bins['variable'].unique()
    # adjust all variables
    if not adj_all_var:
        bins2 = bins.loc[~((bins['bin'] == 'missing') & (bins['count_distr'] >= count_distr_limit))].reset_index(drop=True)
        bins2['badprob2'] = bins2.groupby('variable').apply(lambda x: x['badprob'].shift(1)).reset_index(drop=True)
        bins2 = bins2.dropna(subset=['badprob2']).reset_index(drop=True)
        bins2 = bins2.assign(badprob_trend = lambda x: x.badprob >= x.badprob2)
        xs_adj = bins2.groupby('variable')['badprob_trend'].nunique()
        xs_adj = xs_adj[xs_adj>1].index
    else:
        xs_adj = xs_all
    # length of adjusting variables
    xs_len = len(xs_adj)
    # special_values
    special_values = check_special_values(special_values, xs_adj)
    
    # breakslist of bins
    bins_breakslist = bins_to_breaks(bins,dt)
    # loop on adjusting variables
    if xs_len == 0:
        warnings.warn('The binning breaks of all variables are perfect according to default settings.')
        breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
        return breaks_list
    # else 
    def menu(i, xs_len, x_i):
        print('>>> Adjust breaks for ({}/{}) {}?'.format(i, xs_len, x_i))
        print('1: next \n2: yes \n3: back')
        adj_brk = input("Selection: ")
        while isinstance(adj_brk,str):
            if str(adj_brk).isdigit():
                adj_brk = int(adj_brk)
                if adj_brk not in [0,1,2,3]:
                    warnings.warn('Enter an item from the menu, or 0 to exit.')               
                    adj_brk = input("Selection: ")  
            else: 
                print('Input could not be converted to digit.')
                adj_brk = input("Selection: ") #update by ZK 
        return adj_brk
        
    # init param
    i = 1
    breaks_list = None
    while i <= xs_len:
        breaks = stop_limit = None
        # x_i
        x_i = xs_adj[i-1]
        sv_i = special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None
        # if sv_i is not None:
        #     sv_i = ','.join('\'')
        # basic information of x_i variable ------
        woebin_adj_print_basic_info(i, xs_adj, bins, dt, bins_breakslist)
        # adjusting breaks ------
        adj_brk = menu(i, xs_len, x_i)
        if adj_brk == 0: 
            return 
        
        while adj_brk == 2:
            # modify breaks adj_brk == 2
            breaks = input(">>> Enter modified breaks: ")
            breaks = re.sub("^[,\.]+|[,\.]+$", "", breaks)
            if breaks == 'N':
                stop_limit = 'N'
                breaks = None
            else:
                stop_limit = 0.1
            try:
                breaks = woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method=method)
                print('******变量 %s 的（最新）分箱分割点为：[%s]' % (x_i, breaks))
            except:
                pass
            # adj breaks again
            adj_brk = menu(i, xs_len, x_i)        
            
        if adj_brk == 3:
            # go back adj_brk == 3
            i = i-1 if i>1 else i
        else:
            # go next adj_brk == 1
            if breaks is not None and breaks != '': 
                bins_breakslist[x_i] = breaks
            print('******变量 %s 的【最终】分箱分割点为：[%s]' % (x_i, bins_breakslist[x_i]))
            i += 1

    # return 
    breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
    if save_breaks_list is not None:
        bins_adj = woebin(dt, y, x=bins_breakslist.index, breaks_list=breaks_list)
        bins_to_breaks(bins_adj, dt, to_string=True, save_string=save_breaks_list)
    return breaks_list






















