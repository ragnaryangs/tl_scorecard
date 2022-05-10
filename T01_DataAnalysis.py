import toad
import numpy as np

def display_df_info(df,n,target = 'bad'):
    '''    
    usage:
    对数据集的维度及好坏分布作初步展示。
    parameters:
    1.df：包含目标y和特征x的宽表
    2.n：展示数据集的行数
    3.target：df中目标变量的列名
    returns:
    '''
    print("行列数：",df.shape)
    display(df[target].value_counts(normalize = False))
    display(df.head(n))

def get_new_eda(df):
    '''    
    usage:
    对toad中的数据探查结果eda做更改：
        1是将missing改为可比较的decimal格式
        2是增加空值列
        3是增加出现最多的单一值
        4是增加出现最多的单一值占比
        5是增加数值型变量的负值占比，非数值型不参与
    parameters:
    1.df：包含目标y和特征x的宽表
    returns:
    1.eda：新的eda表
    '''
    eda = toad.detect(df)
    eda["missing"] = eda["missing"].apply(percent_to_float)  #缺失率为百分数格式，将其转化为decimal
    eda["zero"] = [len(df.loc[(df[i]==0) | (df[i]=='0')]) / len(df) for i in eda.index]  #添加"0值占比"的分析    
    eda['top1'] = [get_top1(df,i) for i in eda.index]
    eda['top1_ratio'] = [get_top1_ratio(df,i) for i in eda.index]
    eda['negative_ratio'] = [len(df.loc[(df[i] < 0)]) / len(df) if df[i].dtype.kind in 'ifc' else np.nan for i in eda.index]
    eda = eda[['type', 'size', 'missing', 'unique','zero','negative_ratio','top1','top1_ratio','mean_or_top1', 'std_or_top2',
       'min_or_top3', '1%_or_top4', '10%_or_top5', '50%_or_bottom5',
       '75%_or_bottom4', '90%_or_bottom3', '99%_or_bottom2', 'max_or_bottom1']]
    display(eda)    #展示EDA结果
    return eda
    
def get_top1(df,i):
    '''
    usage：提取数据框中某列，出现最多的单一值（不包含空值）
    ''' 
    if all(df[i].isna()):
        return np.nan
    else:       
        var_sort = df[i].value_counts(normalize=True, dropna=True).reset_index()
        var_top1 = var_sort.iloc[0,0]
        return var_top1

def get_top1_ratio(df,i):
    '''
    usage：提取数据框中某列，出现最多的单一值占比（不包含空值）
    '''
    if all(df[i].isna()):
        return np.nan
    else:
        var_sort = df[i].value_counts(normalize=True, dropna=True).reset_index()
        var_top1_ratio = var_sort.iloc[0,1]
        return var_top1_ratio    

def percent_to_float(s):
    '''
    usage：将带有%字符串形式的数据转化为浮点数。
    '''
    if "%" in s:
        new_s = float(s.strip("%")) / 100
        return new_s
    else:
        print("输入的不是百分数!")

