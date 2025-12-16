#!/usr/bin/env python
# coding: utf-8
import os.path

#导入需要使用的库
import pandas as pd
import numpy as np



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

#核心步骤
#1 读取财务数据，整理出财务因子
#2 读取量价数据，整理出量价因子
#3 合并两个数据表，对因子做因子线性关系分析
#4 使用线性模型得到股票评分表
#5 使用画图工具画出组合收益曲线


# 选取4个需要的财务数据因子--------------------------------------

fact_list = [
# 计算总资产对经营现金流做市值中性化
'Total non-current assets' ,#非流动资产
'Total current assets', #流动资产
'Net cash flows from (used in) operating activities',#经营现金流
'Total basic earnings per share'#每股盈余
]

# 指定需要处理的因子列
factors_to_process = [
    'trading_money_std_20d',
    'ret_20d_std',
    'Total basic earnings per share',
    'NCFF per'
]



#- 读取测试文件数据
def load_examination_data(path,filename,isPrice=False):

    """
        参数:
        path:数据文件夹的路径
        返回:
        DataFrame: 文件的dataFrame，或者None
    """

    #当前文件路径
    cur_file_path = os.path.join(path,filename)

    if os.path.exists(cur_file_path):
        #读取财务报表------------------------------------------------------------------------------
        finial_raw_df = pd.read_csv(cur_file_path)

        if len(finial_raw_df)>0:
            #保留需要的列名
            if isPrice:
                pass
            else:
                finial_raw_df = finial_raw_df[['number','year','period','key_en','value']]
            return finial_raw_df
        else:
            return None
    else:
        print(f"不存在文件{cur_file_path}")
        return None

#转化为视图
def tran2pivot(finial_raw_df):


    """
        参数:
        finial_raw_df:数据文件
        返回:
        DataFrame: 视图文件的dataFrame，或者None
    """

    if finial_raw_df is None:
        return None

    else:
        # 整理成常用的因子数据格式------------------------------------------------------------------------------
        # 将value转换为数值类型
        finial_raw_df['value'] = pd.to_numeric(finial_raw_df['value'], errors='coerce')
        # 创建透视表：以number, period, year为索引，key_en为列，value为值
        finial_pivot_df = finial_raw_df.pivot_table(
            index=['number', 'period', 'year'],
            columns='key_en',
            values='value',
            aggfunc='first'
        )
        # 重置索引，将number, period, year转为列
        finial_pivot_df = finial_pivot_df.reset_index()

        # 按number, period, year排序
        finial_pivot_df = finial_pivot_df.sort_values(by=['number', 'year', 'period'])

        # 重命名列（可选，使列名更清晰）
        finial_pivot_df.columns.name = None

        return finial_pivot_df

#选定因子
def  select_factor(finial_pivot_df):


    """
        参数:
        finial_pivot_df:待筛选因子视图
        返回:
        DataFrame: 筛选因子后的dataFrame，或者None
    """
    if finial_pivot_df is not None:

        # 选取需要的因子计算出财务因子，并且做简单市值中性化，没有行业数据无法做行业中性化------------------------------------------------------------------------------
        # 这里按照常用业界经验选取因子演示
        selet_fact_df = finial_pivot_df[['number',	'year',	'period'] + fact_list]
        #对股票排序
        selet_fact_df = selet_fact_df.sort_values(['number','year','period'],ascending=[True,True,True])

        # 其中Total basic earnings per share本身已经有市值中性化的效果，对 Net cash flows from (used in) operating activities做市值中性化
        selet_fact_df['NCFF per'] = selet_fact_df['Net cash flows from (used in) operating activities']/(
                                    selet_fact_df['Total non-current assets']+selet_fact_df['Total current assets'])
        return selet_fact_df
    else:
        return None

#整理因子
def collect_factor_data(selet_fact_df):


    """
        参数:
        selet_fact_df:待筛选因子视图
        返回:
        DataFrame: 整理因子后的dataFrame，或者None
    """


    if selet_fact_df is not None:

        # 整理财务因子数据日期-----------------------------------------------------------------------------
        # 使用财务因子需要给出财报发布日，给的报表里面没有发布日，这里假设发布日为财报截止的下个月最后一天
        quarter_map = {
            1: ('04', '30'),  # 1季度 -> 3月31日
            2: ('07', '31'),  # 2季度 -> 6月30日
            3: ('10', '31'),  # 3季度 -> 9月30日
            4: ('01', '31')   # 4季度 -> 12月31日
        }

        #财报如果跨年，pub_year需要year+1
        selet_fact_df['pub_year'] = np.where(selet_fact_df['period'] == 4, selet_fact_df['year'] + 1, selet_fact_df['year'])
        selet_fact_df['date'] = selet_fact_df.apply(
            lambda row: f"{int(row['pub_year'])}-{quarter_map[row['period']][0]}-{quarter_map[row['period']][1]}",
            axis=1
        )
        selet_fact_df = selet_fact_df[['number', 'date', 'Total basic earnings per share', 'NCFF per']].copy()
        selet_fact_df.rename(columns={'number': 'stock_id'}, inplace=True)
        selet_fact_df['date'] = selet_fact_df['date'].astype('str')

        return selet_fact_df
    else:
        return None



#通过股票价格计算因子值
def cal_factor_by_price(raw_price_df):
    """
                参数:
                raw_price_df:股票价格数据
                返回:
                DataFrame: 计算因子的dataFrame，或者None
    """


    if raw_price_df is not None:





        # 对股票数据做整理方便后面计算量价因子和合并财务因子--------------------------------------------------------------------------------
        #对股票和日期排序
        sort_price_df = raw_price_df.sort_values(['stock_id','date'], ascending=[True, True])
        # 转换日期格式
        sort_price_df['date']=sort_price_df['date'].apply(lambda x: str(x)[:10])
        #保留需要的列数
        sort_price_df = sort_price_df[['date','stock_id','trading_money','close','open']]
        sort_price_df = sort_price_df.set_index('stock_id')
        sort_price_df.head()


        # In[11]:


        #计算2个量价因子---------------------------------------------------------------------------------------------------------------
        #第一个因子，过去20天的交易额标准差
        sort_price_df['trading_money_std_20d'] = sort_price_df.groupby('stock_id')['trading_money']\
            .transform(lambda x: x.rolling(window=20, min_periods=20).std())

        return sort_price_df
    else:

        return None





#第二个因子，过去20天的收益率标准差
def calculate_volatility_factor(df, window=30, close_col='close', id_col='stock_id', date_col='date'):
    """
    计算股票波动率因子：std( (close / shift(close,1) - 1), window )
    
    参数:
    df (DataFrame): 输入数据，必须包含id_col, date_col, close_col
    window (int): 滚动窗口大小 (默认30日)
    close_col (str): 收盘价列名 (默认'close')
    id_col (str): 股票ID列名 (默认'id')
    date_col (str): 日期列名 (默认'date')
    
    返回:
    DataFrame: 包含原数据 + 新因子列 (格式: f'factor_{window}d_std')
    """
    # 创建数据副本避免修改原始数据
    #df = df.copy()
    
    # 1. 确保日期列是datetime类型
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. 按股票ID分组并按日期排序
    df = df.sort_values(by=[id_col, date_col])
    
    # 3. 计算日收益率 (ret = (close_t / close_{t-1}) - 1)
    df['ret'] = df.groupby(id_col)[close_col].transform(
        lambda x: x / x.shift(1) - 1
    )
    
    # 4. 计算滚动标准差
    factor_name = f'ret_{window}d_std'
    df[factor_name] = df.groupby(id_col)['ret'].transform(
        lambda x: x.rolling(window=window, min_periods=window).std()
    )
    
    # 5. 清理临时列
    df = df.drop(columns=['ret'])
    
    return df




#未来10天的收益率
def calculate_Return(df, Ci_col, Cj_col, i, j):
    # 计算收益率的值
    Ci = df[Ci_col].shift(-i)
    Cj = df[Cj_col].shift(-j)
    df_r = (Cj - Ci) / Ci  # 收益率
    # 返回一个 Series
    return df_r


# 求每天因子与收益率相关系数
def get_fac_IC_df(df, fac_list, price_col='open'):
    '''
    返回因子的ICIR
    '''
    # df['returnO1O6'] = df.groupby('instrument',group_keys = False).apply(lambda x:calculate_Return(x, price_col,price_col,1,6))
    # 求每天因子与收益率相关系数
    IC_df = pd.DataFrame()
    for name in fac_list:  # factor_dict.keys() 因子名
        temp_df = df.set_index('date').groupby('date', group_keys=False).apply(
            lambda x: dataIC(x, name, 'returnO1O11', ''))

        IC_df = pd.concat([temp_df, IC_df], axis=1)

    Stats_df = pd.DataFrame(getColStats(IC_df, returnData=('mean', 'std', 'zScore', 'absGt0.02_p', 'Gt0_p'))).T
    Stats_df.columns = ['IC', 'std', 'ICIR', 'absGt0.02_p', 'Gt0_p']
    return IC_df.mean(), IC_df.mean() / IC_df.std(), Stats_df


# 求一个横截面相关系数
def dataIC(data, fName, retName, other, method='spearman'):
    """
    相关系数方法：method = 'pearson'/'spearman'/'Kendall Tau'
    """
    IC_df = data[fName].corr(data[retName], method=method)

    IC_df = pd.DataFrame(IC_df, columns=['%s' % (fName)], index=[data.index.get_level_values(0).unique()[0]])
    IC_df.index.name = 'date'
    return IC_df


def getColStats(col, returnData=('mean', 'std', 'zScore', 'absGt2_p', 'Gt0_p')):
    # 计算因子数据
    statsMethodDict = {
        'mean': lambda col: col.mean(),
        'std': lambda col: col.std(),
        'zScore': lambda col: col.mean() / col.std(),
        'absGt2_p': lambda col: (abs(col) > 2).sum() / len(col),
        'absGt0.02_p': lambda col: (abs(col) > 0.02).sum() / len(col),
        'Gt0_p': lambda col: ((col) > 0).sum() / len(col),
        'nan_p': lambda col: col.isna().sum() / len(col)
    }
    if isinstance(returnData, str): return statsMethodDict[returnData](col)
    return list(map(lambda x: statsMethodDict[x](col), returnData))


def process_factors(df, factors, window=20, date_col='date'):
    """
    按日期分组处理因子列：去极值 → 标准化 → NA填充0

    参数:
    df (DataFrame): 输入数据（必须包含date_col和指定的factors列）
    factors (list): 需要处理的因子列名列表
    window (int): 滚动窗口大小（用于去极值，这里用20日，但实际去极值不依赖窗口）
    date_col (str): 日期列名（默认'date'）

    返回:
    DataFrame: 处理后的DataFrame（原数据+处理后的因子列）
    """
    # 1. 创建副本避免修改原始数据
    df = df.copy()

    # 2. 确保日期列是字符串格式（YYYY-MM-DD）
    df[date_col] = df[date_col].astype(str)

    # 3. 用0填充所有因子列的NA（确保后续计算不产生NA）
    df[factors] = df[factors].fillna(0)

    # 4. 按日期分组处理每个因子
    for factor in factors:
        # a. 去极值（使用0.05和0.95分位数）
        q_low = df.groupby(date_col)[factor].transform(lambda x: x.quantile(0.05))
        q_high = df.groupby(date_col)[factor].transform(lambda x: x.quantile(0.95))
        df[factor] = df[factor].clip(lower=q_low, upper=q_high)

        # b. 标准化（减均值，除以标准差，避免除以0）
        mean = df.groupby(date_col)[factor].transform('mean')
        std = df.groupby(date_col)[factor].transform('std')
        std = std.replace(0, 1)  # 避免除以0
        df[factor] = (df[factor] - mean) / std

        # c. 确保无NA（理论上不会产生，但保险起见）
        df[factor] = df[factor].fillna(0)

    return df


def preprocess_data(processed_df):
    """
    预处理数据：处理缺失值和无穷值
    """
    processed_df_clean = processed_df.copy()
    
    # 确保日期是datetime类型
    processed_df_clean['date'] = pd.to_datetime(processed_df_clean['date'])
    
    # 处理returnO1O2中的缺失值和无穷值
    processed_df_clean['returnO1O2'] = processed_df_clean['returnO1O2'].replace([np.inf, -np.inf], np.nan)
    processed_df_clean['returnO1O2'] = processed_df_clean['returnO1O2'].fillna(0)
    
    # 处理sum_alpha中的缺失值
    processed_df_clean['sum_alpha'] = processed_df_clean['sum_alpha'].fillna(-1e10)
    
    return processed_df_clean

def calculate_portfolio_returns_with_transaction_cost(processed_df, top_n=20, transaction_cost_rate=0.0002):
    """
    计算两个投资组合的日收益率，考虑交易手续费
    transaction_cost_rate: 单边交易手续费率，默认万2（0.02%）
    """
    processed_df_clean = preprocess_data(processed_df)
    
    # 按日期分组计算收益率
    dates = sorted(processed_df_clean['date'].unique())
    portfolio_returns = pd.DataFrame(index=dates, columns=['top20_return', 'all_return', 'turnover_rate'])
    
    # 存储前一日的持仓股票，用于计算换手率
    prev_day_top_stocks = None
    
    for i, date in enumerate(dates):
        # 获取当日数据
        daily_data = processed_df_clean[processed_df_clean['date'] == date]
        
        if len(daily_data) > 0:
            # 1. 计算Top 20组合的收益率
            # 按sum_alpha降序排序，取前top_n个
            top20_stocks = daily_data.nlargest(min(top_n, len(daily_data)), 'sum_alpha')
            current_top_stocks = set(top20_stocks['stock_id'].tolist())
            
            # 计算换手率（如果是第一天，换手率为100%，因为全部是新买入）
            if prev_day_top_stocks is None:
                turnover_rate = 1.0  # 第一天全部买入
            else:
                # 计算持仓变化：卖出的股票数/总持仓数
                sold_stocks = prev_day_top_stocks - current_top_stocks
                turnover_rate = len(sold_stocks) / len(prev_day_top_stocks) if len(prev_day_top_stocks) > 0 else 0
            
            portfolio_returns.loc[date, 'turnover_rate'] = turnover_rate
            
            if len(top20_stocks) > 0:
                # 计算等权重平均收益率
                price_return = top20_stocks['returnO1O2'].mean()
                
                # 计算交易成本：卖出旧持仓 + 买入新持仓
                # 假设双边手续费各0.02%，总手续费率 = 换手率 * 0.04%
                # 卖出时支付手续费，买入时支付手续费
                total_transaction_cost = turnover_rate * (transaction_cost_rate * 2)
                
                # 净收益率 = 价格收益率 - 交易成本
                top20_return = price_return - total_transaction_cost
                
                # 避免负的净收益率过大
                if top20_return < -0.5:  # 限制最大亏损
                    top20_return = -0.5
            else:
                top20_return = 0
            
            # 2. 计算全市场组合的收益率（不考虑调仓成本）
            all_return = daily_data['returnO1O2'].mean()
            
            portfolio_returns.loc[date, 'top20_return'] = top20_return
            portfolio_returns.loc[date, 'all_return'] = all_return
            
            # 更新前一日的持仓股票
            prev_day_top_stocks = current_top_stocks
        else:
            portfolio_returns.loc[date, 'top20_return'] = 0
            portfolio_returns.loc[date, 'all_return'] = 0
            portfolio_returns.loc[date, 'turnover_rate'] = 0
    
    # 填充缺失值
    portfolio_returns['top20_return'] = portfolio_returns['top20_return'].fillna(0)
    portfolio_returns['all_return'] = portfolio_returns['all_return'].fillna(0)
    portfolio_returns['turnover_rate'] = portfolio_returns['turnover_rate'].fillna(0)
    
    return portfolio_returns

def calculate_portfolio_returns_no_transaction_cost(processed_df, top_n=20):
    """
    不考虑交易手续费的版本，用于对比
    """
    processed_df_clean = preprocess_data(processed_df)
    
    # 按日期分组计算收益率
    dates = sorted(processed_df_clean['date'].unique())
    portfolio_returns = pd.DataFrame(index=dates, columns=['top20_return', 'all_return'])
    
    for date in dates:
        # 获取当日数据
        daily_data = processed_df_clean[processed_df_clean['date'] == date]
        
        if len(daily_data) > 0:
            # 1. 计算Top 20组合的收益率
            # 按sum_alpha降序排序，取前top_n个
            top20_stocks = daily_data.nlargest(min(top_n, len(daily_data)), 'sum_alpha')
            
            if len(top20_stocks) > 0:
                # 计算等权重平均收益率
                top20_return = top20_stocks['returnO1O2'].mean()
            else:
                top20_return = 0
            
            # 2. 计算全市场组合的收益率
            all_return = daily_data['returnO1O2'].mean()
            
            portfolio_returns.loc[date, 'top20_return'] = top20_return
            portfolio_returns.loc[date, 'all_return'] = all_return
        else:
            portfolio_returns.loc[date, 'top20_return'] = 0
            portfolio_returns.loc[date, 'all_return'] = 0
    
    # 填充缺失值
    portfolio_returns['top20_return'] = portfolio_returns['top20_return'].fillna(0)
    portfolio_returns['all_return'] = portfolio_returns['all_return'].fillna(0)
    
    return portfolio_returns

def calculate_cumulative_returns(portfolio_returns, initial_value=1.0):
    """
    计算累计收益率
    """
    cum_returns = portfolio_returns.copy()
    
    # 计算累计净值
    cum_returns['top20_cum'] = (1 + cum_returns['top20_return']).cumprod() * initial_value
    cum_returns['all_cum'] = (1 + cum_returns['all_return']).cumprod() * initial_value
    
    # 计算超额累计收益率 (Top20 - All)
    cum_returns['excess_cum'] = cum_returns['top20_cum'] - cum_returns['all_cum']
    
    return cum_returns

def analyze_portfolio_performance(cum_returns, portfolio_returns, initial_value=1.0, risk_free_rate=0.03):
    """
    分析投资组合表现
    """
    # 计算累计收益率
    top20_total_return = (cum_returns['top20_cum'].iloc[-1] - initial_value) / initial_value
    all_total_return = (cum_returns['all_cum'].iloc[-1] - initial_value) / initial_value
    
    # 计算年化收益率
    dates = cum_returns.index
    if len(dates) > 1:
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        years = (end_date - start_date).days / 365.25
        
        if years > 0:
            top20_annual_return = (1 + top20_total_return) ** (1 / years) - 1
            all_annual_return = (1 + all_total_return) ** (1 / years) - 1
        else:
            top20_annual_return = 0
            all_annual_return = 0
    else:
        top20_annual_return = 0
        all_annual_return = 0
    
    # 计算年化波动率
    if len(portfolio_returns) > 1:
        top20_annual_vol = portfolio_returns['top20_return'].std() * np.sqrt(min(252, len(portfolio_returns)))
        all_annual_vol = portfolio_returns['all_return'].std() * np.sqrt(min(252, len(portfolio_returns)))
    else:
        top20_annual_vol = 0
        all_annual_vol = 0
    
    # 计算夏普比率
    if top20_annual_vol > 0:
        top20_sharpe = (top20_annual_return - risk_free_rate) / top20_annual_vol
    else:
        top20_sharpe = 0
    
    if all_annual_vol > 0:
        all_sharpe = (all_annual_return - risk_free_rate) / all_annual_vol
    else:
        all_sharpe = 0
    
    # 计算最大回撤
    def calculate_max_drawdown(cum_series):
        if len(cum_series) < 2:
            return 0
        cummax = cum_series.cummax()
        drawdown = (cum_series - cummax) / cummax
        return drawdown.min()
    
    top20_max_dd = calculate_max_drawdown(cum_returns['top20_cum'])
    all_max_dd = calculate_max_drawdown(cum_returns['all_cum'])
    
    # 计算胜率
    top20_win_rate = (portfolio_returns['top20_return'] > 0).sum() / len(portfolio_returns)
    all_win_rate = (portfolio_returns['all_return'] > 0).sum() / len(portfolio_returns)
    
    # 计算超额收益统计
    excess_returns = portfolio_returns['top20_return'] - portfolio_returns['all_return']
    excess_total_return = (cum_returns['excess_cum'].iloc[-1] - 0) / 1.0
    excess_annual_return = (1 + excess_total_return) ** (1 / years) - 1 if years > 0 else 0
    excess_sharpe = excess_annual_return / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
    excess_win_rate = (excess_returns > 0).sum() / len(excess_returns)
    
    # 计算换手率统计
    if 'turnover_rate' in portfolio_returns.columns:
        avg_turnover = portfolio_returns['turnover_rate'].mean()
        avg_annual_turnover = avg_turnover * 252
    else:
        avg_turnover = 0
        avg_annual_turnover = 0
    
    return {
        'top20': {
            'total_return': top20_total_return,
            'annual_return': top20_annual_return,
            'annual_volatility': top20_annual_vol,
            'sharpe_ratio': top20_sharpe,
            'max_drawdown': top20_max_dd,
            'win_rate': top20_win_rate
        },
        'all': {
            'total_return': all_total_return,
            'annual_return': all_annual_return,
            'annual_volatility': all_annual_vol,
            'sharpe_ratio': all_sharpe,
            'max_drawdown': all_max_dd,
            'win_rate': all_win_rate
        },
        'excess': {
            'total_return': excess_total_return,
            'annual_return': excess_annual_return,
            'sharpe_ratio': excess_sharpe,
            'win_rate': excess_win_rate,
            'mean_daily': excess_returns.mean(),
            'std_daily': excess_returns.std()
        },
        'turnover': {
            'avg_daily_turnover': avg_turnover,
            'avg_annual_turnover': avg_annual_turnover
        }
    }

def plot_portfolio_comparison_with_split(cum_returns, portfolio_returns, cum_returns_no_cost, portfolio_returns_no_cost, 
                                        analysis_results, analysis_results_no_cost, split_date='2023-01-01'):
    """
    绘制投资组合对比图，包含样本内外分割线和超额收益率曲线
    同时显示考虑手续费和不考虑手续费的结果对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 将分割日期转换为datetime格式
    split_date = pd.to_datetime(split_date)
    
    # 1. 净值曲线对比 + 超额累计收益率 + 分割线
    ax1 = axes[0, 0]
    
    # 绘制净值曲线（考虑手续费）
    ax1.plot(cum_returns.index, cum_returns['top20_cum'], 
             label='Top 20 Portfolio (with transaction cost)', linewidth=2, alpha=0.8, color='blue')
    ax1.plot(cum_returns.index, cum_returns['all_cum'], 
             label='All Stocks Portfolio', linewidth=2, alpha=0.8, color='orange')
    
    # 绘制净值曲线（不考虑手续费，虚线）
    ax1.plot(cum_returns_no_cost.index, cum_returns_no_cost['top20_cum'], 
             label='Top 20 Portfolio (no transaction cost)', linewidth=2, alpha=0.5, color='blue', linestyle='--')
    
    # 添加样本内外分割线
    ax1.axvline(x=split_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='In/Out of Sample Split')
    
    # 添加区域标注
    ax1.axvspan(cum_returns.index[0], split_date, alpha=0.1, color='green', label='In Sample')
    ax1.axvspan(split_date, cum_returns.index[-1], alpha=0.1, color='orange', label='Out of Sample')
    
    # 添加文本标注
    ax1.text(0.02, 0.95, 'In Sample', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(0.85, 0.95, 'Out of Sample', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax1.set_title('Portfolio NAV Comparison (With vs Without Transaction Cost)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加收益率标签
    top20_return = analysis_results['top20']['total_return'] * 100
    all_return = analysis_results['all']['total_return'] * 100
    top20_return_no_cost = analysis_results_no_cost['top20']['total_return'] * 100
    ax1.text(0.02, 0.05, f'Top 20 (with cost): {top20_return:.1f}%\nTop 20 (no cost): {top20_return_no_cost:.1f}%\nAll Stocks: {all_return:.1f}%',
             transform=ax1.transAxes, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. 超额累计收益率曲线对比
    ax2 = axes[0, 1]
    
    # 计算超额累计收益率百分比
    excess_cum_pct = (cum_returns['excess_cum'] / cum_returns['all_cum'].iloc[0]) * 100
    excess_cum_pct_no_cost = (cum_returns_no_cost['excess_cum'] / cum_returns_no_cost['all_cum'].iloc[0]) * 100
    
    # 绘制超额累计收益率
    ax2.plot(cum_returns.index, excess_cum_pct, 
             label='Excess Return (with transaction cost)', 
             color='purple', linewidth=2, alpha=0.8)
    ax2.plot(cum_returns_no_cost.index, excess_cum_pct_no_cost, 
             label='Excess Return (no transaction cost)', 
             color='purple', linewidth=2, alpha=0.5, linestyle='--')
    
    # 添加分割线
    ax2.axvline(x=split_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加零线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # 添加区域标注
    ax2.axvspan(cum_returns.index[0], split_date, alpha=0.1, color='green')
    ax2.axvspan(split_date, cum_returns.index[-1], alpha=0.1, color='orange')
    
    # 计算最终超额收益率
    final_excess = excess_cum_pct.iloc[-1]
    final_excess_no_cost = excess_cum_pct_no_cost.iloc[-1]
    ax2.text(0.02, 0.95, f'Final Excess (with cost): {final_excess:.2f}%\nFinal Excess (no cost): {final_excess_no_cost:.2f}%', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax2.set_title('Excess Cumulative Return Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Excess Cumulative Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 日换手率曲线
    ax3 = axes[0, 2]
    
    if 'turnover_rate' in portfolio_returns.columns:
        # 绘制换手率曲线
        ax3.plot(portfolio_returns.index, portfolio_returns['turnover_rate'] * 100, 
                 label='Daily Turnover Rate', color='brown', linewidth=1.5, alpha=0.7)
        
        # 计算滚动平均换手率
        rolling_window = min(20, len(portfolio_returns))
        if rolling_window > 0:
            rolling_turnover = portfolio_returns['turnover_rate'].rolling(window=rolling_window).mean() * 100
            ax3.plot(rolling_turnover.index, rolling_turnover, 
                     label=f'{rolling_window}-day rolling avg', 
                     color='darkred', linewidth=2, alpha=0.8)
        
        # 添加分割线
        ax3.axvline(x=split_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 添加区域标注
        ax3.axvspan(cum_returns.index[0], split_date, alpha=0.1, color='green')
        ax3.axvspan(split_date, cum_returns.index[-1], alpha=0.1, color='orange')
        
        # 添加平均换手率线
        avg_turnover = portfolio_returns['turnover_rate'].mean() * 100
        ax3.axhline(y=avg_turnover, color='blue', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'Avg: {avg_turnover:.2f}%')
        
        ax3.set_title('Daily Turnover Rate of Top 20 Portfolio', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Turnover Rate (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 样本内外表现对比（考虑手续费）
    ax4 = axes[1, 0]
    
    # 分割样本内外数据
    split_date = pd.to_datetime(split_date)
    in_sample_mask = cum_returns.index < split_date
    out_sample_mask = cum_returns.index >= split_date
    
    # 计算样本内外的累计收益率
    if in_sample_mask.any():
        in_sample_top20_return = (cum_returns.loc[in_sample_mask, 'top20_cum'].iloc[-1] / 
                                  cum_returns.loc[in_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        in_sample_all_return = (cum_returns.loc[in_sample_mask, 'all_cum'].iloc[-1] / 
                                cum_returns.loc[in_sample_mask, 'all_cum'].iloc[0] - 1) * 100
    else:
        in_sample_top20_return = 0
        in_sample_all_return = 0
    
    if out_sample_mask.any():
        out_sample_top20_return = (cum_returns.loc[out_sample_mask, 'top20_cum'].iloc[-1] / 
                                   cum_returns.loc[out_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        out_sample_all_return = (cum_returns.loc[out_sample_mask, 'all_cum'].iloc[-1] / 
                                 cum_returns.loc[out_sample_mask, 'all_cum'].iloc[0] - 1) * 100
    else:
        out_sample_top20_return = 0
        out_sample_all_return = 0
    
    # 绘制柱状图
    x = np.arange(2)
    width = 0.35
    
    ax4.bar(x - width/2, [in_sample_top20_return, out_sample_top20_return], 
            width, label='Top 20 (with cost)', alpha=0.8)
    ax4.bar(x + width/2, [in_sample_all_return, out_sample_all_return], 
            width, label='All Stocks', alpha=0.8)
    
    ax4.set_xlabel('Sample Period')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.set_title('Portfolio Performance: In Sample vs Out of Sample (With Cost)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['In Sample', 'Out of Sample'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (v1, v2) in enumerate(zip([in_sample_top20_return, out_sample_top20_return], 
                                     [in_sample_all_return, out_sample_all_return])):
        ax4.text(i - width/2, v1, f'{v1:.1f}%', ha='center', va='bottom' if v1 > 0 else 'top', fontsize=9)
        ax4.text(i + width/2, v2, f'{v2:.1f}%', ha='center', va='bottom' if v2 > 0 else 'top', fontsize=9)
    
    # 5. 手续费影响分析
    ax5 = axes[1, 1]
    
    # 计算手续费对收益的影响
    cost_impact_pct = (cum_returns_no_cost['top20_cum'] - cum_returns['top20_cum']) / cum_returns_no_cost['top20_cum'] * 100
    
    # 绘制手续费影响曲线
    ax5.plot(cost_impact_pct.index, cost_impact_pct, 
             label='Transaction Cost Impact', color='red', linewidth=1.5, alpha=0.7)
    
    # 计算累计手续费影响
    cumulative_cost_impact = (cum_returns_no_cost['top20_cum'] - cum_returns['top20_cum']) / cum_returns['top20_cum'].iloc[0] * 100
    ax5.plot(cumulative_cost_impact.index, cumulative_cost_impact, 
             label='Cumulative Cost Impact', color='darkred', linewidth=2, alpha=0.8)
    
    # 添加分割线
    ax5.axvline(x=split_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加区域标注
    ax5.axvspan(cum_returns.index[0], split_date, alpha=0.1, color='green')
    ax5.axvspan(split_date, cum_returns.index[-1], alpha=0.1, color='orange')
    
    ax5.set_title('Impact of Transaction Costs on Top 20 Portfolio', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Cost Impact (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 表现指标对比
    ax6 = axes[1, 2]
    metrics = ['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max DD']
    with_cost_values = [
        analysis_results['top20']['annual_return'] * 100,
        analysis_results['top20']['annual_volatility'] * 100,
        analysis_results['top20']['sharpe_ratio'],
        analysis_results['top20']['max_drawdown'] * 100
    ]
    no_cost_values = [
        analysis_results_no_cost['top20']['annual_return'] * 100,
        analysis_results_no_cost['top20']['annual_volatility'] * 100,
        analysis_results_no_cost['top20']['sharpe_ratio'],
        analysis_results_no_cost['top20']['max_drawdown'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax6.bar(x - width/2, with_cost_values, width, label='With Transaction Cost', alpha=0.8, color='blue')
    ax6.bar(x + width/2, no_cost_values, width, label='No Transaction Cost', alpha=0.8, color='lightblue')
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Value')
    ax6.set_title('Top 20 Portfolio Performance: With vs Without Cost', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (v1, v2) in enumerate(zip(with_cost_values, no_cost_values)):
        ax6.text(i - width/2, v1, f'{v1:.1f}', ha='center', va='bottom' if v1 > 0 else 'top', fontsize=9)
        ax6.text(i + width/2, v2, f'{v2:.1f}', ha='center', va='bottom' if v2 > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_performance_summary_with_cost(analysis_results, analysis_results_no_cost, cum_returns, cum_returns_no_cost, split_date='2023-01-01'):
    """
    打印包含手续费影响的性能总结
    """
    split_date = pd.to_datetime(split_date)
    
    # 分割样本内外数据
    in_sample_mask = cum_returns.index < split_date
    out_sample_mask = cum_returns.index >= split_date
    
    # 计算手续费对总收益的影响
    total_cost_impact = (analysis_results_no_cost['top20']['total_return'] - analysis_results['top20']['total_return']) * 100
    annual_cost_impact = (analysis_results_no_cost['top20']['annual_return'] - analysis_results['top20']['annual_return']) * 100
    
    # 计算最终净值差异
    final_nav_with_cost = cum_returns['top20_cum'].iloc[-1]
    final_nav_no_cost = cum_returns_no_cost['top20_cum'].iloc[-1]
    nav_difference = final_nav_no_cost - final_nav_with_cost
    nav_difference_pct = (nav_difference / final_nav_with_cost) * 100
    
    print("=" * 120)
    print("PORTFOLIO PERFORMANCE SUMMARY (With Transaction Cost Analysis)")
    print("=" * 120)
    print(f"Split Date: {split_date.date()}")
    print(f"Transaction Cost Rate: 0.02% per side (0.04% round-trip)")
    print(f"Total Cost Impact on Final NAV: {nav_difference:.2f} ({nav_difference_pct:.2f}%)")
    print("-" * 120)
    print(f"{'Metric':<25} {'Top 20 (With Cost)':<20} {'Top 20 (No Cost)':<20} {'Difference':<20} {'All Stocks':<20}")
    print("-" * 120)
    
    # 全样本表现
    print(f"{'FULL SAMPLE:':<25}")
    print(f"{'  Total Return:':<25} {analysis_results['top20']['total_return']*100:>19.2f}% "
          f"{analysis_results_no_cost['top20']['total_return']*100:>20.2f}% "
          f"{total_cost_impact:>20.2f}% "
          f"{analysis_results['all']['total_return']*100:>20.2f}%")
    print(f"{'  Annual Return:':<25} {analysis_results['top20']['annual_return']*100:>19.2f}% "
          f"{analysis_results_no_cost['top20']['annual_return']*100:>20.2f}% "
          f"{annual_cost_impact:>20.2f}% "
          f"{analysis_results['all']['annual_return']*100:>20.2f}%")
    print(f"{'  Sharpe Ratio:':<25} {analysis_results['top20']['sharpe_ratio']:>20.2f} "
          f"{analysis_results_no_cost['top20']['sharpe_ratio']:>20.2f} "
          f"{analysis_results_no_cost['top20']['sharpe_ratio'] - analysis_results['top20']['sharpe_ratio']:>20.2f} "
          f"{analysis_results['all']['sharpe_ratio']:>20.2f}")
    print(f"{'  Win Rate:':<25} {analysis_results['top20']['win_rate']*100:>19.2f}% "
          f"{analysis_results_no_cost['top20']['win_rate']*100:>20.2f}% "
          f"{'':>20} "
          f"{analysis_results['all']['win_rate']*100:>20.2f}%")
    
    # 换手率信息
    if 'turnover' in analysis_results:
        print(f"{'  Avg Daily Turnover:':<25} {analysis_results['turnover']['avg_daily_turnover']*100:>19.2f}% "
              f"{'':>20} "
              f"{'':>20} "
              f"{'':>20}")
        print(f"{'  Avg Annual Turnover:':<25} {analysis_results['turnover']['avg_annual_turnover']*100:>19.2f}% "
              f"{'':>20} "
              f"{'':>20} "
              f"{'':>20}")
    
    # 样本内表现
    if in_sample_mask.any():
        in_sample_with_cost = (cum_returns.loc[in_sample_mask, 'top20_cum'].iloc[-1] / 
                               cum_returns.loc[in_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        in_sample_no_cost = (cum_returns_no_cost.loc[in_sample_mask, 'top20_cum'].iloc[-1] / 
                            cum_returns_no_cost.loc[in_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        in_sample_all = (cum_returns.loc[in_sample_mask, 'all_cum'].iloc[-1] / 
                         cum_returns.loc[in_sample_mask, 'all_cum'].iloc[0] - 1) * 100
        
        print(f"\n{'IN SAMPLE:':<25}")
        print(f"{'  Total Return:':<25} {in_sample_with_cost:>19.2f}% "
              f"{in_sample_no_cost:>20.2f}% "
              f"{in_sample_no_cost - in_sample_with_cost:>20.2f}% "
              f"{in_sample_all:>20.2f}%")
    
    # 样本外表现
    if out_sample_mask.any():
        out_sample_with_cost = (cum_returns.loc[out_sample_mask, 'top20_cum'].iloc[-1] / 
                                cum_returns.loc[out_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        out_sample_no_cost = (cum_returns_no_cost.loc[out_sample_mask, 'top20_cum'].iloc[-1] / 
                             cum_returns_no_cost.loc[out_sample_mask, 'top20_cum'].iloc[0] - 1) * 100
        out_sample_all = (cum_returns.loc[out_sample_mask, 'all_cum'].iloc[-1] / 
                          cum_returns.loc[out_sample_mask, 'all_cum'].iloc[0] - 1) * 100
        
        print(f"\n{'OUT OF SAMPLE:':<25}")
        print(f"{'  Total Return:':<25} {out_sample_with_cost:>19.2f}% "
              f"{out_sample_no_cost:>20.2f}% "
              f"{out_sample_no_cost - out_sample_with_cost:>20.2f}% "
              f"{out_sample_all:>20.2f}%")
    
    print("=" * 120)

# 主程序
if __name__ == "__main__":


    is_multifactor = True#是否通过多因子选股
    is_show_result = True#是否显示绩效
    is_save_result = True#是否保存结果

    #----------设置参数--------------------------------------

    # 设置分割日期区分样本内和样本外数据
    split_date = '2023-01-01'
    # 设置当前目录下的 'data' 文件夹作为数据路径
    data_path = os.path.join(os.getcwd(), "data")
    result_path = 'result'
    # 设置手续费率（单边万2）
    transaction_cost_rate = 0.0002  # 0.2%

    if is_multifactor:

        print('读取测评数据')
        #读取测评数据
        finial_raw_df = load_examination_data(data_path, 'reports_202511122033.csv',isPrice=False)
        assert '数据路径错误',finial_raw_df is not  None

        #计算视图
        finial_pivot_df = tran2pivot(finial_raw_df)


        print('选择因子并整理')
        #选择因子并整理
        selet_fact_df = select_factor(finial_pivot_df)
        selet_fact_df = collect_factor_data(selet_fact_df)

        # 读取价格数据---------------------------------------------------------------------------------------------------
        raw_price_df = load_examination_data(data_path, 'taiwan_stock_price_202511122027.csv',isPrice=True)
        assert '数据路径错误', raw_price_df is not None

        # 用股票价格计算因子
        sort_price_df = cal_factor_by_price(raw_price_df)
        # 计算波动因子
        sort_price_df = calculate_volatility_factor(sort_price_df, window=20)
        sort_price_df = sort_price_df.reset_index()

        # 整理量价数据方便合并---------------------------------------------------------------------------------
        sort_price_df['date'] = sort_price_df['date'].apply(lambda x: str(x)[:10])
        sort_price_df['date'] = sort_price_df['date'].astype('str')

        print('合并因子数据并整理')
        # 合并因子数据并整理--------------------------------------------------------------------------------------------------------------------
        merger_df = pd.merge(sort_price_df, selet_fact_df, on=['stock_id', 'date'], how='outer')
        # 对merge_df排序并且填充缺失值
        sort_merger_df = merger_df.sort_values(['stock_id', 'date'], ascending=[True, True])
        # 对部分缺失数值年份进行填充
        for fact_i in ['NCFF per', 'Total basic earnings per share']:
            sort_merger_df[fact_i] = sort_merger_df[['stock_id', fact_i]].groupby('stock_id').fillna(method='ffill')
        # 删除close没有数据的日期
        sort_merger_df = sort_merger_df.dropna(subset=['close'])

        print('计算收益率，对4个因子进行检查的因子检验')
        # 计算收益率，对4个因子进行检查的因子检验-----------------------------------------------------------------------------------------------------

        # 用来检测IC值
        sort_merger_df['returnO1O11'] = sort_merger_df.groupby('stock_id', group_keys=False).apply(
            lambda x: calculate_Return(x, 'open', 'open', 1, 11))
        # 用来计算收益率
        sort_merger_df['returnO1O2'] = sort_merger_df.groupby('stock_id', group_keys=False).apply(
            lambda x: calculate_Return(x, 'open', 'open', 1, 2))

        # 对4个因子进行IC检验
        testIC_fac_df = sort_merger_df[sort_merger_df.date < split_date].copy()

        # 检查单因子的相关性
        # 求每天因子与收益率相关系数矩阵
        fac_IC_df, fac_ICIR_df, Stats_df = get_fac_IC_df(testIC_fac_df, price_col='open',
                                                         fac_list=['trading_money_std_20d', 'ret_20d_std',
                                                                   'Total basic earnings per share', 'NCFF per'])
        #     print('转换前因子IC矩阵',fac_IC_df)
        print('因子统计结果')
        print(Stats_df)

        # 清洗因子数据，可以后面比较计算总分------------------------------------------------------------------------------------------------------
        # trading_money_std_20d IC值为负，转为正直
        positive_sort_merger_df = sort_merger_df.copy()
        positive_sort_merger_df['trading_money_std_20d'] = -1 * positive_sort_merger_df['trading_money_std_20d']


        # 处理因子
        processed_df = process_factors(positive_sort_merger_df, factors_to_process)
        # 计算因子总分
        processed_df['sum_alpha'] = processed_df[factors_to_process].sum(1)




        if is_show_result:

            # 检查数据
            print("数据基本信息:")
            print(f"数据形状: {processed_df.shape}")
            print(f"日期范围: {processed_df['date'].min()} 到 {processed_df['date'].max()}")
            print(f"股票数量: {processed_df['stock_id'].nunique()}")
            print(f"交易日数量: {processed_df['date'].nunique()}")

            # 检查收益率列
            print(f"\n收益率列统计:")
            print(f"returnO1O2 - 最小值: {processed_df['returnO1O2'].min()}")
            print(f"returnO1O2 - 最大值: {processed_df['returnO1O2'].max()}")
            print(f"returnO1O2 - 均值: {processed_df['returnO1O2'].mean():.6f}")

            # 检查缺失值和无穷值
            print(f"\n缺失值和无穷值检查:")
            print(f"returnO1O2 中 NaN 数量: {processed_df['returnO1O2'].isna().sum()}")
            inf_count = ((processed_df['returnO1O2'] == np.inf).sum() + (processed_df['returnO1O2'] == -np.inf).sum())
            print(f"returnO1O2 中 Inf/-Inf 数量: {inf_count}")
            print(f"sum_alpha 中 NaN 数量: {processed_df['sum_alpha'].isna().sum()}")



            # 计算投资组合收益率（考虑手续费）
            print(f"\n计算投资组合收益率（考虑手续费，单边{transaction_cost_rate*10000:.0f}bp）...")
            portfolio_returns = calculate_portfolio_returns_with_transaction_cost(
                processed_df, top_n=20, transaction_cost_rate=transaction_cost_rate
            )

            # 计算投资组合收益率（不考虑手续费，用于对比）
            print("计算投资组合收益率（不考虑手续费，用于对比）...")
            portfolio_returns_no_cost = calculate_portfolio_returns_no_transaction_cost(processed_df, top_n=20)

            # 计算累计收益率
            print("计算累计收益率...")
            cum_returns = calculate_cumulative_returns(portfolio_returns, initial_value=1.0)
            cum_returns_no_cost = calculate_cumulative_returns(portfolio_returns_no_cost, initial_value=1.0)

            # 分析投资组合表现
            print("分析投资组合表现...")
            analysis_results = analyze_portfolio_performance(cum_returns, portfolio_returns, initial_value=1.0)
            analysis_results_no_cost = analyze_portfolio_performance(cum_returns_no_cost, portfolio_returns_no_cost, initial_value=1.0)



            # 打印性能总结
            print_performance_summary_with_cost(analysis_results, analysis_results_no_cost, cum_returns, cum_returns_no_cost, split_date)

            # 打印收益率统计
            print("\n收益率统计:")
            print(f"Top 20组合平均日收益率（考虑手续费）: {portfolio_returns['top20_return'].mean()*100:.6f}%")
            print(f"Top 20组合平均日收益率（不考虑手续费）: {portfolio_returns_no_cost['top20_return'].mean()*100:.6f}%")
            print(f"全部股票组合平均日收益率: {portfolio_returns['all_return'].mean()*100:.6f}%")
            print(f"平均每日换手率: {portfolio_returns['turnover_rate'].mean()*100:.2f}%")
            print(f"平均年化换手率: {portfolio_returns['turnover_rate'].mean()*252 * 100:.2f}%")

            # 计算手续费对每日收益的影响
            daily_cost_impact = portfolio_returns_no_cost['top20_return'] - portfolio_returns['top20_return']
            print(f"手续费对每日收益的平均影响: {daily_cost_impact.mean()*100:.6f}%")
            print(f"手续费对累计收益的影响: {(analysis_results_no_cost['top20']['total_return'] - analysis_results['top20']['total_return'])*100:.2f}%")

            # 绘制对比图
            print(f"\n生成对比图表 (分割日期: {split_date})...")
            fig1 = plot_portfolio_comparison_with_split(
                cum_returns, portfolio_returns,
                cum_returns_no_cost, portfolio_returns_no_cost,
                analysis_results, analysis_results_no_cost,
                split_date
            )

            if is_save_result:

                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                # 保存数据
                portfolio_returns.to_csv(os.path.join(result_path,'portfolio_daily_returns_with_cost.csv'))
                cum_returns.to_csv(os.path.join(result_path,'portfolio_cumulative_returns_with_cost.csv'))
                portfolio_returns_no_cost.to_csv(os.path.join(result_path,'portfolio_daily_returns_no_cost.csv'))
                cum_returns_no_cost.to_csv(os.path.join(result_path,'portfolio_cumulative_returns_no_cost.csv'))

                # 保存样本内外数据
                split_date_dt = pd.to_datetime(split_date)
                in_sample_mask = cum_returns.index < split_date_dt
                out_sample_mask = cum_returns.index >= split_date_dt

                cum_returns[in_sample_mask].to_csv(os.path.join(result_path,'portfolio_cumulative_returns_in_sample_with_cost.csv'))
                cum_returns[out_sample_mask].to_csv(os.path.join(result_path,'portfolio_cumulative_returns_out_sample_with_cost.csv'))

                print("\n数据已保存到文件:")
                print("1. portfolio_daily_returns_with_cost.csv - 日收益率数据（考虑手续费）")
                print("2. portfolio_cumulative_returns_with_cost.csv - 累计收益率数据（考虑手续费）")
                print("3. portfolio_daily_returns_no_cost.csv - 日收益率数据（不考虑手续费）")
                print("4. portfolio_cumulative_returns_no_cost.csv - 累计收益率数据（不考虑手续费）")
                print("5. portfolio_cumulative_returns_in_sample_with_cost.csv - 样本内累计收益率数据（考虑手续费）")
                print("6. portfolio_cumulative_returns_out_sample_with_cost.csv - 样本外累计收益率数据（考虑手续费）")

