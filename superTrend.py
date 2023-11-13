import os
import logging
import argparse
import numpy as np
import pandas as pd
import talib.abstract as ta
import pandas_ta as pta
from hedgeMode import HedgeMode
logger = logging.getLogger('trade')


class SuperTrend(HedgeMode):
    def __init__(self,dataframe,vars):
        super(SuperTrend, self).__init__(dataframe,vars)
        self.indicators += ['sti_direction_30min','sti_direction_1h','sti_direction_2h','is_trend']#'rsi336','rsi672','rsi168','rsi84','rsi42','ma50','ma100','ma200','ma500','ma1000','ma1500','ma2000','ma2500','ma10000','ma20000']

    n_fails = 0
    def getNewOrders(self,pair, price, index):
        safety_factor = self.vars.get('safety_factor')
        if safety_factor is None:
            safety_factor = 1.05
        kpos = int(self.vars.get('kpos', -1))
        k = int(self.vars.get('k', 0))
        
        start_money = float(self.vars.get('start_money', 1))
        increase_factor = float(self.vars.get('increase_factor', 1))
        normal_fail = int(self.vars.get('normal_fail', 4))
        max_fail = int(self.vars.get('max_fail', 8))
        if k>self.vars['max_fail']:
            k = self.vars['k'] = 1
            self.n_fails +=1
        tp = float(self.vars.get('tp', 1.018))
        sl = float(self.vars.get('sl', 0.019))
        tp = max(0.001, tp - 1)

        reward_risk = tp / sl
        modified_start_trade_money = start_money / (tp * self.vars['leverage'])
        
        quantities = (np.zeros(max_fail + 1) + 1) * modified_start_trade_money
        
        if type(safety_factor).__name__=='float':
            for i in range(1, max_fail + 1):
                quantities[i] = ((sum(quantities[0 : i]) / ( reward_risk )) + modified_start_trade_money)*safety_factor
        else:
            for i in range(1, max_fail + 1):
                quantities[i] = ((sum(quantities[0 : i]) / ( reward_risk )) + modified_start_trade_money)*safety_factor[i-1]
        qs = [0,0]
        if kpos ==-1:
            qs[0] = quantities[k] / price * self.vars['leverage']
            qs[1] = quantities[k] / price * self.vars['leverage']
        else:
            qs[kpos] = quantities[k] / price * self.vars['leverage']
            qs[1-kpos] = quantities[0] / price * self.vars['leverage']

        if(normal_fail > 0 and increase_factor > 0 and k<normal_fail):
            qs = [q*increase_factor for q in qs]

        tp = 1+tp
        sl = 1-sl
        tp = price * tp
        sl = price * sl

        order1 = {'pair' : pair, 'entry_price' : price, 'side' : 'BUY', 'quantity' : qs[0], 'sl' : sl, 'tp' : tp, 'positionSide':'LONG', 'k':k}
        order2 = {'pair' : pair, 'entry_price' : price, 'side' : 'SELL', 'quantity' : qs[1], 'sl' : tp, 'tp' : sl, 'positionSide':'SHORT', 'k':k}
        return [order1,order2]

    def Profit(self,tp_side,index):
        if self.vars.get('temp_kpos',-1)!=-1:
            kpos = self.vars['temp_kpos']
        else:
            kpos = self.vars.get('kpos',-1)
        market_type = 'trend' if self.prices.at[index,'is_trend'] else 'range'
        if kpos==-1:
            kpos=tp_side
            self.vars['kpos']==kpos
            self.vars['k'] +=1
            return
        if tp_side==kpos:
            self.vars['k'] = 1
            if market_type == "range":
                self.vars['kpos']=1-kpos
        else:
            self.vars['k'] +=1
            if market_type=="trend":
                self.vars['kpos'] = tp_side


    def populate_indicators(self):
        self.prices['date'] = pd.to_datetime(self.prices['date'])
        self.prices.set_index('date',inplace=True)
        df30min = self.prices.resample('30min').agg({
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last'
        })
        df1h = self.prices.resample('1h').agg({
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last'
        })
        df2h = self.prices.resample('2h').agg({
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last'
        })
        sti = pta.supertrend(df30min['high'], df30min['low'], df30min['close'], length=10, multiplier=3)
        df30min['sti_direction_30min'] = sti[sti.columns[1]]
        df30min = df30min['sti_direction_30min'].shift(periods=1)
        sti = pta.supertrend(df1h['high'], df1h['low'], df1h['close'], length=10, multiplier=3)
        df1h['sti_direction_1h'] = sti[sti.columns[1]]
        df1h = df1h['sti_direction_1h'].shift(periods=1)
        sti = pta.supertrend(df2h['high'], df2h['low'], df2h['close'], length=10, multiplier=3)
        df2h['sti_direction_2h'] = sti[sti.columns[1]]
        df2h = df2h['sti_direction_2h'].shift(periods=1)
        self.prices = self.prices.merge(df30min,left_index=True,right_index=True,how='left').merge(df1h,left_index=True,right_index=True,how='left').merge(df2h,left_index=True,right_index=True,how='left').ffill()
        self.prices['is_trend'] = (self.prices['sti_direction_30min']==self.prices['sti_direction_1h']) & (self.prices['sti_direction_30min']==self.prices['sti_direction_2h'])
        self.prices = self.prices.reset_index()
        # print(f"columns = {self.prices.columns}")

    
if __name__=='__main__':
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(description='Back Tessting of strategy.')
    parser.add_argument('-pair',type=str,required=False,help='Pair that is used for backstep.',default='BTC/USDT:USDT')
    parser.add_argument('-timerange',type=str,required=False,help='Time range that backsteps run on it.',default='-')
    parser.add_argument('-data_path',type=str,required=False,help='Path which contains OHLCV data.',default='data')
    parser.add_argument('-data_file',type=str,required=False,help='File which contains OHLCV data.',default='data/Binance_BNBUSDT_5Min.csv')
    parser.add_argument('-output',type=str,required=False,help='Output csv file which results will be saved in it.',default='supertrend.csv')
    parser.add_argument('-timeframe',required=False,type=str,help='Time frame that is used for backtesting.',default='5m')
    parser.add_argument('-init_money',type=int,required=False,help='Initial Money in the wallet.',default=10000)
    parser.add_argument('-tp',type=float,required=False,help='Initial Money in the wallet.',default=0.04)
    parser.add_argument('-market_type',type=str,required=False,help='Initial Money in the wallet.',default='range')
    parser.add_argument('-activation_k',type=int,required=False,help='K value that activate technical indicators watching.',default=3)

    args = parser.parse_args()

    if args.data_file is not None:
        df = pd.read_csv(args.data_file)
    else:
        if len(args.pair) != 13 or args.pair[3]!='/' or args.pair[8]!=':':
            print("Pair name must have format of pair/base:settle.")
            exit()
        data_file = os.path.join(args.data_path,f"{args.pair.replace('/','_').replace(':','_')}-{args.timeframe}-futures.feather")
        df = pd.read_feather(data_file)
    
    if not '-' in args.timerange:
        print(f"--timerange must have format of [YYYYMMDD]-[YYYYMMDD].")
        exit()
    start,end = args.timerange.split('-')
    
    if start=='':
        start = df['date'].min()
    else:
        start = pd.Timestamp(year=int(start[:4]),month=int(start[4:6]),day=int(start[6:]),tz='UTC')
    if end=='':
        end = df['date'].max()
    else:
        end = pd.Timestamp(year=int(end[:4]),month=int(end[4:6]),day=int(end[6:]),tz='UTC')


    df = df[(df['date']>=start) & (df['date']<=end)]
    # sti = pta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
    # df['sti_direction'] = sti[sti.columns[1]]

    vars = {
        'k':0,
        'safety_factor':1.2,
        'kpos':0,
        'start_money':1,
        'increase_factor':1,
        'normal_fail':4,
        'max_fail':20,
        'tp':1+args.tp,
        'sl':args.tp,
        'leverage':100,
        'market_type':args.market_type,
    }
    print(f"vars={vars}")
    strategy = SuperTrend(df,vars)
    # df = strategy.populate_indicators(df)
    result_df = strategy.back_test()
    result_df.to_csv(f'{args.output}',index=False)
    print(f"Number of trades: {len(result_df)}")
    k_dist = result_df['long_k'].value_counts()
    # k_dist.at[len(k_dist)]=0
    # k_dist = k_dist.diff(-1).dropna()[1:].astype(int)
    print(f"Long K distributions: {k_dist}")
    max_k = max(k_dist.keys())
    k_dist = result_df['short_k'].value_counts()
    # k_dist.at[len(k_dist)]=0
    # k_dist = k_dist.diff(-1).dropna()[1:].astype(int)
    print(f"Short K distributions: {k_dist}")
    max_k = max(max(k_dist.keys()),max_k)
    print(f"Max(k) = {max_k}")

    