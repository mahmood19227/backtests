import os
import logging
import argparse
import numpy as np
import pandas as pd
import talib.abstract as ta
from hedgeMode import HedgeMode
logger = logging.getLogger('trade')


class SmartHedge2(HedgeMode):
    def __init__(self,vars):
        self.vars = vars
        # self.indicators = ['rsi336','rsi672','rsi168','rsi84','rsi42','ma50','ma100','ma200','ma500','ma1000','ma1500','ma2000','ma2500','ma10000','ma20000','range1','range2']
        self.n_fails = 0

    def is_trend(self,k):
        for i in range(1,k+1):
            if self.positionHistory[-i]['range1'] or self.positionHistory[-i]['range2']:
                return False
        print(f"Trend at position no: {len(self.positionHistory)}")
        return True

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


        # if k>self.vars['activation_k'] and price < self.prices.at[index,'ma50'] <  self.prices.at[index,'ma200'] and self.prices.at[index,'rsi84']<43:
        #     self.vars['kpos'] = kpos = 1
        # elif k>self.vars['activation_k'] and price > self.prices.at[index,'ma50'] >  self.prices.at[index,'ma200'] and self.prices.at[index,'rsi84']>57:
        #     self.vars['kpos'] = kpos = 0
        # else:
        #     self.vars['temp_kpos'] = -1
        # if len(self.positionHistory)==364:
        #     print("danger")
        if k>self.vars['activation_k'] and 50<=self.prices.at[index,'rsi84']<=self.vars['rsi_up'] and self.is_trend(k):
            self.vars['kpos'] = kpos = 0
        if k>self.vars['activation_k'] and self.vars['rsi_down']<=self.prices.at[index,'rsi84']<=50 and self.is_trend(k):
            self.vars['kpos'] = kpos = 1

        # if k>self.vars['activation_k'] and self.vars['temp_kpos']==-1 and price < self.prices.at[index,'ma50'] <  self.prices.at[index,'ma200'] and self.prices.at[index,'rsi84']<43 and self.prices.at[index,'rsi168']<45:
        #     self.vars['temp_kpos'] = kpos = 1
        #     # print(f"temp kpos[{self.prices.at[index,'date']}] = {kpos}")
        # elif k>self.vars['activation_k'] and self.vars['temp_kpos']==-1 and price > self.prices.at[index,'ma50'] >  self.prices.at[index,'ma200'] and self.prices.at[index,'rsi84']>57 and self.prices.at[index,'rsi168']>55:
        #     self.vars['temp_kpos'] = kpos = 0
        #     # print(f"temp kpos[{self.prices.at[index,'date']}] = {kpos}")
        # else:
        #     self.vars['temp_kpos'] = -1


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

    # def Profit(self,tp_side):
    #     if self.vars.get('temp_kpos',-1)!=-1:
    #         kpos = self.vars['temp_kpos']
    #     else:
    #         kpos = self.vars.get('kpos',-1)
    #     market_type = self.vars.get("market_type","range")
    #     if kpos==-1:
    #         kpos=tp_side
    #         self.vars['kpos']==kpos
    #         self.vars['k'] +=1
    #         return
    #     if tp_side==kpos:
    #         self.vars['k'] = 1
    #         if market_type == "range":
    #             self.vars['kpos']=1-kpos
    #     else:
    #         self.vars['k'] +=1
    #         if market_type=="trend":
    #             self.vars['kpos'] = tp_side

    def back_test(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.prices = dataframe
        self.indicators = list(dataframe.columns)
        hasPositions = False
        self.positionHistory = []
        self.n_fails = 0
        for i,r in dataframe.iterrows():
            if hasPositions:
                if  r['low'] < tp < r['high']:
                    position['exit_price'] = tp
                    position['long_result'] = 'WIN'
                    position['short_result'] = 'LOSS'
                    self.Profit(0)
                    hasPositions = False
                elif r['low'] < sl < r['high']:
                    # if len(self.positionHistory)==1712:
                    #     print("danger")
                    self.Profit(1)
                    position['exit_price'] = sl
                    position['long_result'] = 'LOSS'
                    position['short_result'] = 'WIN'
                    hasPositions = False
                if not hasPositions:
                    position['exit_time'] =r['date']
                    position['close'] =r['close']
                    for indicator in self.indicators:
                        position[indicator] = self.prices.at[i,indicator]
                    self.positionHistory.append(position)
            else:
                orders = self.getNewOrders("dummy",r['open'],i-1)
                tp = orders[0]['tp']
                sl = orders[0]['sl']
                hasPositions = True
                position = {
                       'entry_price':r['open'],
                       'long_volume':orders[0]['quantity'],
                       'short_volume':orders[1]['quantity'],
                       'entry_time':r['date'],
                       'long_k':self.vars['k'] if orders[0]['quantity']>orders[1]['quantity'] else 0,
                       'short_k':self.vars['k'] if orders[1]['quantity']>orders[0]['quantity'] else 0,
                       'range1':self.prices.at[max(0,i-1),'range1'],
                       'range2':self.prices.at[max(0,i-1),'range2']
                       }
                
        self.positionHistory = pd.DataFrame(self.positionHistory)
        self.positionHistory['long_PNL'] = (self.positionHistory['exit_price']-self.positionHistory['entry_price'])*self.positionHistory['long_volume']
        self.positionHistory['short_PNL'] = (self.positionHistory['entry_price']-self.positionHistory['exit_price'])*self.positionHistory['short_volume']

        return self.positionHistory

if __name__=='__main__':
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
    )
    parser = argparse.ArgumentParser(description='Back Tessting of strategy.')
    parser.add_argument('-pair',type=str,required=False,help='Pair that is used for backstep.',default='BTC/USDT:USDT')
    parser.add_argument('-timerange',type=str,required=False,help='Time range that backsteps run on it.',default='20200101-')
    parser.add_argument('-data_path',type=str,required=False,help='Path which contains OHLCV data.',default='~/freqtrade/user_data/data/binance/futures')
    parser.add_argument('-data_file',type=str,required=False,help='File which contains OHLCV data.',default=None)
    parser.add_argument('-output',type=str,required=False,help='Output csv file which results will be saved in it.',default='smart2.csv')
    parser.add_argument('-timeframe',required=False,type=str,help='Time frame that is used for backtesting.',default='5m')
    parser.add_argument('-init_money',type=int,required=False,help='Initial Money in the wallet.',default=10000)
    parser.add_argument('-tp',type=float,required=False,help='Initial Money in the wallet.',default=0.014)
    parser.add_argument('-market_type',type=str,required=False,help='Initial Money in the wallet.',default='range')
    parser.add_argument('-activation_k',type=int,required=False,help='K value that activate technical indicators watching.',default=3)

    parser.add_argument('-ma1',type=int,required=False,help='Initial Money in the wallet.',default=2500)
    parser.add_argument('-ma2',type=int,required=False,help='Initial Money in the wallet.',default=10000)
    parser.add_argument('-ma3',type=int,required=False,help='Initial Money in the wallet.',default=20000)
    parser.add_argument('-rsi_up',type=int,required=False,help='Initial Money in the wallet.',default=65)
    parser.add_argument('-rsi_down',type=int,required=False,help='Initial Money in the wallet.',default=35)

    args = parser.parse_args()

    if args.data_file is not None:
        df = pd.read_feather(args.data_file)
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

    df['rsi672'] = ta.RSI(df,timeperiod=672)
    df['rsi336'] = ta.RSI(df,timeperiod=336)
    df['rsi168'] = ta.RSI(df,timeperiod=168)
    df['rsi84'] = ta.RSI(df,timeperiod=84)
    df['rsi42'] = ta.RSI(df,timeperiod=42)
    df['ma50'] = ta.SMA(df,timeperiod=50)
    df['ma100'] = ta.SMA(df,timeperiod=100)
    df['ma200'] = ta.SMA(df,timeperiod=200)
    df['ma500'] = ta.SMA(df,timeperiod=500)
    df['ma1000'] = ta.SMA(df,timeperiod=1000)
    df['ma1500'] = ta.SMA(df,timeperiod=1500)
    df['ma2000'] = ta.SMA(df,timeperiod=2000)
    df['ma2500'] = ta.SMA(df,timeperiod=2500)
    df['ma10000'] = ta.SMA(df,timeperiod=10000)
    df['ma20000'] = ta.SMA(df,timeperiod=20000)
    df['BBANDS_U'],df['BBANDS_M'],df['BBANDS_L'] = ta.BBANDS(df, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0)    
    df['range1'] = (df['close'] > df[f'ma{args.ma1}']) & (df['close'] < df[f'ma{args.ma2}']) | (df['close'] < df[f'ma{args.ma1}']) & (df['close'] > df[f'ma{args.ma2}'])
    df['range2'] = (df['close'] > df[f'ma{args.ma1}']) & (df['close'] < df[f'ma{args.ma3}']) | (df['close'] < df[f'ma{args.ma1}']) & (df['close'] > df[f'ma{args.ma3}'])
    # df['range1'] = df['range1'].shift()
    # df['range2'] = df['range2'].shift()
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
        'rsi_up':args.rsi_up,
        'rsi_down':args.rsi_down,
        'activation_k':args.activation_k
    }
    print(f"vars={vars}")
    strategy = SmartHedge2(vars)
    result_df = strategy.back_test(df)
    result_df.to_csv(f'{args.output}',index=False)
    print(f"Number of trades: {len(result_df)}")

    k_dist = result_df['long_k'].value_counts()
    k_dist.at[len(k_dist)]=0
    k_dist = k_dist.diff(-1).dropna()[1:].astype(int)
    print(f"Long K distributions: {k_dist}")

    k_dist = result_df['short_k'].value_counts()
    k_dist.at[len(k_dist)]=0
    k_dist = k_dist.diff(-1).dropna()[1:].astype(int)
    print(f"Short K distributions: {k_dist}")