import os
import logging
import argparse
import numpy as np
import pandas as pd
import talib.abstract as ta
logger = logging.getLogger('trade')


class HedgeMode():
    timeframe = '5m'
    n_fails = 0
    def __init__(self,vars):
        self.vars = vars

    def getNewOrders(self,pair, price,index):
        leverage = self.vars['leverage']
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
            # logger.error(f"Reached to max_fail. reseting k to 1.")
            k = self.vars['k'] = 1
            self.n_fails +=1
        tp = float(self.vars.get('tp', 1.018))
        sl = float(self.vars.get('sl', 0.019))
        tp = max(0.001, tp - 1)

        reward_risk = tp / sl
        modified_start_trade_money = start_money / (tp * leverage)
        
        quantities = (np.zeros(max_fail + 1) + 1) * modified_start_trade_money
        
        if type(safety_factor).__name__=='float':
            for i in range(1, max_fail + 1):
                quantities[i] = ((sum(quantities[0 : i]) / ( reward_risk )) + modified_start_trade_money)*safety_factor
        else:
            for i in range(1, max_fail + 1):
                quantities[i] = ((sum(quantities[0 : i]) / ( reward_risk )) + modified_start_trade_money)*safety_factor[i-1]
        # print(f"quantities = {quantities}")
        qs = [0,0]
        logger.debug(f"kpos for new orders ={kpos}")
        if kpos ==-1:   #first time
            qs[0] = quantities[k] / price * leverage
            qs[1] = quantities[k] / price * leverage
        else:
            qs[kpos] = quantities[k] / price * leverage
            qs[1-kpos] = quantities[0] / price * leverage

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
        kpos = self.vars.get('kpos',-1)
        market_type = self.vars.get("market_type","range")
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

    def back_test(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.prices = dataframe
        self.indicators = list(dataframe.columns)

        hasPositions = False
        self.positionHistory = []
        self.n_fails = 0
        for i,r in dataframe.iterrows():
            if not hasPositions:
                orders = self.getNewOrders("dummy",r['open'],i)
                tp = orders[0]['tp']
                sl = orders[0]['sl']
                hasPositions = True
                position = {
                       'entry_price':r['open'],
                       'long_volume':orders[0]['quantity'],
                       'short_volume':orders[1]['quantity'],
                       'entry_time':r['date'],
                       'long_k':self.vars['k'] if orders[0]['quantity']>orders[1]['quantity'] else 0,
                       'short_k':self.vars['k'] if orders[1]['quantity']>orders[0]['quantity'] else 0
                       }
            if hasPositions:
                if tp < r['high']:
                    position['exit_price'] = tp
                    position['long_result'] = 'WIN'
                    position['short_result'] = 'LOSS'
                    self.Profit(0,i+1)
                    hasPositions = False
                elif r['low'] < sl:
                    # if len(self.positionHistory)==1712:
                    #     print("danger")
                    self.Profit(1,i+1)
                    position['exit_price'] = sl
                    position['long_result'] = 'LOSS'
                    position['short_result'] = 'WIN'
                    hasPositions = False
                if not hasPositions:
                    position['exit_time'] =r['date']
                    # position['close'] =r['close']
                    for indicator in self.indicators:
                        position[indicator] = self.prices.at[i,indicator]
                    self.positionHistory.append(position)
        
        self.positionHistory = pd.DataFrame(self.positionHistory)
        self.positionHistory['long_PNL'] = (self.positionHistory['exit_price']-self.positionHistory['entry_price'])*self.positionHistory['long_volume']
        self.positionHistory['short_PNL'] = (self.positionHistory['entry_price']-self.positionHistory['exit_price'])*self.positionHistory['short_volume']

        return self.positionHistory

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Back Tessting of strategy.')
    parser.add_argument('-pair',type=str,required=False,help='Pair that is used for backstep.',default='BTC/USDT:USDT')
    parser.add_argument('-timerange',type=str,required=False,help='Time range that backsteps run on it.',default='20200101-')
    parser.add_argument('-data_path',type=str,required=False,help='Path which contains OHLCV data.',default='~/freqtrade/user_data/data/binance/futures')
    parser.add_argument('-data_file',type=str,required=False,help='File which contains OHLCV data.',default=None)
    parser.add_argument('-output',type=str,required=False,help='Output csv file which results will be saved in it.',default='output.csv')
    parser.add_argument('-timeframe',required=False,type=str,help='Time frame that is used for backtesting.',default='5m')
    parser.add_argument('-init_money',type=int,required=False,help='Initial Money in the wallet.',default=10000)
    parser.add_argument('-tp',type=float,required=False,help='Initial Money in the wallet.',default=0.017)
    parser.add_argument('-market_type',type=str,required=False,help='Initial Money in the wallet.',default='range')

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
        print(f"--timerange must have format of YYYYMMDD-[YYYYMMDD].")
        exit()
    start,end = args.timerange.split('-')
    
    start = pd.Timestamp(year=int(start[:4]),month=int(start[4:6]),day=int(start[6:]),tz='UTC')
    if end=='':
        end = df['date'].max()
    else:
        end = pd.Timestamp(year=int(end[:4]),month=int(end[4:6]),day=int(end[6:]))

    df = df[(df['date']>=start) & (df['date']<=end)]
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
    strategy = HedgeMode(vars)
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