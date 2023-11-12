import os
import logging
import argparse
import numpy as np
import pandas as pd
import talib.abstract as ta
logger = logging.getLogger('trade')


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