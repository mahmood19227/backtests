# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
import logging
logger = logging.getLogger('trade')

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class HedgeMode(IStrategy):
    """
    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }
    timeframe='1h'

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 1

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    # order_types = {
    #     'entry': 'market',
    #     'exit': 'market',
    #     'stoploss': 'market',
    #     'stoploss_on_exchange': False
    # }

    # entry_pricing = {
    #     'price_side': 'other'
    # }

    # exit_pricing = {
    #     'price_side': 'other'
    # }
    # Optional order time in force.
    # order_time_in_force = {
    #     'entry': 'GTC',
    #     'exit': 'GTC'
    # }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def getNewOrders(self,pair, price, leverage, vars):
        safety_factor = vars.get('safety_factor')
        if safety_factor is None:
            safety_factor = 1.05
        kpos = int(vars.get('kpos', -1))
        k = int(vars.get('k', 0))
        
        start_money = float(vars.get('start_money', 1))
        increase_factor = float(vars.get('increase_factor', 1))
        normal_fail = int(vars.get('normal_fail', 4))
        max_fail = int(vars.get('max_fail', 8))
        
        tp = float(vars.get('tp', 1.018))
        sl = float(vars.get('sl', 0.019))
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

    def Profit(self,vars,tp_side):
        sides = ['BUY','SELL']
        kpos = vars.get('kpos',-1)
        market_type = vars.get("market_type","range")
        if kpos==-1:
            kpos=tp_side
            vars['kpos']==kpos
            vars['k'] +=1
            return
        print(f"Profit for position {sides[tp_side]} , kpos={kpos}, tp_side={tp_side}")
        if tp_side==kpos:
            vars['k'] = 1
            if market_type == "range":
                vars['kpos']=1-kpos
        else:
            vars['k'] +=1
            if market_type=="trend":
                vars['kpos'] = tp_side

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        vars = {
            'k':0,
            'safety_factor':1.2,
            'kpos':0,
            'start_money':1,
            'increase_factor':1,
            'normal_fail':4,
            'max_fail':9,
            'tp':1.005,
            'sl':0.005,
            'leverage':100
            }

        # dataframe['enter_long']=0
        # dataframe['enter_short']=0
        # dataframe['exit_long']=0
        # dataframe['exit_short']=0
        print(dataframe.head(10))

        pre = None
        hasPositions = False

        for i,r in dataframe.iterrows():
            if hasPositions:
                if  r['low'] < tp < r['high']:
                    print(f"{r['date']} : Position Long reached to TP (k={vars['k']})")
                    self.Profit(vars,0)
                    dataframe.at[i,'exit_long'] = 1
                    # dataframe.at[i,'exit_short'] = 1
                    hasPositions = False
                elif r['low'] < sl < r['high']:
                    print(f"{r['date']} : Position Short reached to TP (k={vars['k']})")
                    self.Profit(vars,1)
                    dataframe.at[i,'exit_long'] = 1
                    # dataframe.at[i,'exit_short'] = 1
                    hasPositions = False
                    # dataframe.at[i+1,'has_position'] = True
            else:
                orders = self.getNewOrders("dummy",r['close'],vars['leverage'],vars)
                dataframe.at[i,'vol_long'] = orders[0]['quantity']
                dataframe.at[i,'vol_short'] = orders[1]['quantity']
                tp = orders[0]['tp']
                sl = orders[0]['sl']
                dataframe.at[i,'enter_long'] = 1
                # dataframe.at[i,'enter_short'] = 1
                hasPositions = True
        #         if i+1<len(dataframe):
        #             dataframe.at[i+1,'has_positions'] = True
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe[(dataframe['enter_long']==1) | (dataframe['exit_long']==1)]
        print(df[['date','enter_long','exit_long','vol_long','vol_short']].head(10))
        return dataframe
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
        #         (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
        #         (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
            # 'exit_long'] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)        
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
        #         (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
        #         (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'exit_short'] = 1
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.loc[-1].squeeze()

        if side == 'long':
            print(f"long_volume={current_candle['vol_long']}")
            return current_candle['vol_long']
        else:
            print(f"short_volume={current_candle['vol_short']}")
            return current_candle['vol_short']

        # if current_candle['fastk_rsi_1h'] > current_candle['fastd_rsi_1h']:
        #     if self.config['stake_amount'] == 'unlimited':
        #         # Use entire available wallet during favorable conditions when in compounding mode.
        #         return max_stake
        #     else:
        #         # Compound profits during favorable conditions instead of using a static stake.
        #         return self.wallets.get_total_stake_amount() / self.config['max_open_trades']

        # # Use default stake amount.
        # return proposed_stake
