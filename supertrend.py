import os
import ccxt
import config
import schedule
import numpy as np
import pandas as pd
import ta
import warnings
import pprint
import json
import time
from datetime import datetime, date

pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
bot_start = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')

# Global variables
symbol = 'ETH/USDT'
timeframe = '1m'
limit = 1000
in_position = False
usd_stake = 200
quantity_buy_sell = 0
cost, profit, fees = 0, 0, 0
backtest = True

# CCXT config
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': config.BINANCE_API_KEY,
    'secret': config.BINANCE_SECRET_KEY,
    'timeout': 50000,
    'enableRateLimit': True,
})


def log_write(msg, bot_start=bot_start, df=False, json=False, print_it=True):
    timestamp = datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H:%M:%S.%f')
    path = os.path.dirname(os.path.realpath(__file__))
    log = open('{}/log/supertrend-{}.log'.format(path, bot_start), 'a')

    if not df and not json:
        entry = f'{timestamp}: {msg}\n'
    elif df:
        entry = f'{timestamp}\n{msg}\n'
    elif json:
        entry = '{}\n{}\n'.format(st, pprint.pprint(json.loads(msg)))

    log.write(entry)

    if print_it:
        print(entry.strip())


def ema(df, period=20):
    ema = ta.trend.ema_indicator(df.close, window=period)
    return ema


def macd(df):
    macd = ta.trend.macd_diff(df.close)
    return macd


def rsi(df, period=14):
    rsi = ta.momentum.rsi(df.close, window=period)
    return rsi


def stoch_rsi(df, period=14, smooth=3):
    stoch_rsi = ta.momentum.StochRSIIndicator(df.close, window=period, smooth1=smooth, smooth2=smooth)
    return stoch_rsi


def cmf(df, period=20):
    cmf = ta.volume.chaikin_money_flow(df.high, df.low, df.close, df.volume, window=period)
    return cmf
   

def adx(df, period=14):
    adx = ta.trend.adx(df.high, df.low, df.close, window=period)
    return adx


def psar(df, step=0.02, max_step=0.2):
    psar = ta.trend.PSARIndicator(df.high, df.low, df.close, step=step, max_step=max_step)
    return psar.psar()


def tr(df):
    df['high_low'] = abs(df.high - df.low)
    df['high_close'] = abs(df.high - df.close.shift())
    df['low_close'] = abs(df.low - df.close.shift())
    tr = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    return tr


def atr(df, period):
    df['tr'] = tr(df)
    atr = df.tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return atr


def supertrend(df, period=7, atr_multiplier=3):
    hl2 = (df.high + df.low) / 2
    df['atr'] = atr(df, period)
    df['upperband'] = hl2 + (atr_multiplier * df.atr)
    df['lowerband'] = hl2 - (atr_multiplier * df.atr)
    df['in_uptrend'] = True

    for index, row in df.iterrows():
        if index == 0:
            continue

        if df.close[index] > df.upperband[index-1]:
            df.in_uptrend[index] = True
        elif df.close[index] < df.lowerband[index-1]:
            df.in_uptrend[index] = False
        else:
            df.in_uptrend[index] = df.in_uptrend[index-1]

            if df.in_uptrend[index] and (df.lowerband[index] < df.lowerband[index-1]):
              df.lowerband[index] = df.lowerband[index-1]
            if not df.in_uptrend[index] and (df.upperband[index] > df.upperband[index-1]):
              df.upperband[index] = df.upperband[index-1]

    return df


def add_ichimoku(df, conversion=9, baseline=26, leadingspan=52, displacement=26):
    # Tenkan-sen (Conversion Line)
    nine_period_high = df.high.rolling(window=conversion).max()
    nine_period_low = df.low.rolling(window=conversion).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = df.high.rolling(window=baseline).max()
    period26_low = df.low.rolling(window=baseline).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    df['senkou_span_a'] = ((df.tenkan_sen + df.kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = df.high.rolling(window=leadingspan).max()
    period52_low = df.low.rolling(window=leadingspan).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(displacement)

    # Chikou Span (Lagging Span): The most current closing price plotted 26 time periods behind (optional)
    df['chikou_span'] = df.close.shift(-displacement)


def check_buy_sell_signals(df, ticker):
    global in_position, symbol, backtest, quantity_buy_sell, usd_stake, profit, cost

    if not backtest:
        ask_price = float(ticker['info']['askPrice'])
    else:
        ask_price = float(df.close.iloc[-1])


    # MACD crossover, long-term uptrend
    #if (df.macd.iloc[-2] < 0) and (df.macd.iloc[-1] > 0) and (ask_price > df.ema100.iloc[-1]):

    if not df.in_uptrend.iloc[-2] and df.in_uptrend.iloc[-1]:
        if not in_position:
            quantity_buy_sell = (usd_stake / ask_price)
            cost = (quantity_buy_sell * ask_price)
            log_write(f'\n*** BUY {symbol} @ {df.timestamp.iloc[-1]} ***\namount: {quantity_buy_sell:.4f}\tprice: {df.close.iloc[-1]}\tcost: {cost:.2f}')
            #order = exchange.create_market_buy_order(symbol, quantity_buy_sell)
            #log_write(order)
            in_position = True

            #log_write(df.tail(2), df=True)

    if df.in_uptrend.iloc[-2] and not df.in_uptrend.iloc[-1]:
        if in_position:
            received = (quantity_buy_sell * ask_price)
            trade_profit = (received - cost)
            profit += trade_profit
            log_write(f'\n*** SELL {symbol} @ {df.timestamp.iloc[-1]} ***\namount: {quantity_buy_sell}\tprice: {df.close.iloc[-1]}\tprofit: {trade_profit:.2f}')
            #order = exchange.create_market_sell_order(symbol, quantity_buy_sell)
            #log_write(order)
            in_position = False

            #log_write(df.tail(2), df=True)


def run_backtest(df, ticker):
    df_size = df.shape[0]
    log_write(f'Running backtest on {df_size} data points...')
    for i in range(0, df_size):
        if i > 200:
            check_buy_sell_signals(df[0:i], ticker)


def run_bot(backtest=False):
    global symbol, timeframe, limit

    df = None

    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        ticker = exchange.fetchTicker(symbol)

        df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
    except ccxt.NetworkError as e:
        msg = f'{exchange.id} - fetch_ohlcv failed due to a network error: {e}'
    except ccxt.ExchangeError as e:
        msg = f'{exchange.id} - fetch_ohlcv failed due to an exchange error: {e}'
    except Exception as e:
        msg = f'{exchange.id} - fetch_ohlcv failed due to an error: {e}'

    if df is None:
        log_write(msg)
    else:
        stoch = stoch_rsi(df, period=14, smooth=3)
        df['stoch_k'] = round((stoch.stochrsi_k() * 100), 2)
        df['stoch_d'] = round((stoch.stochrsi_d() * 100), 2)
        df['macd'] = macd(df)
        df['rsi'] = rsi(df, period=14)
        df['ema100'] = ema(df, period=100)
        df['ema200'] = ema(df, period=200)
        df['cmf'] = cmf(df)
        df['adx'] = adx(df)
        df['psar'] = psar(df)
        #add_ichimoku(df)
        supertrend_data = supertrend(df, period=7, atr_multiplier=3)

        if not backtest:
            check_buy_sell_signals(supertrend_data, ticker)
        else:
            log_write(f'Backtesting from {df.timestamp.iloc[0]} to {df.timestamp.iloc[-1]}')
            run_backtest(supertrend_data, ticker)
            print(supertrend_data)


def main():
    global profit, backtest

    msg = f'Bot started - symbol: {symbol}, timeframe: {timeframe}, limit: {limit}, in_position: {in_position}, quantity_buy_sell: {quantity_buy_sell}, usd_stake: {usd_stake}'
    log_write(msg)

    if not backtest:
        schedule.every(10).seconds.do(run_bot, backtest=backtest)

        while True:
            schedule.run_pending()
            time.sleep(exchange.rateLimit / 1000)
    else:
        run_bot(backtest=backtest)
        log_write(f'Backtest finished, total profit: {profit:.2f}')


if __name__ == '__main__':
    main()
