%load_ext zipline

% % zipline - -start 2017 - 4 - 23 - -end 2018 - 3 - 27 - -capital - base 1050.0 - o
sma_strategy.pkl
from zipline.api import order_target, record, symbol, get_datetime, set_benchmark
from zipline.finance import commission
from zipline import run_algorithm
import matplotlib.pyplot as plt
import numpy as np
import pytz
import pandas as pd
import datetime

# parameters
ma_periods = 3
selected_stock = 'AAPL'
n_stocks_to_buy = 10


def initialize(context):
    context.time = 0
    context.asset = symbol(selected_stock)
    set_benchmark(symbol('AAPL'))
    # 1. manually setting the commission
    context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))


def handle_data(context, data):
    # 2. warm-up period
    context.time += 1
    # if context.time < ma_periods:
    # return
    # 3. access price history
    price_history = data.history(context.asset, fields="price", bar_count=ma_periods, frequency="1d")
    return1 = price_history[1] / price_history[0] - 1
    pred = pd.read_csv("test1.csv")
    pred["pred return"] = pred["adj_close"].shift(1) / pred["adj_close"] - 1
    pred["pred return"] = pred["pred return"].fillna(0)
    pred["date"] = pd.to_datetime(pred["date"])
    # 4. calculate moving averages
    # ma2 = price_history.mean()
    tday = get_datetime().date()
    pr1 = pred[pred["date"] == tday]["pred return"].values
    ma1 = pred[pred["date"] == tday]["adj_close"].values
    try:
        ma = ma1[0]
        pr = pr1[0]
    except:
        ma = price_history.mean()
        date = tday - datetime.timedelta(days=1)
        if date == '2017-08-07':
            pr = pred[pred["date"] == '2017-08-05']["pred return"].values
        else:
            pr = pred[pred["date"] == '2017-11-07']["pred return"].values
        pr = pr[0]

    # 5. trading logic
    # cross up
    # if (price_history[-2] < ma) & (price_history[-1] > ma):
    # order_target(context.asset, n_stocks_to_buy)
    # cross down
    # elif (price_history[-2] > ma) & (price_history[-1] < ma):
    # order_target(context.asset, 0)
    if (return1 * pr) < 0:
        print(tday, return1, pr)
        if return1 < 0 and pr > 0:
            print("sell:{}".format(tday))
            order_target(context.asset, 0)
        elif return1 > 0 and pr < 0:
            print("buy:{}".format(tday))
            order_target(context.asset, n_stocks_to_buy)

    # save values for later inspection
    record(price=data.current(context.asset, 'price'),
           predict_price=ma, predict_returns=pr)


# 6. analyze block
def analyze(context, perf):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=[16, 9])
    # portfolio value
    perf.portfolio_value.plot(ax=ax[0])
    ax[0].set_ylabel('portfolio value in $')

    # asset
    perf[['price', 'predict_price']].plot(ax=ax[1])
    ax[1].set_ylabel('price in $')

    # mark transactions
    perf_trans = perf.loc[[t != [] for t in perf.transactions]]
    buys = perf_trans.loc[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.loc[[t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax[1].plot(buys.index, perf.price.loc[buys.index], '^', markersize=10, color='g', label='buy')
    ax[1].plot(sells.index, perf.price.loc[sells.index], 'v', markersize=10, color='r', label='sell')
    ax[1].legend()

    # daily returns
    print(perf.columns)
    perf.algorithm_period_return.plot(ax=ax[2])
    perf.benchmark_period_return.plot(ax=ax[2])
    ax[2].set_ylabel('daily returns')

    fig.suptitle('Simple Moving Average Strategy - Apple', fontsize=16)
    plt.legend()
    plt.show()

    print('Final portfolio value (including cash): {}$'.format(np.round(perf.portfolio_value[-1], 2)))