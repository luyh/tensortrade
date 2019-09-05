import time
import numpy as np
import pandas as pd

from typing import Dict, List
from gym.spaces import Space, Box
from ccxt import Exchange

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import AssetExchange

from onetoken_api.exchange import Exchange


class OnetokenExchange(AssetExchange):
    """An asset exchange for trading on CCXT-supported cryptocurrency exchanges."""

    def __init__(self, exchange: Exchange, base_asset: str = 'btc', **kwargs):
        super().__init__(dtype=kwargs.get('dtype', np.float16))

        self._exchange = exchange
        self._base_asset = base_asset

        self._exchange.enableRateLimit = kwargs.get('enable_rate_limit', True)

        self._observation_type = kwargs.get('observation_type', 'trades')
        self._observation_symbol = kwargs.get('observation_symbol', 'eos.btc')
        self._observation_timeframe = kwargs.get('observation_timeframe', '10m')
        self._observation_window_size = kwargs.get('observation_window_size', 10)

        self._async_timeout_in_ms = kwargs.get('async_timeout_in_ms', 15)
        self._max_trade_wait_in_sec = kwargs.get('max_trade_wait_in_sec', 60)

    @property
    def base_precision(self) -> float:
        return self._markets[self._markets.name == self._observation_symbol]['min_change'].values[0]

    @base_precision.setter
    def base_precision(self, base_precision: float):
        raise ValueError('Cannot set the precision of `onetoken` exchanges.')

    @property
    def asset_precision(self) -> float:
        return self._markets[self._markets.name == self._observation_symbol]['unit_amount'].values[0]

    @asset_precision.setter
    def asset_precision(self, asset_precision: float):
        raise ValueError('Cannot set the precision of `onetoken` exchanges.')

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        self._exchange.get_balance()
        return self._exchange.position.loc[self._base_asset, 'available']

    @property
    def portfolio(self) -> pd.DataFrame:
        self._exchange.get_balance()

        return self._exchange.position

    @property
    def trades(self) :
        return self._exchange.get_trans()

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def observation_space(self) -> Space:
        low_price = 0
        high_price = np.inf
        low_volume = 0
        high_volume = np.inf

        if self._observation_type == 'ohlcv':
            low = (low_price, low_price, low_price, low_price, low_volume)
            high = (high_price, high_price, high_price, high_price, high_volume)
            dtypes = (self._dtype, self._dtype, self._dtype, self._dtype, np.int64)
            obs_shape = (self._observation_window_size, 5)
        else:
            low = (0, low_price, low_price, low_price, low_volume)
            high = (1, high_price, high_volume, high_price * high_volume)
            dtypes = (np.int8, self._dtype, self._dtype, self._dtype)
            obs_shape = (self._observation_window_size, 4)

        return Box(low=low, high=high, shape=obs_shape, dtype=dtypes)

    @property
    def has_next_observation(self) -> bool:
        if self._observation_type == 'ohlcv':
            return self._exchange.has['fetchOHLCV']

        return self._exchange.has['fetchTrades']

    def next_observation(self) -> pd.DataFrame:
        if self._observation_type == 'ohlcv':
            ohlcv = self._exchange.fetch_ohlcv(
                self._observation_symbol, timeframe=self._observation_timeframe)

            obs = [l[1:] for l in ohlcv]
        elif self._observation_type == 'trades':
            trades = self._exchange.fetch_trades(self._observation_symbol)

            obs = [[0 if t['side'] == 'buy' else 1, t['price'], t['amount'], t['cost']]
                   for t in trades]

        if len(obs) < self._observation_window_size:
            return np.pad(obs, (self._observation_window_size - len(obs), len(obs[0])))

        return obs

    def current_price(self, symbol: str) -> float:
        return self._exchange.fetch_ticker(symbol)['close']

    def execute_trade(self, trade: Trade) -> Trade:
        if trade.trade_type == TradeType.LIMIT_BUY:
            order = self._exchange.create_limit_buy_order(
                symbol=trade.symbol, amount=trade.amount, price=trade.price)
        elif trade.trade_type == TradeType.MARKET_BUY:
            order = self._exchange.create_market_buy_order(symbol=trade.symbol, amount=trade.amount)
        elif trade.trade_type == TradeType.LIMIT_SELL:
            order = self._exchange.create_limit_sell_order(
                symbol=trade.symbol, amount=trade.amount, price=trade.price)
        elif trade.trade_type == TradeType.MARKET_SELL:
            order = self._exchange.create_market_sell_order(
                symbol=trade.symbol, amount=trade.amount)

        max_wait_time = time.time() + self._max_trade_wait_in_sec

        while order['status'] is 'open' and time.time() < max_wait_time:
            order = self._exchange.fetch_order(order.id)

        if order['status'] is 'open':
            self._exchange.cancel_order(order.id)

        self._performance.append({
            'balance': self.balance,
            'net_worth': self.net_worth,
        }, ignore_index=True)

        return Trade(symbol=trade.symbol, trade_type=trade.trade_type, amount=order['filled'], price=order['price'])

    def reset(self):
        self._markets = self._exchange.contract
        self._exchange.close()
        time.sleep(3)
        self._exchange.get_balance()
        self._initial_balance = self._exchange.position.loc[self._base_asset,'available']
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

if __name__ == '__main__':
    okex = Exchange('okex/mock-luyh-okex')

    exchange = OnetokenExchange(exchange = okex,base_asset = 'btc')
    exchange.reset()

    base_precision = exchange.base_precision
    asset_precision = exchange.asset_precision

    portfolio = exchange.portfolio

    trades = exchange.trades

    print('debug')


