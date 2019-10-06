import ccxt
import os
from tensortrade.exchanges.live.ccxt_exchange import CCXTExchange

okex = ccxt.okex()
okex.apiKey = os.getenv('apiKey')
okex.secret = os.getenv('secret')

exchange = CCXTExchange(exchange = okex,
                        observation_symbol='BTC/USDT',
                        base_instrument = 'USDT',
                        timeframe='1h')
balance = exchange.balance

next_observation = exchange.next_observation()



print('debug')