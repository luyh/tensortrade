import ccxt
import os,time

okex = ccxt.okex3()
okex.apiKey = os.getenv('apiKey')
okex.secret = os.getenv('secret')
okex.password = os.getenv('password')
print(okex.fetch_balance())

from tensortrade.exchanges.live.ccxt_exchange import CCXTExchange
exchange = CCXTExchange(exchange = okex,observation_symbol='BTC/USDT', timeframe='1h')
print(exchange.balance)

for _ in range(5):
    next_observation = exchange.next_observation()
    time.sleep(3)



print('debug')