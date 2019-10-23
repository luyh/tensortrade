from develop.strategy.environment.btc_simulate import get_env
from tensortrade.strategies.neat_trading_strategy import NeatTradingStrategy

df_file_path = 'environment/exchange/data/coinbase-1h-btc-usd.csv'

environment = get_env(df_file_path)

CONFIG = "./config"
neat_strategy = NeatTradingStrategy(environment = environment,neat_config = CONFIG)

performance, winner, stats = neat_strategy.run(generations = 10, testing = False)