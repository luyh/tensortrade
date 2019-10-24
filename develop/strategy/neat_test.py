from develop.strategy.environment.btc_simulate import get_env
from tensortrade.strategies.neat_trading_strategy import NeatTradingStrategy

df_file_path = 'environment/exchange/data/coinbase-1h-btc-usd.csv'

environment = get_env(df_file_path)

CONFIG = "./config"
neat_strategy = NeatTradingStrategy(environment = environment,neat_config = CONFIG)

def run():
    performance, winner, stats = neat_strategy.run(generations = 10, testing = False)

    # visualize training
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

def evaluate(CHECKPOINT):

    neat_strategy.evaluation(CHECKPOINT)
    # visualize.draw_net(p.config, winner, True, node_names=node_names)

TRAINING = False
if __name__ == '__main__':
    if TRAINING:
        run()
    else:
        CHECKPOINT= 9
        evaluate(CHECKPOINT)

print('done')