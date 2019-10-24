from develop.strategy.environment.btc_simulate import get_env
from tensortrade.strategies.neat_trading_strategy import NeatTradingStrategy
import visualize,neat
import pickle

df_file_path = 'environment/exchange/data/coinbase-1h-btc-usd.csv'

environment = get_env(df_file_path)

CONFIG = "./config"
neat_strategy = NeatTradingStrategy(environment = environment,neat_config = CONFIG)

def run():
    performance, winner, stats = neat_strategy.run(generations = 10, testing = False)

    # visualize training
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

def evaluate(CHECKPOINT):
    p = neat.Checkpointer.restore_checkpoint('./checkpoint/neat-checkpoint-%i' % CHECKPOINT)
    #winner = p.run(neat_strategy._eval_population, 1)  # find the winner in restored population
    winner = p.population[947]

    with open('./checkpoint/winner.pkl', 'w') as f:
        pickle.dumps(winner, f)

    with open('./checkpoint/winner.pkl', 'r') as f:
        winner = winner.load(f)

    neat_strategy.eval_genome(winner)

    visualize.draw_net(p.config, winner, True)

TRAINING = False
if __name__ == '__main__':
    if TRAINING:
        run()
    else:
        CHECKPOINT= 9
        evaluate(CHECKPOINT)

print('done')