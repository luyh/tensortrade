import pandas as pd
STEP = 100
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Open','High','Low','Close','VolumeFrom']]

    df.rename(columns={'Open': 'open',
                       'High': 'high',
                       'Low' : 'low',
                       'Close':'close',
                       'VolumeFrom':'volumn'
                        }, inplace = True)

    return df

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy

def get_env(file_path):
    df = load_data(file_path)

    normalize = MinMaxNormalizer(inplace=True)
    difference = FractionalDifference(difference_order=0.6, inplace=True)
    feature_pipeline = FeaturePipeline(steps=[normalize, difference])

    reward_strategy = SimpleProfitStrategy()
    action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='BTC/USDT')

    exchange = SimulatedExchange(base_instrument='USDT',
                                 should_pretransform_obs=True,
                                 feature_pipeline=feature_pipeline
                                 )
    exchange.data_frame = df[:STEP]
    environment = TradingEnvironment(exchange=exchange,
                                     action_strategy=action_strategy,
                                     reward_strategy=reward_strategy,
                                     feature_pipeline=feature_pipeline)

    return environment

if __name__ == '__main__':
    file_path = './exchange/data/coinbase-1h-btc-usd.csv'
    environment = get_env(file_path)

    print('environment.exchange.data_frame','\n',environment.exchange.data_frame.head())

    obs = environment.reset()

    action = 1
    steps = 0
    episodes = 5
    steps_completed = 0
    episodes_completed = 0
    average_reward = 0

    performance = {}
    while (episodes is not None and episodes_completed < episodes):

        obs, reward, done, info = environment.step(action)

        if steps_completed % 10 == 0:
            print('eposide: {}/{},step: {}/{}'.format(episodes_completed, episodes, steps_completed, steps))

        steps_completed += 1
        average_reward -= average_reward / steps_completed
        average_reward += reward / (steps_completed + 1)

        exchange_performance = info.get('exchange').performance
        performance = exchange_performance if len(exchange_performance) > 0 else performance

        if done:
            episodes_completed += 1
            obs = environment.reset()

    print("Finished testing btc_simulate_environment.")
    print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
    print("Average reward: {}.".format(average_reward))
    print("Performance[-5]: {}.".format(performance[-5:]))

    print('done')