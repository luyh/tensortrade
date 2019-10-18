EPOSIDE = 50

from environment.btc_simulate import get_env
from matplotlib import pyplot as plt

from tensortrade.strategies import TensorforceTradingStrategy
from tensorforce.agents import Agent

network_spec = [
    #dict(type='dense', size=128, activation="tanh"),
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

agent_spec = {
    "type": "ppo",
    "learning_rate": 1e-4,
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
    "estimate_terminal": False,
    "max_episode_timesteps": 2000,
    "network": network_spec,
    "batch_size": 10,
    "update_frequency": "never"
}


def load_strategy(df_file_path,agent_spec = agent_spec):
    env = get_env(df_file_path)
    strategy = TensorforceTradingStrategy(environment=env,
                                          agent_spec= agent_spec)

    return strategy

import numpy as np
def episode_callback(r):
    if r.global_episodes %10 ==0:
        print( '  average_reward:  %.2f' %(np.mean(r.episode_rewards[:-10])))

    return True

import datetime
if __name__ == '__main__':
    df_file_path = 'environment/exchange/data/coinbase-1h-btc-usd.csv'
    strategy = load_strategy(df_file_path, agent_spec=agent_spec)

    if True:
        try:
            strategy.restore_agent(directory="./data/agents\agent")
        except:pass

    performance = strategy.run(episodes=EPOSIDE ,evaluation=False,episode_callback=episode_callback)
    print(performance[-5:])
    performance.balance.plot()
    #plt.show()
    now = datetime.datetime.now().strftime("%m%d%H%M%S")
    plt.savefig('./data/figure/balance_{}.png'.format(now))

    checkpoint_path = strategy.save_agent(directory='./data/agents')
    print(checkpoint_path)

    print('done')