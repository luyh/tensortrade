import pandas as pd
df = pd.read_csv('./input/coinbase-1h-btc-usd.csv')


from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath('')))

from tensortrade.environments import TradingEnvironment
from tensortrade.actions.simple_discrete_strategy import SimpleDiscreteStrategy
from tensortrade.rewards.simple_profit_strategy import SimpleProfitStrategy
from tensortrade.exchanges.simulated import FBMExchange,GANExchange



env = DummyVecEnv([lambda: TradingEnvironment(action_strategy=SimpleDiscreteStrategy(),
                                              reward_strategy=SimpleProfitStrategy(),
                                              exchange=FBMExchange())])

agent = PPO2(MlpLnLstmPolicy, env, verbose=1, nminibatches=1)

def evaluate(agent, num_steps=1000):
    obs = env.reset()

    state = None
    exchange = None

    for _ in range( 2500 ):
        action, state = agent.predict( obs, state=state )
        obs, reward, _, info = env.step( action )

        exchange = info[0]['exchange']

    print( 'Trades: ', exchange.trades() )
    print( 'Balance: ', exchange.balance() )
    print( 'Portfolio: ', exchange.portfolio() )
    print( 'P/L: ', exchange.profit_loss_percent() )

evaluate(agent,num_steps=1000)

print('debug')