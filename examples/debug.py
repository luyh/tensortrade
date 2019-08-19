
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('')))

from tensortrade.environments import TradingEnvironment
from tensortrade.environments.actions.discrete import SimpleDiscreteStrategy
from tensortrade.environments.rewards.simple import IncrementalProfitStrategy
from tensortrade.exchanges.simulated import FBMExchange

env = DummyVecEnv([lambda: TradingEnvironment(action_strategy=SimpleDiscreteStrategy(),
                                              reward_strategy=IncrementalProfitStrategy(),
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