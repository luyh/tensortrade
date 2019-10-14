import sys
import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter( action='ignore', category=FutureWarning )

import gym
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

sys.path.append( os.path.dirname( os.path.abspath( '' ) ) )

from tensortrade.environments import TradingEnvironment
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.exchanges.simulated import FBMExchange,SimulatedExchange
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.rewards import SimpleProfitStrategy
import pandas as pd

df = pd.read_csv('./data/coinbase-1h-btc-usd.csv')
df = df[['Open','High','Low','Close','VolumeFrom']]
df.rename(columns={'Open': 'open',
                   'High': 'high',
                   'Low' : 'low',
                   'Close':'close',
                   'VolumeFrom':'volumn'
                    }, inplace = True)

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True,)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

#exchange = FBMExchange( times_to_generate=100000 )
exchange = SimulatedExchange(data_frame=df, base_instrument='USD')
action_strategy = DiscreteActionStrategy()
reward_strategy = SimpleProfitStrategy()

env = TradingEnvironment( exchange=exchange,
                          action_strategy=action_strategy,
                          reward_strategy=reward_strategy,
                          feature_pipeline=feature_pipeline )

agent_config = {
    "type": "dqn_agent",

    "update_mode": {
        "unit": "timesteps",
        "batch_size": 64,
        "frequency": 4
    },

    "memory": {
        "type": "replay",
        "capacity": 10000,
        "include_next_states": True
    },

    "optimizer": {
        "type": "clipped_step",
        "clipping_value": 0.1,
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-3
        }
    },

    "discount": 0.999,
    "entropy_regularization": None,
    "double_q_model": True,

    "target_sync_frequency": 1000,
    "target_update_weight": 1.0,

    "actions_exploration": {
        "type": "epsilon_anneal",
        "initial_epsilon": 0.5,
        "final_epsilon": 0.,
        "timesteps": 1000000000
    },

    "saver": {
        "directory": None,
        "seconds": 600
    },
    "summarizer": {
        "directory": None,
        "labels": ["graph", "total-loss"]
    },
    "execution": {
        "type": "single",
        "session_config": None,
        "distributed_spec": None
    }
}

ppo_agent_spec = {
    "type": "ppo_agent",
    "step_optimizer": {
        "type": "adam",
        "learning_rate": 1e-4
    },
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
}

network_spec = [
    dict( type='dense', size=64 ),
    dict( type='dense', size=32 )
]

agent = Agent.from_spec(
    spec=ppo_agent_spec,
    kwargs=dict(
        states=env.states,
        actions=env.actions,
        network=network_spec,
    )
)

# Create the runner
runner = Runner( agent=agent, environment=env )


# Callback function printing episode statistics
def episode_finished(r):
    print( "Finished episode {ep} after {ts} timesteps (reward: {reward})".format( ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1] ) )
    #print(r.environment.exchange.trades)
    return True


# Start learning
if sys.platform == 'win32':
    runner.run( episodes=2, max_episode_timesteps=100, episode_finished=episode_finished )
else:
    runner.run(episodes=300, max_episode_timesteps=10000, episode_finished=episode_finished)
runner.close()

# Print statistics
print( "Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean( runner.episode_rewards ) )
)

###
# output information:
#  50%|█████     | 1/2 [02:02<02:02, 122.89s/it]Finished episode 1 after 101 timesteps (reward: -95.33134897071872)
# 100%|██████████| 2/2 [04:04<00:00, 122.62s/it]
# Finished episode 2 after 101 timesteps (reward: -213.66788510414324)
# Learning finished. Total episodes: 2. Average reward of last 100 episodes: -154.499617037431.