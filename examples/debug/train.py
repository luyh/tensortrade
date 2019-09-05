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
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.rewards import SimpleProfitStrategy

exchange = FBMExchange( times_to_generate=100000 )
action_strategy = DiscreteActionStrategy()
reward_strategy = SimpleProfitStrategy()

env = TradingEnvironment( exchange=exchange,
                          action_strategy=action_strategy,
                          reward_strategy=reward_strategy,
                          feature_pipeline=None )

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

network_spec = [
    dict( type='dense', size=64 ),
    dict( type='dense', size=32 )
]

agent = Agent.from_spec(
    spec=agent_config,
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
    return True


# Start learning
runner.run( episodes=300, max_episode_timesteps=10000, episode_finished=episode_finished )
runner.close()

# Print statistics
print( "Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean( runner.episode_rewards ) )
)