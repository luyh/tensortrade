from develop import EPOSIDE
from develop.strategy.environment.btc_simulate import get_env

from tensortrade.strategies import TensorforceTradingStrategy
from tensorforce.agents import Agent


dqn_config = {
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

ppo_spec = {
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

def agent(env,sepc = ppo_spec):

    return Agent.from_spec(
        spec=sepc,
        kwargs=dict(
            states=env.states,
            actions=env.actions,
            network=network_spec,
        )
    )


def load_strategy(df_file_path,agent_spec = ppo_spec):
    env = get_env(df_file_path)
    strategy = TensorforceTradingStrategy(environment=env,
                                          agent_spec= agent_spec,
                                          network_spec=network_spec)

    return strategy

if __name__ == '__main__':
    df_file_path = './exchange/data/coinbase-1h-btc-usd.csv'
    strategy = load_strategy(df_file_path, agent_spec=ppo_spec)

    performance = strategy.run(episodes=EPOSIDE, testing=True)

    print(performance[-5:])
    print('done')