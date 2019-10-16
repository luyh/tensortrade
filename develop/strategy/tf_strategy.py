from develop.envirnment.btc_simulate import environment

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

def agent(sepc = ppo_spec):

    return Agent.from_spec(
        spec=sepc,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec,
        )
    )

strategy = TensorforceTradingStrategy(environment=environment,
                                      agent_spec=agent(sepc = ppo_spec),
                                      network_spec=network_spec)

