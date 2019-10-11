import ccxt

from tensorforce.agents import Agent
from tensorforce.environments import Environment

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.strategies import TensorforceTradingStrategy

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True,
                                  all_column_names=["open", "high", "low", "close", "volume"])
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='ETH/BTC')

exchange = FBMExchange(base_instrument='BTC',
                       timeframe='1h',
                       times_to_generate = int(1e6),
                       should_pretransform_obs=True)

exchange.reset()

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

agent_spec = {
    "type": "ppo_agent",
    "step_optimizer": {
        "type": "adam",
        "learning_rate": 1e-4
    },
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
}

network_spec = [
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

strategy = TensorforceTradingStrategy(environment=environment,
                                      agent_spec=agent_spec,
                                      network_spec=network_spec)

# Callback function printing episode statistics
def episode_finished(r):
    print( "Finished episode {ep} after {ts} timesteps (reward: {reward})".format( ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1] ) )

performance = strategy.run(episodes=2,
                           testing=True,
                           episode_callback=episode_finished)

print(performance[-5:])
print('done')