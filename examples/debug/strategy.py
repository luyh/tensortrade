from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.environments import TradingEnvironment
from tensortrade.strategies.tensorforce_trading_strategy import TensorforceTradingStrategy

normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
difference = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(normalize_price, difference)
exchange = FBMExchange(base_instrument='BTC',
                       timeframe='1h',
                       feature_pipeline=feature_pipeline)
reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20,
                                        instrument_symbol='ETH/BTC')
environment = TradingEnvironment(exchange=exchange,
                                 feature_pipeline=feature_pipeline,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy)

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

performance = strategy.run(steps=100000,
                           should_train=True,
                           episode_callback=episode_finished)

print('done')