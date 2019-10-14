from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.environments import TradingEnvironment

exchange = FBMExchange(base_instrument='BTC', timeframe='1h')
normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
difference = FractionalDifference(difference_order=0.6)

#feature_pipeline = FeaturePipeline(normalize_price)
feature_pipeline = FeaturePipeline(normalize_price, difference)
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

from tensortrade.strategies import TensorforceStrategy
strategy = TensorforceStrategy(environment=environment,
                                      agent_spec=agent_spec,
                                      network_spec=network_spec)

performance = strategy.run(steps=100000,
                           should_train=True,
                           #episode_callback=stop_early_callback,
                            )

strategy.save_agent(path="../agents/ppo_btc_1h")

print('debug done')