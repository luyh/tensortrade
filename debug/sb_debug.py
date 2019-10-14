from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.strategies import TensorforceTradingStrategy

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from tensortrade.strategies import StableBaselinesTradingStrategy

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='ETH/BTC')

exchange = FBMExchange(base_instrument='BTC',
                       timeframe='1h',
                       times_to_generate = int(1e6),
                       should_pretransform_obs=True,
                       feature_pipeline=feature_pipeline)

exchange.reset()

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)


model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5 }
agent = model(policy, environment, model_kwargs=params)


sb_strategy = StableBaselinesTradingStrategy(environment=environment,
                                            model=model,
                                            policy=policy,
                                             model_kwargs=params)

##tensorforce_stragegy
# agent_spec = {
#     "type": "ppo_agent",
#     "step_optimizer": {
#         "type": "adam",
#         "learning_rate": 1e-4
#     },
#     "discount": 0.99,
#     "likelihood_ratio_clipping": 0.2,
# }
#
# network_spec = [
#     dict(type='dense', size=64, activation="tanh"),
#     dict(type='dense', size=32, activation="tanh")
# ]
#
# a_strategy = TensorforceTradingStrategy(environment=environment,
#                                       agent_spec=agent_spec,
#                                       network_spec=network_spec)

performance = sb_strategy.run(episodes=1)

print(performance[-5:])
print('done')