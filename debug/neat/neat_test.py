from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.environments import TradingEnvironment
from tensortrade.strategies.neat_trading_strategy import NeatTradingStrategy
from tensortrade.environments.neat_environment import NeatEnvironment

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='ETH/BTC')

exchange = FBMExchange(base_instrument='BTC',
                       timeframe='1h',
                       should_pretransform_obs=True)
exchange.reset()
environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

# neat_env = NeatEnvironment(exchange=exchange,
#                                  action_strategy=action_strategy,
#                                  reward_strategy=reward_strategy,
#                                  feature_pipeline=feature_pipeline)
CONFIG = "./config"
neat_strategy = NeatTradingStrategy(environment = environment,neat_config = CONFIG)

performance, winner, stats = neat_strategy.run(generations = 10, testing = False)