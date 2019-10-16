from develop.envirnment.data.load_data import df
from ..config import STEP

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy


normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='BTC/USDT')

exchange = SimulatedExchange(base_instrument='USDT',
                             should_pretransform_obs=True,
                             feature_pipeline=feature_pipeline
                             )

exchange.data_frame = df[:STEP]

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)