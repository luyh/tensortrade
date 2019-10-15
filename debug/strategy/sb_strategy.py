import pandas as pd

STEP = 100
df = pd.read_csv('./data/coinbase-1h-btc-usd.csv')
df = df[['Open','High','Low','Close','VolumeFrom']][:STEP]
df.rename(columns={'Open': 'open',
                   'High': 'high',
                   'Low' : 'low',
                   'Close':'close',
                   'VolumeFrom':'volumn'
                    }, inplace = True)

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import FBMExchange,SimulatedExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy

exchange = SimulatedExchange(data_frame=df, base_instrument='USD')

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='ETH/BTC')

# exchange = FBMExchange(base_instrument='BTC',
#                        timeframe='1h',
#                        times_to_generate = int(1e5),
#                        should_pretransform_obs=True,
#                        feature_pipeline=feature_pipeline)



environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)


from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from tensortrade.strategies import StableBaselinesTradingStrategy

model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5 ,
           'verbose':1,
           'nminibatches':1}

sb_strategy = StableBaselinesTradingStrategy(environment=environment,
                                            model=model,
                                            policy=policy,
                                             model_kwargs=params)

performance = sb_strategy.run(episodes=5)

print(performance[-5:])
print('done')