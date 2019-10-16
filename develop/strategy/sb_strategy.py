from develop.envirnment.btc_simulate import environment

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from tensortrade.strategies import StableBaselinesTradingStrategy

model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5 ,
           'verbose':1,
           'nminibatches':1}

strategy = StableBaselinesTradingStrategy(environment=environment,
                                            model=model,
                                            policy=policy,
                                             model_kwargs=params)

