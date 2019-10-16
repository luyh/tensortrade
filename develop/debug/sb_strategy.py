from develop.envirnment.btc_simulate import environment
from ..config import STEP,EPISODE

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

performance = sb_strategy.run(episodes=EPISODE)

print(performance[-5:])
print('done')