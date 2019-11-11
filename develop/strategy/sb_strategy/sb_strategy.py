from develop.strategy.environment.btc_simulate import get_env

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from tensortrade.strategies import StableBaselinesTradingStrategy

def load_strategy(df_file_path,model,policy):
    env = get_env(df_file_path)
    strategy = StableBaselinesTradingStrategy(environment=env,
                                                model=model,
                                                policy=policy,
                                                model_kwargs=params)
    return strategy

if __name__ == '__main__':
    model = PPO2
    policy = MlpLnLstmPolicy
    params = {"learning_rate": 1e-5,
              'verbose': 1,
              'nminibatches': 1,
              'tensorboard_log':"./logs/"}

    df_file_path = './environment/exchange/data/coinbase-1h-btc-usd.csv'
    strategy = load_strategy(df_file_path,model,policy)
    try:
        strategy.restore_agent("ppo_btc_1h")
    except:pass
    strategy._agent.learn(total_timesteps=5000)
    strategy.save_agent("ppo_btc_1h")

    performance = strategy.run(episodes = 1)

    print(performance[-5:])
    print('done')