from ..strategy.tf_strategy import agent
from develop.envirnment.btc_simulate import environment
from tensorforce.execution import Runner
from ..config import STEP,EPISODE
import numpy as np


agent = agent()

# Create the runner
runner = Runner( agent=agent, environment=environment )


# Callback function printing episode statistics
def episode_finished(r):
    print( "Finished episode {ep} after {ts} timesteps (reward: {reward})".format( ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1] ) )
    #print(r.environment.exchange.trades)
    return True


# Start learning
import sys
if sys.platform == 'win32':
    runner.run( episodes=EPISODE, max_episode_timesteps=100, episode_finished=episode_finished )
else:
    runner.run(episodes=EPISODE*20, max_episode_timesteps=10000, episode_finished=episode_finished)
runner.close()

# Print statistics
print( "Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean( runner.episode_rewards ) )
)

###
# output information:
#  50%|█████     | 1/2 [02:02<02:02, 122.89s/it]Finished episode 1 after 101 timesteps (reward: -95.33134897071872)
# 100%|██████████| 2/2 [04:04<00:00, 122.62s/it]
# Finished episode 2 after 101 timesteps (reward: -213.66788510414324)
# Learning finished. Total episodes: 2. Average reward of last 100 episodes: -154.499617037431.