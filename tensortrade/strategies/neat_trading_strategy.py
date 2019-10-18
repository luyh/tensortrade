# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

import neat

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy


class NeatTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating using the NEAT Neuralevolution."""

    # todo: pass in config file
    def __init__(self, environment: TradingEnvironment, neat_config: str, **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            neat_sepc: A specification dictionary for the `Tensorforce` agent's model network.
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        self._environment = environment

        self._max_episode_timesteps = kwargs.get('max_episode_timesteps', None)
        self._neat_config_filename = neat_config
        self._config = self.load_config()
        # save config file
        # self._agent = Agent.from_spec(spec=agent_spec,
        #                               kwargs=dict(network=network_spec,
        #                                           states=environment.states,
        #                                           actions=environment.actions))
        #
        # self._runner = Runner(agent=self._agent, environment=environment)

    # @property
    # def agent(self):
    #     """A Tensorforce `Agent` instance that will learn the strategy."""
    #     return self._agent

    @property
    def environment(self):
        return self._environment

    def load_config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         self._neat_config_filename)

    def restore_agent(self, path: str, model_path: str = None):
        raise NotImplementedError

    def save_agent(self, path: str, model_path: str = None, append_timestep: bool = False):
        raise NotImplementedError

    def _finished_episode_cb(self) -> bool:
        n_episodes = runner.episode
        n_timesteps = runner.episode_timestep
        avg_reward = np.mean(runner.episode_rewards)

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def eval_genome(self, genomes, config):
        print("config", config)
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # while True:
            #     steps_completed = 0
            #     episodes_completed = 0
            #     average_reward = 0
            #
            #     obs, state, dones = self._environment.reset(), None, [False]
            #
            #     performance = {}
            #
            #     while (steps is not None and (steps == 0 or steps_completed < steps)) or (episodes is not None and episodes_completed < episodes):
            #         actions, state = self._agent.predict(obs, state=state, mask=dones)
            #         obs, rewards, dones, info = self._environment.step(actions)
            #
            #         steps_completed += 1
            #         average_reward -= average_reward / steps_completed
            #         average_reward += rewards[0] / (steps_completed + 1)
            #
            #         exchange_performance = info[0].get('exchange').performance
            #         performance = exchange_performance if len(exchange_performance) > 0 else performance
            #
            #         if dones[0]:
            #             if episode_callback is not None and episode_callback(self._environment._exchange.performance):
            #                 break
            #
            #             episodes_completed += 1
            #             obs = self._environment.reset()

            # net = neat.nn.FeedForwardNetwork.create(genome, config)
            # while True:
            #     action = net.activate(xi)
            #     observation, reward, done, info = self.environment.step(action)
            #     # env.render()
            #     if done:
            #         print("info:", info)
            #         break

        return reward

    def run(self, generations: int = None, testing: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:

        # create population
        p = neat.Population(self._config)

        # add reporting
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(self.eval_genome, generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = winner_net.activate(xi)
        #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        p.run(self._environment.reward_strategy, 10)



        # self._runner.run(testing=testing,
        #                  num_timesteps=steps,
        #                  num_episodes=episodes,
        #                  max_episode_timesteps=self._max_episode_timesteps,
        #                  episode_finished=episode_callback)
        #
        # n_episodes = self._runner.episode
        # n_timesteps = self._runner.timestep
        # avg_reward = np.mean(self._runner.episode_rewards)
        #
        # print("Finished running strategy.")
        # print("Total episodes: {} ({} timesteps).".format(n_episodes, n_timesteps))
        # print("Average reward: {}.".format(avg_reward))
        #
        # self._runner.close()

        return self._environment._exchange.performance
