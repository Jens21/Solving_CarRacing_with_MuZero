from shared_memory import SharedMemory as SharedMemory
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
import gym
from pyvirtualdisplay import Display
import numpy as np
from torch.multiprocessing import Process, Pool
from network import Network as Network
import torch as th
import time

class Simulator():
    processes = []

    def __init__(self, shared_memory : SharedMemory, replay_buffer : PriorityReplayBuffer, n_repeat_actions : int, n_simulations : int, T : float):
        self.shared_memory = shared_memory
        self.replay_buffer = replay_buffer
        self.n_repeat_actions = n_repeat_actions
        self.n_simulations = n_simulations
        self.T = T

    def start(self):
        with Pool(self.n_simulations) as p:
            self.proc_res = p.map(self._collect, np.random.randint(0, 1e9, self.n_simulations).tolist())

    def add_data_for_replay_buffer(self, observations, action_indices, rewards, values, policies):
        items = self.network.convert_simulation_data_to_buffer_items(observations, action_indices, rewards, values, policies)

        # self.replay_buffer.add_samples(items)

    def get_network_prediction(self, observations, action_indices):
        action, action_idx, value, policy = self.network.get_network_prediction(observations, action_indices, self.T)

        return action, action_idx, value, policy

    def only_grass_visible(self, obs):
        cropped_obs = obs[:84]
        cropped_obs[67:77, 46:50, 1] = 0 # the position of the car
        road1 = cropped_obs[:, :, 1] == 107  # not visited road slide
        road2 = cropped_obs[:, :, 1] == 105  # not visited road slide
        road3 = cropped_obs[:, :, 1] == 102  # visited road slide

        not_grass = np.logical_or(np.logical_or(road1, road2), road3)

        return not not_grass.any()

    def edit_observation(self, obs):
        edited_obs = obs[:, :, 1]/255

        return edited_obs

    def _collect(self, seed):
        observations = []
        action_indices = []
        rewards = []
        values = []
        policies = []

        display = Display(visible=0, size=(800, 600))
        display.start()
        env = gym.make('CarRacing-v0')
        env.seed(seed)

        ret = self.shared_memory.get_net_dict()
        self.network = Network('cpu').eval().to('cpu')
        self.network.load_state_dict(th.load('file.p7'))
        observations.append(self.edit_observation(env.reset()))
        frame_idx = 0

        while frame_idx < 100: # TODO 600
            print('collect')
            action, action_idx, value, policy = self.get_network_prediction(observations, action_indices)
            for _ in range(self.n_repeat_actions):
                obs, rew, done, _ = env.step(action)
                frame_idx += 1

            # TODO, either append every observation, reward and action or just the last ones
            observations.append(self.edit_observation(obs))
            rewards.append(rew)
            action_indices.append(action_idx)
            values.append(value)
            policies.append(policy)

            done = True if self.only_grass_visible(obs) and frame_idx >= 20 else done
            if done:
                break

        self.add_data_for_replay_buffer(observations, action_indices, rewards, values, policies)

        env.close()
        display.stop()

    def join(self):
        print(self.proc_res)
        # for item in self.ps:
        #     self.replay_buffer.add_samples(item)