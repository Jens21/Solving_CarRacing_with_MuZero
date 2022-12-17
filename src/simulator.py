from shared_memory import SharedMemory as SharedMemory
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
import concurrent.futures
import gym
from pyvirtualdisplay import Display
import numpy as np
from multiprocessing import Process
from network import Network as Network

class Simulator():
    processes = []

    def __init__(self, shared_memory : SharedMemory, replay_buffer : PriorityReplayBuffer, n_repeat_actions : int, n_simulations : int):
        self.shared_memory = shared_memory
        self.replay_buffer = replay_buffer
        self.n_repeat_actions = n_repeat_actions
        self.n_simulations = n_simulations
        self.network = Network()

    def collect_samples(self):
        for _ in range(self.n_simulations):
            p = Process(target= self._collect, args=(np.random.randint(0, 1e9),))
            p.start()
            self.processes.append(p)

    def add_data_to_replay_buffer(self, observations, action_indices, rewards):
        items = self.network.convert_simulation_data_to_buffer_items(observations, action_indices, rewards)

        for item in items:
            self.replay_buffer.add_sample(item)

    def get_network_prediction(self, current_obs, observations, action_indices):
        action, action_idx = self.network.get_network_prediction(current_obs, observations, action_indices)

        return action, action_idx

    def only_grass_visible(self, current_obs):
        current_obs = current_obs[:84]
        current_obs[67:77, 46:50, 1] = 204 # the position of the car
        not_grass = np.logical_or(np.logical_or(current_obs[:, :, 1] == 107, current_obs[:, :, 1] == 105), current_obs[:, :, 1] == 102)

        return not not_grass.any()

    def _collect(self, seed):
        observations = []
        action_indices = []
        rewards = []

        display = Display(visible=0, size=(800, 600))
        display.start()
        env = gym.make('CarRacing-v0')
        env.seed(seed)

        for round_idx in range(1):
            self.network.load_state_dict(self.shared_memory.get_net_dict())
            current_obs = env.reset()
            frame_idx = 0

            while frame_idx < 600:
                action, action_idx = self.get_network_prediction(current_obs, observations, action_indices)
                for _ in range(self.n_repeat_actions):
                    obs, rew, done, _ = env.step(action)
                    frame_idx += 1

                observations.append(current_obs)
                rewards.append(rew)
                action_indices.append(action_idx)

                current_obs = obs

                done = True if self.only_grass_visible(current_obs) else done
                if done:
                    break

            self.add_data_to_replay_buffer(observations, action_indices, rewards)

        env.close()
        display.stop()

    def join(self):
        for p in self.processes:
            p.join()