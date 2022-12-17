import gym
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
from shared_memory import SharedMemory as SharedMemory
from network import Network as Network
import threading
import time
from pyvirtualdisplay import Display

class Simulator():
    observations = []
    rewards = []
    action_indices = []
    do_sample = True

    def __init__(self, shared_memory : SharedMemory, replay_buffer: PriorityReplayBuffer, n_repeat_actions : int, n_history : int):
        self.shared_memory = shared_memory
        self.replay_buffer = replay_buffer
        self.n_repeat_actions = n_repeat_actions
        self.n_history = n_history

        self.thread = threading.Thread(target=self.sample_observations)

        self.display = Display(visible=0, size=(800, 600))
        self.display.start()

        self.env = gym.make('CarRacing-v0')

        self.network = Network()

    def start(self):
        self.thread.start()

    def sample_observations(self):
        env = self.env

        while self.do_sample:
            self.network.load_state_dict(self.shared_memory.get_net_dict())
            frame_idx = 0
            obs = env.reset()

            while frame_idx < 600:
                inp = self.network.convert_simulation_data_to_network_input(self.observations, self.action_indices)
                action, action_idx = self.network(inp)

                for _ in range(self.n_repeat_actions):
                    next_obs, r, _, _ = env.step(action.tolist())

                self.observations.append(obs)
                self.action_indices.append(action_idx)
                self.rewards.append(r)

                obs = next_obs
                frame_idx += 1

            items = self.network.convert_simulation_data_to_buffer_items(self.observations, self.action_indices, self.rewards)
            for item in items:
                self.replay_buffer.add_sample(item)

        env.close()

    def shutdown(self):
        self.do_sample = False

    def join(self):
        self.thread.join()
        self.display.stop()

if __name__ == '__main__':
    mem_size = 100_000
    n_history = 4
    n_repeat_actions = 2

    replay_buffer = PriorityReplayBuffer(mem_size)
    shared_memory = SharedMemory()

    n_envs = 2
    simulators = [Simulator(shared_memory, replay_buffer, n_repeat_actions, n_history) for _ in range(n_envs)]
    shared_memory.set_net_dict(simulators[0].network.state_dict())

    for sim in simulators:
        sim.start()

    time.sleep(1)

    for sim in simulators:
        sim.shutdown()
    for sim in simulators:
        sim.join()