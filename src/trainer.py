from shared_memory import SharedMemory as SharedMemory
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
from network import Network as Network
import time

class Trainer():
    def __init__(self, replay_buffer : PriorityReplayBuffer, shared_memory : SharedMemory, n_update_network, n_warmup, device):
        self.replay_buffer = replay_buffer
        self.shared_memory = shared_memory
        self. n_update_network= n_update_network
        self.n_warmup = n_warmup

        self.network = Network(device).to(device)
        self.shared_memory.set_net_dict(self.network.state_dict())

    def start_training(self, n_train_steps, batch_size):
        while self.replay_buffer.len() < self.n_warmup:
            time.sleep(1)

        self.network.load_state_dict(self.shared_memory.get_net_dict())
        for i in range(n_train_steps):
            if i % self.n_update_network == 0:
                self.shared_memory.set_net_dict(self.network.state_dict())

            self.network.do_train_step(self.replay_buffer, batch_size)


        self.shared_memory.set_net_dict(self.network.state_dict())