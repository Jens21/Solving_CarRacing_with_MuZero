from shared_memory import SharedMemory as SharedMemory
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
from network import Network as Network
import time

class Trainer():
    def __init__(self, replay_buffer : PriorityReplayBuffer, shared_memory : SharedMemory, n_update_network, device):
        self.replay_buffer = replay_buffer
        self.shared_memory = shared_memory
        self. n_update_network= n_update_network

        self.network = Network().to(device)
        self.shared_memory.set_net_dict(self.network.state_dict())

    def start_training(self, n_train_steps, batch_size):
        pass