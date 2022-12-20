from shared_memory import SharedMemory as SharedMemory
from priority_replay_buffer import PriorityReplayBuffer as PriorityReplayBuffer
from simulator import Simulator as Simulator
from trainer import Trainer as Trainer
import torch as th
import numpy as np
import multiprocessing
import time

th.manual_seed(54321)
np.random.seed(12345)

n_cpus = multiprocessing.cpu_count()
N_SIMULATORS = 2 # TODO n_cpus-1
N_REPEAT_ACTIONS = 2
N_HISTORY = 5
TRAINING_STEPS = 10_000
MEM_SIZE = 10_000
BATCH_SIZE = 32
N_UPDATE_NETWORK = 500
K = 5
N_TRAINING_STEPS = 10
N_WARMUP = 0 # TODO 1000
N_ACTIONS = 5

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    shared_memory = SharedMemory()
    replay_memory = PriorityReplayBuffer(MEM_SIZE, N_HISTORY, K, N_ACTIONS)

    # start the simulations and the trainer
    trainer = Trainer(replay_memory, shared_memory, N_UPDATE_NETWORK, N_WARMUP, device)
    simulator = Simulator(shared_memory, replay_memory, N_REPEAT_ACTIONS, N_SIMULATORS, T=1)

    # the actual training
    begin_time = time.time()
    simulator.start()
    trainer.start()

    while time.time()-begin_time < 5:
        time.sleep(1)

    simulator.stop()
    trainer.stop()