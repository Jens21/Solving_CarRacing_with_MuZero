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

N_SIMULATORS = 2
N_REPEAT_ACTIONS = 2
N_HISTORY = 4
TRAINING_STEPS = 100_000
MEM_SIZE = 100_000
BATCH_SIZE = 32
N_UPDATE_NETWORK = 500

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    shared_memory = SharedMemory()
    replay_memory = PriorityReplayBuffer(MEM_SIZE)

    # start the simulations and the trainer
    n_cpus = multiprocessing.cpu_count()
    trainer = Trainer(replay_memory, shared_memory, N_UPDATE_NETWORK, device)
    simulator = Simulator(shared_memory, replay_memory, N_REPEAT_ACTIONS, n_cpus-1)

    # the actual training
    steps = 1 # TODO, should be something like 100_000/23/600 for 23 processors
    begin_time = time.time()
    for _ in range(steps):
        simulator.collect_samples()
        trainer.start_training(n_cpus * 600, BATCH_SIZE)
        simulator.join()

    print(time.time()-begin_time)