import sys
import os
import random
import time
import datetime
import hydra
from collections import deque
import numpy as np
import torch

from IRCR.misc.rollout_storage import RolloutStorage
from IRCR.misc.env_wrappers import MuJoCoEnv

from IRCR.buffers.fifo import FIFOBuffer
from IRCR.buffers.minheap import MinHeapBuffer

def setup(cfg):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)

@hydra.main(config_path='config/mujoco.yaml', strict=True)
def main(cfg):

    print(cfg.pretty())
    setup(cfg)

    wrappers = ['episodic_rewards']
    env = MuJoCoEnv(cfg.env_name, wrappers, cfg.seed)

    cfg.algo.params.obs_dim = env.observation_space.shape[0]
    cfg.algo.params.action_dim = env.action_space.shape[0]
    cfg.algo.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())]

    fifo_buffer = FIFOBuffer(env.observation_space.shape, env.action_space.shape, int(cfg.fifo_buffer_capacity))
    mh_buffer = MinHeapBuffer(env.observation_space.shape, env.action_space.shape, cfg.mh_buffer_capacity)

    rollout_storage = RolloutStorage(env)
    actor_critic = hydra.utils.instantiate(cfg.algo)

    start_time = time.time()
    total_timesteps = 0
    moving_returns = deque(maxlen=30)
    eval_marker = 1

    # some initial exploration to fill up the buffers
    for _ in range(cfg.exploration.num_init_explr):
        path = rollout_storage.collect_rollout(actor_critic, fifo_buffer, mh_buffer, stochastic=True, update_agent=False)
        fifo_buffer.add_paths([path])
        mh_buffer.add_paths([path])
    assert fifo_buffer.min_credit_val < fifo_buffer.max_credit_val, "Need more initial data for min-max normalization. Consider increasing cfg.exploration.num_init_explr!"

    print('Starting with the main training loop... Printing performance after every {} timesteps...'.format(cfg.eval_granularity))
    while total_timesteps < int(cfg.num_train_steps):

        paths = []
        paths.append(rollout_storage.collect_rollout(actor_critic, fifo_buffer, mh_buffer, stochastic=True, update_agent=True))

        # if desired, generate additional experience data
        for _ in range(cfg.exploration.num_periodic_explr):
            paths.append(rollout_storage.collect_rollout(actor_critic, fifo_buffer, mh_buffer, stochastic=True, update_agent=False))

        total_timesteps += sum([path['rewards'].shape[0] for path in paths])
        moving_returns.extend([path['rewards'].sum() for path in paths]) 

        # add the generated paths to the buffers
        fifo_buffer.add_paths(paths)
        mh_buffer.add_paths(paths)

        # print performance
        if total_timesteps >= eval_marker * cfg.eval_granularity:
            duration = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            start_time = time.time()

            print("Duration={}, Total-timesteps:{}, Average returns for last {} episodes={:.2f}".format(
                duration, total_timesteps, len(moving_returns), np.average(moving_returns)))
            sys.stdout.flush()
            eval_marker += 1

if __name__ == "__main__":
    main()
