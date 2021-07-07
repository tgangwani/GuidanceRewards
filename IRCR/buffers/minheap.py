from heapq import heappush, heappushpop
from copy import deepcopy
import numpy as np

class MinHeapBuffer:
    def __init__(self, obs_shape, acs_shape, max_trajs):

        self.obs_shape = obs_shape
        self.acs_shape = acs_shape
        self.heap = []
        self.max_trajs = max_trajs
        self.num_trajs = 0
        self.traj_data = dict()

        # the {min, max} credit value assigned to a transition in the MinHeapBuffer
        self.max_credit_val = None
        self.min_credit_val = None

        self.obs = None
        self.next_obs = None
        self.acs = None
        self.dones = None
        self.rewards = None
        self.credits = None

    def __len__(self):
        return self.obs.shape[0] if self.obs is not None else 0

    def add_paths(self, paths):
        updated = False

        for path in paths:
            priority = np.asscalar(sum(path['rewards']))

            if self.num_trajs < self.max_trajs:
                heappush(self.heap, [priority, self.num_trajs])
                loc = self.num_trajs
                self.num_trajs += 1
            else:
                min_priority, loc = self.heap[0]
                if priority < min_priority:
                    continue

                # replace min-priority entry
                heappushpop(self.heap, [priority, loc])

            # free memory
            if loc in self.traj_data.keys():
                del self.traj_data[loc]

            # update data
            self.traj_data[loc] = deepcopy(path)
            updated = True

        if updated:
            self.obs = np.empty((0, *self.obs_shape), dtype=np.float32)
            self.next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)
            self.acs = np.empty((0, *self.acs_shape), dtype=np.float32)
            self.dones = np.empty((0, 1), dtype=np.uint8)
            self.rewards = np.empty((0, 1), dtype=np.float32)
            self.credits = np.empty((0, 1), dtype=np.float32)

            for path in self.traj_data.values():
                l = path['observations'].shape[0]
                self.obs = np.append(self.obs, path['observations'], axis=0)
                self.next_obs = np.append(self.next_obs, np.concatenate([path['observations'][1:], path['final_observation']], axis=0), axis=0)
                self.acs = np.append(self.acs, path['actions'], axis=0)
                self.dones = np.append(self.dones, path['dones'], axis=0)
                self.rewards = np.append(self.rewards, path['rewards'], axis=0)
                self.credits = np.append(self.credits, path['rewards'].mean(axis=0, keepdims=True).repeat(l, axis=0), axis=0)

            # update credit values
            self.min_credit_val = self.credits.min().item()
            self.max_credit_val = self.credits.max().item()

    def sample(self, batch_size, repeat=True, weighted=False):
        if len(self) == 0:
            return None

        if len(self) < batch_size:
            if not repeat:
                return None
            inds = np.random.choice(np.arange(len(self)), size=batch_size, replace=True)
        else:
            p = None
            if weighted:
                p = self.credits[:len(self)]
                p = p / np.sum(p)
            inds = np.random.choice(np.arange(len(self)), size=batch_size, replace=False, p=p)

        data = dict(
                observations=self.obs[inds],
                next_observations=self.next_obs[inds],
                actions=self.acs[inds],
                rewards=self.rewards[inds],
                credits=self.credits[inds],
                dones=self.dones[inds]
                )

        return data
