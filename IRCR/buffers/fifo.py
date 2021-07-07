import numpy as np

class FIFOBuffer:
    def __init__(self, obs_shape, acs_shape, max_steps):

        self.obs_shape = obs_shape
        self.acs_shape = acs_shape
        self.max_steps = max_steps

        self.obs = np.zeros((self.max_steps, *self.obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.max_steps, *self.obs_shape), dtype=np.float32)
        self.acs = np.zeros((self.max_steps, *self.acs_shape), dtype=np.float32)

        self.dones = np.zeros((self.max_steps, 1), dtype=np.uint8)
        self.rewards = np.zeros((self.max_steps, 1), dtype=np.float32)
        self.credits = np.zeros((self.max_steps, 1), dtype=np.float32)

        # the {min, max} credit value assigned to a transition in the FIFOBuffer
        self.min_credit_val = None
        self.max_credit_val = None

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def add_paths(self, paths):
        all_obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        all_next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        all_acs = np.empty((0, *self.acs_shape), dtype=np.float32)
        all_dones = np.empty((0, 1), dtype=np.uint8)
        all_rews = np.empty((0, 1), dtype=np.float32)
        all_cr = np.empty((0, 1), dtype=np.float32)

        for path in paths:
            l = path['observations'].shape[0]
            all_obs = np.append(all_obs, path['observations'], axis=0)
            all_next_obs = np.append(all_next_obs, np.concatenate([path['observations'][1:], path['final_observation']], axis=0), axis=0)
            all_acs = np.append(all_acs, path['actions'], axis=0)
            all_dones = np.append(all_dones, path['dones'], axis=0)
            all_rews = np.append(all_rews, path['rewards'], axis=0)
            all_cr = np.append(all_cr, path['rewards'].mean(axis=0, keepdims=True).repeat(l, axis=0), axis=0)

        nentries = all_obs.shape[0]
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs = np.roll(self.obs, rollover, axis=0)
            self.next_obs = np.roll(self.next_obs, rollover, axis=0)
            self.acs = np.roll(self.acs, rollover, axis=0)
            self.dones = np.roll(self.dones, rollover, axis=0)
            self.rewards = np.roll(self.rewards, rollover, axis=0)
            self.credits = np.roll(self.credits, rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_steps

        self.obs[self.curr_i:self.curr_i + nentries] = all_obs
        self.next_obs[self.curr_i:self.curr_i + nentries] = all_next_obs
        self.acs[self.curr_i:self.curr_i + nentries] = all_acs
        self.dones[self.curr_i:self.curr_i + nentries] = all_dones
        self.rewards[self.curr_i:self.curr_i + nentries] = all_rews
        self.credits[self.curr_i:self.curr_i + nentries] = all_cr

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

        # update credit values
        self.min_credit_val = self.credits[:len(self)].min().item()
        self.max_credit_val = self.credits[:len(self)].max().item()

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None

        inds = np.random.choice(np.arange(len(self)), size=batch_size, replace=False)
        data = dict(
                observations=self.obs[inds],
                next_observations=self.next_obs[inds],
                actions=self.acs[inds],
                rewards=self.rewards[inds],
                credits=self.credits[inds],
                dones=self.dones[inds]
                )

        return data
