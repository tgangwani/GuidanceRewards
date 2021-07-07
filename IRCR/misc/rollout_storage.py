from copy import deepcopy
import numpy as np
import torch

class RolloutStorage:
    def __init__(self, env):
        self.env = env

        obs_shape = env.observation_space.shape
        acs_shape = env.action_space.shape
        self.observations = np.zeros((self.env.max_episode_steps, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.env.max_episode_steps, *acs_shape), dtype=np.float32)
        self.rewards = np.zeros((self.env.max_episode_steps, 1), dtype=np.float32)

    def collect_rollout(self, actor_critic, fifo_buff, mh_buffer, stochastic, update_agent):
        """
        Generate a single episode of experience. This function also updates the policy and
        the critic depending on the input parameters
        """

        ob = self.env.reset()
        done = False
        step = 0

        while not done:

            if update_agent:
                actor_critic.update(fifo_buff, mh_buffer, step)

            with torch.no_grad():
                ac = actor_critic.act(ob, sample=stochastic)

            self.observations[step] = ob
            self.actions[step] = ac
            ob, reward, done, info = self.env.step(ac)
            self.rewards[step] = reward
            step += 1

        dones = np.zeros((step, 1), dtype=np.uint8)
        # if the episode ended due to time constraints, we don't mark "done" as True
        if step != self.env.max_episode_steps:
            dones[-1] = 1

        return dict(
                observations=deepcopy(self.observations[:step]),
                final_observation=np.array([ob], dtype=np.float32),
                actions=deepcopy(self.actions[:step]),
                rewards=deepcopy(self.rewards[:step]),
                dones=dones,
                info=info)
