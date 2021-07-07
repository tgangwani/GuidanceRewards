import gym

class EpisodicEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        print('== Episodic Rewards Wrapper ==')
        gym.Wrapper.__init__(self, env)
        self.total_episode_reward = 0.

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.total_episode_reward += reward

        if done:
            reward = self.total_episode_reward
            self.total_episode_reward = 0.  # reset
        else:
            reward = 0.

        return observation, reward, done, info

class MuJoCoEnv:
    def __init__(self, env_name, wrappers, seed):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.max_episode_steps = self.env._max_episode_steps
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if 'episodic_rewards' in wrappers:
            self.env = EpisodicEnvWrapper(self.env)
        else: assert len(wrappers) == 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
