import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IRCR.misc.utils import to_np, soft_update
from IRCR.algorithm import Agent

DEBUG = False

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device,
                 critic_cfg, actor_cfg, discount, init_temperature, alpha_lr,
                 alpha_betas, actor_lr, actor_betas, actor_update_frequency,
                 critic_lr, critic_betas, critic_tau,
                 critic_target_update_frequency, batch_size):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        stochastic_action, _, mean = self.actor.sample(obs, reparameterized=False)
        action = stochastic_action if sample else mean
        assert torch.equal(action, action.clamp(*self.action_range))
        assert action.ndimension() == 2 and action.shape[0] == 1
        return to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs, reparameterized=False)
            assert torch.equal(next_action, next_action.clamp(*self.action_range))
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
        loss_val = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.)
        self.critic_optimizer.step()

        return loss_val

    def update_actor_and_alpha(self, obs):
        action, log_prob, _ = self.actor.sample(obs, reparameterized=True)
        assert torch.equal(action, action.clamp(*self.action_range))
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        loss_val = actor_loss.item()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.)
        self.actor_optimizer.step()

        alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return loss_val

    def _np2pt(self, arr):
        return torch.from_numpy(arr).to(self.device)

    @staticmethod
    def compute_rg(credits, fifo_buffer, mh_buffer):
        r_max = max(mh_buffer.max_credit_val, fifo_buffer.max_credit_val)
        r_min = min(mh_buffer.min_credit_val, fifo_buffer.min_credit_val)
        return (credits - r_min) / (r_max - r_min)

    def update(self, fifo_buffer, mh_buffer, step):

        # return if the buffer has insufficient number of transitions
        samples = fifo_buffer.sample(self.batch_size)
        if samples is None:
            return

        obs = self._np2pt(samples['observations'])
        action = self._np2pt(samples['actions'])
        next_obs = self._np2pt(samples['next_observations'])
        not_done = 1 - self._np2pt(samples['dones']).float()
        credits = self._np2pt(samples['credits'])

        # gather some transitions from the MinHeapBuffer as well
        samples = mh_buffer.sample(self.batch_size)
        obs = torch.cat([obs, self._np2pt(samples['observations'])], dim=0)
        action = torch.cat([action, self._np2pt(samples['actions'])], dim=0)
        next_obs = torch.cat([next_obs, self._np2pt(samples['next_observations'])], dim=0)
        not_done = torch.cat([not_done, 1 - self._np2pt(samples['dones']).float()], dim=0)
        credits = torch.cat([credits, self._np2pt(samples['credits'])], dim=0)

        # calculate guidance reward by normalizing return (line 18, Algorithm 2 in the paper)
        guidance_reward = SACAgent.compute_rg(credits, fifo_buffer, mh_buffer)
        assert torch.equal(guidance_reward, guidance_reward.clamp(min=0., max=1.))

        critic_loss = self.update_critic(obs, action, guidance_reward, next_obs, not_done)

        if step % self.actor_update_frequency == 0:
            actor_loss = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)

        if DEBUG:
            print("Training statistics: Actor-loss = {:.3f}, Critic-loss = {:.3f}".format(actor_loss, critic_loss))
