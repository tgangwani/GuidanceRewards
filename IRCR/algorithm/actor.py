import torch
from torch import nn
from torch.distributions import Normal
from IRCR.misc.utils import mlp, weight_init

EPS = 1e-6

class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_range, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

        self.apply(weight_init)

        self.action_scale = torch.tensor((action_range[1] - action_range[0]) / 2.)
        self.action_shift = torch.tensor((action_range[1] + action_range[0]) / 2.)
        self.use_stable_lp = True

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_shift = self.action_shift.to(device)
        return super().to(device)

    def forward(self, obs):
        mean, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)

        return mean, log_std

    def sample(self, obs, reparameterized):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        arctanh_action = normal.rsample() if reparameterized else normal.sample()
        action = torch.tanh(arctanh_action)
        action_scaled_shifted = action * self.action_scale + self.action_shift
        mean = torch.tanh(mean) * self.action_scale + self.action_shift

        # log_prob with tanh `flow` and scale-shift `flow`
        log_prob = normal.log_prob(arctanh_action)

        if self.use_stable_lp:
            # use a numerically stable formula for log(1 - tanh(x)^2), adapted from
            # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
            # formula => log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
            f = - arctanh_action - torch.nn.functional.softplus(-2. * arctanh_action)
            log_prob -= (2*f + torch.log(4. * self.action_scale))
        else:
            log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + EPS)

        log_prob = log_prob.sum(1, keepdim=True)
        return action_scaled_shifted, log_prob, mean
