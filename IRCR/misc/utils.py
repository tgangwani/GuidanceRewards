import numpy as np
import torch.nn as nn

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def to_np(t):
    if t is None:
        return None
    if t.nelement() == 0:
        return np.array([])
    return t.cpu().detach().numpy()

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, activation="relu", output_mod=None):
    if activation == "relu":
        actv = nn.ReLU
    elif activation == "tanh":
        actv = nn.Tanh
    else:
        raise ValueError("Unsupported MLP activation")

    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), actv()]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), actv()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
