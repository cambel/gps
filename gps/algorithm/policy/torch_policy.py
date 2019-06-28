import pickle
import os
import uuid

import numpy as np
import torch

from gps.algorithm.policy.policy import Policy


class TorchPolicy(Policy):
    """
    A neural network policy implemented in pytorch. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        net: Network module.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, net, var, dev):
        Policy.__init__(self)
        self.net = net
        self.dev = dev
        self.chol_pol_covar = np.diag(np.sqrt(var))

        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # CB: returns a prediction?
        # pred = weights(obs) + bias

        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias
        obs = torch.tensor(obs, dtype=torch.float32, device=self.dev)

        action_mean = None
        with torch.no_grad():
            action_mean = self.net(obs).cpu().numpy() #forward
        
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)

        return u
