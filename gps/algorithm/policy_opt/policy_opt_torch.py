""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TORCH
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy.torch_policy import TorchPolicy

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader)

LOGGER = logging.getLogger(__name__)

def preprocess(x, y, z, dev):
    return x.to(dev), y.to(dev), z.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func, dev):
        self.dl = dl
        self.func = func
        self.dev = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b, dev=self.dev))

###############################################
# compute loss for batch
###############################################
def loss_batch(model, loss_func, xb, yb, prc, opt=None):
    loss = loss_func(model(xb), yb, prc)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = torch.unsqueeze(vector, 1)
    mult_result = torch.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = torch.squeeze(mult_result, 1)
    return squeezed_result

def euclidean_loss_layer(a, b, precision, batch_size=25):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = torch.tensor(2* batch_size)
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = torch.sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor

class PolicyOptTorch(PolicyOpt):    
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TORCH)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.batch_size = self._hyperparams['batch_size']
        self.lr = self._hyperparams['lr']

        if self._hyperparams['use_gpu']:
            self.dev = torch.device("cuda") \
                  if torch.cuda.is_available() \
                  else torch.device("cpu")
        else:
            self.dev = torch.device("cpu")

        self._init_network()
        self.var = self._hyperparams['init_var'] * np.ones(dU)

        self.policy = TorchPolicy(self.model, self.var, self.dev)

    def _init_network(self):
        self.model = Net(self._dO, self._dU)
        self.model.to(self.dev)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.loss_func = euclidean_loss_layer

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        if self.policy.scale is None or self.policy.bias is None:
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(1.0 / np.maximum(np.std(obs, axis=0),
                                                         1e-3))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
        obs = obs.dot(self.policy.scale) + self.policy.bias

        epochs = self._hyperparams['iterations']

        obs = torch.tensor(obs, dtype=torch.float32, device=self.dev)
        tgt_mu = torch.tensor(tgt_mu, dtype=torch.float32, device=self.dev)
        tgt_prc = torch.tensor(tgt_prc, dtype=torch.float32, device=self.dev)

        # self.obs = np.float32(self.obs)
        # self.tgt_mu = np.float32(self.tgt_mu)
        # self.tgt_prc = np.float32(self.tgt_prc)

        train_ds = DynamicsDataset(obs, tgt_mu, tgt_prc)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        # train_dl = WrappedDataLoader(train_dl, preprocess, self.dev)

        
        val_loss = 0
        for epoch in range(epochs):
            self.model.train()
            losses, nums = list(zip(
                    *[loss_batch(self.model, self.loss_func, xb, yb, prc) for xb, yb, prc in train_dl]
                ))
            val_loss += np.sum(np.multiply(losses, nums)) / np.sum(nums)

            if (epoch+1) % 200 == 0:
                print((epoch+1, val_loss/ 200.))
                val_loss = 0

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if self.policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, :] = obs[n, :, :].dot(self.policy.scale) + \
                        self.policy.bias

        output = np.zeros((N, T, dU))
        
        obs = torch.tensor(obs, dtype=torch.float32, device=self.dev)

        self.model.eval()
        for i in range(N):
            for t in range(T):
                with torch.no_grad():
                    # Feed in data.
                    
                    predict = self.model(obs[i, t])
                            

                    # Assume that the first output blob is what we want.
                    output[i, t, :] = \
                            predict.cpu().numpy()

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

# # init network (Model)
# # net.load_state_dict(torch.load('net_params.pkl'))


#     # For pickling.
#     def __getstate__(self):
#         torch.save(net.state_dict(), 'net_params.pkl')
#         return {
#             'hyperparams': self._hyperparams,
#             'dO': self._dO,
#             'dU': self._dU,
#             'scale': self.policy.scale,
#             'bias': self.policy.bias,
#             'caffe_iter': self.caffe_iter,
#             'var': self.var,
#         }

#     # For unpickling.
#     def __setstate__(self, state):
#         self.__init__(state['hyperparams'], state['dO'], state['dU'])
#         self.policy.scale = state['scale']
#         self.policy.bias = state['bias']
#         self.caffe_iter = state['caffe_iter']
#         self.var = state['var']
#         self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))
#         self.solver.restore(
#             self._hyperparams['weights_file_prefix'] + '_iter_' +
#             str(self.caffe_iter) + '.solverstate'
#         )
#         self.policy.net.copy_from(
#             self._hyperparams['weights_file_prefix'] + '_iter_' +
#             str(self.caffe_iter) + '.caffemodel'
#         )


###############################################
# Define a Convolution Neural Network
###############################################
class Net(nn.Module):
    def __init__(self, dX, dU):
        super(Net, self).__init__()
        # 1 input vector (state x action)
        self.fc1 = nn.Linear(dX, 60)
        self.fc2 = nn.Linear(60, 40)
        self.fc3 = nn.Linear(40, dU)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DynamicsDataset(Dataset):
    """Reinforcement learning dataset
       input (state, action)
       target (next_state)
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        # input, precision, action(label)
        return self.a[idx], self.b[idx], self.c[idx]