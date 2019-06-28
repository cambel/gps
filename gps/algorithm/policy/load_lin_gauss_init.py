""" Initializations for linear Gaussian controllers.
    Loading trained trajectories

    Author: CBeltran """
import copy
import numpy as np

from gps.algorithm.policy.config import INIT_LG_LQR

def load_traj_lqr(hyperparams):
    traj = np.load(hyperparams["traj_file"])
    traj = traj.flatten()[0]
    traj = fit_matrix(hyperparams, traj)
    return traj

def fit_matrix(hyperparams, traj):
    config = copy.deepcopy(INIT_LG_LQR)
    config.update(hyperparams)

    dX, dU = config['dX'], config['dU']
    T = config['T']
    
    traj.dX = dX
    pad = dX - traj.K.shape[2]
    if pad > 0:
        # print ''
        _fill = np.zeros((T, dU, pad))
        # print "1", _fill.shape
        # print "2", traj.K[:T,:,:].shape
        traj.K = np.c_[traj.K[:T,:,:], _fill]
    
    return traj

        
