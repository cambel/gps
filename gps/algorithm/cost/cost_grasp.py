""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_GRASP
from gps.algorithm.cost.cost import Cost

from gps.proto.gps_pb2 import GRIPPING


class CostGrasp(Cost):
    """ Reward if grasping is being done. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_GRASP)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate forward kinematics (end-effector penalties) cost.
        Temporary note: This implements the 'joint' penalty type from
            the matlab code, with the velocity/velocity diff/etc.
            penalties removed. (use CostState instead)
        Args:
            sample: A single sample.
        """
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # Choose target.
        gripping = sample.get(GRIPPING)

        # l = self._hyperparams['grasp_cost'] if gripping[-1] == 1 else self._hyperparams['no_grasp_cost']
        l = self._hyperparams['grasp_cost'] * gripping 

        return l, lx, lu, lxx, luu, lux