""" Hyperparameters for UR3 """
# two end effector points

from __future__ import division

from datetime import datetime
import os.path
import os

import numpy as np

from pyquaternion import Quaternion

import yaml 

from gps import __file__ as gps_filepath
# Agent
from gps.agent.ur.agent_ur import AgentUR
from gps.agent.ur.constants import * 
from gps.agent.utils.model import Model
# Algorithm
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
# Cost functions
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_ft import CostFT
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
# Dynamics
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# Policy
## Init policy
from gps.algorithm.policy.lin_gauss_init import init_lqr
# Policy representation
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS, \
    JOINT_VELOCITIES, ACTION, END_EFFECTOR_POINT_VELOCITIES
from gps.algorithm.cost.cost_utils import RAMP_LINEAR
from gps.gui.config import generate_experiment_info
from gps.utility.general_utils import get_ee_points

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/experiments/ur3/lqr/6d_joints/'

stream = open(EXP_DIR + 'params.yaml', 'r')
params = yaml.load(stream)

EE_POINTS = np.array([[0.0, 0.0, -0.0], [0.0, 0.05, 0.00], [0.0, -0.05, 0.00]])
NUM_EE_POINTS = EE_POINTS.shape[0]

#Hyperparamters to be tuned for optimizing policy learning on the specific robot
UR_GAINS = np.array([0.147, 0.407, 0.118, 0.008, 0.003, 0.001])
# UR_GAINS = np.ones(6)

DOFs = 6 # Degrees of freedom

SENSOR_DIMS = {
    JOINT_ANGLES: DOFs,                    # Read from robot
    JOINT_VELOCITIES: DOFs,                # Actions
    END_EFFECTOR_POINTS: 3 * NUM_EE_POINTS,
    END_EFFECTOR_POINT_VELOCITIES: 3 * NUM_EE_POINTS,
    ACTION: DOFs,
}

# Set to identity unless you want the goal to have a certain orientation.
EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Set the number of seconds per step of a sample.
FRECUENCY = params['frecuency'] # hz CB: affects duration of episode but not speed
# Set the number of timesteps per sample.
STEP_COUNT = params['steps'] # Typically 100.
# Set the number of samples per condition.
SAMPLE_COUNT = params['samples'] # Typically 5.
# set the number of conditions per iteration.
CONDITIONS = params['conditions'] # Typically 2 for Caffe and 1 for LQR.
# Set the number of trajectory iterations to collect.
ITERATIONS = params['iterations'] # Typically 10.
# How many iterations with verbose active
VERBOSE = 5
# 'SLOWNESS' is how far in the future (in seconds) position control extrapolates
# when it publishs actions for robot movement.  1.0-10.0 is fine for simulation.
# CBeltran: Affects speed of movement but not overall duration of episode
SLOWNESS = params['slowness']
# 'RESET_SLOWNESS' is how long (in seconds) we tell the robot to take when
# returning to its start configuration.
RESET_SLOWNESS = 2

x0s = []
ee_tgts = []
reset_conditions = []

common = {
    'experiment_name': params['exp_global_name'],
    'experiment_dir': EXP_DIR,
    'data_dir': EXP_DIR + 'data_files/',
    'data_files_dir': EXP_DIR + 'data_files/' + params['exp_name'] +'/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
}

# Target pose #
eeposes = [
    [0.0131,  0.4019,  0.3026],  #  q = [ 1.3057, -0.9192,  0.5927,  0.7478,  2.0003, -0.    ]
    [-0.1042, 0.3904, 0.16],  #  insert**
    [-0.1042, 0.3904, 0.16],  #  insert**
    [-0.1042, 0.3904, 0.16],  #  insert**
]

eerot = [
    [0.8855,  0.4149, -0.1352,  0.1595],  # center?
    [0.6918, 0.72209, 0.00201, -0.00091],  # center?
    [0.6918, 0.72209, 0.00201, -0.00091],  # center?
    [0.6918, 0.72209, 0.00201, -0.00091],  # center?
]

init_q = [
    [ 1.7263, -0.9193,  0.3836,  0.586 ,  1.7603, -0.    ],
    [ 1.7263, -0.9193,  0.3836,  0.586 ,  1.7603, -0.    ],
    [ 1.7263, -0.9193,  0.3836,  0.586 ,  1.7603, -0.    ],
    [ 1.7263, -0.9193,  0.3836,  0.586 ,  1.7603, -0.    ],
]

# Set up each condition.
for i in xrange(common['conditions']):
    
    ee_pos_tgt = np.array(eeposes[i]).reshape(1, 3)
    ee_rot_quat = Quaternion(np.roll(eerot[i], 1))  # shift one position so that we have [w,x,y,z]
    ee_rot_tgt = ee_rot_quat.rotation_matrix.reshape((3, 3))

    state_space = sum(SENSOR_DIMS.values()) - SENSOR_DIMS[ACTION]
    print "state_space", state_space

    # Initialized to start position and inital velocities are 0
    x0 = np.zeros(state_space)
    x0[:DOFs] = init_q[i]

    # Initialize target end effector position
    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T).reshape((1, -1))

    reset_condition = {
        JOINT_ANGLES: init_q[i],
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentUR,
    'dt': 1/FRECUENCY,
    'dU': SENSOR_DIMS[ACTION],
    'conditions': common['conditions'],
    'T': STEP_COUNT,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'num_ee_points': NUM_EE_POINTS,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'slowness': SLOWNESS,
    'reset_slowness': RESET_SLOWNESS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
    'smooth_noise_var': params['noise_var'],
    'noise_var_reduction': True,
    'image': False,
    'eep_diff': True,
    'ft_sensor': False,
    'data_dir': common['data_dir'],
    'exp_name': params['exp_name'],
    'experiment_name': common['experiment_name'],
    'limits': np.ones(DOFs)*np.pi/128,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': ITERATIONS,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  params['init']['gains'] / UR_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': params['init']['var'], 
    'stiffness': params['init']['stiffness'],
    'stiffness_vel': params['init']['stiffness_vel'],
    'final_weight': params['init']['final_weight'],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': params['action_cost'] / UR_GAINS,
}

# This cost function takes into account the distance between the end effector's
# current and target positions, weighted in a linearly increasing fassion
# as the number of trials grows from 0 to T-1.
dist_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * NUM_EE_POINTS) if agent['eep_diff'] else ee_tgts[i],
    'wp': np.ones(3 * NUM_EE_POINTS), 
    'wp_final_multiplier': params['dist_cost1']['final_weight'],  # Weight multiplier.
    'l1': params['dist_cost1']['l1'],
    'l2': params['dist_cost1']['l2'],
    'ramp_option': RAMP_LINEAR,
}

# Combines the cost functions in 'costs' to produce a single cost function
algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, dist_cost1],
    'weights': [1.0, 1.0, 1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': params['dynamics']['regularization'],
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': params['dynamics']['max_clusters'],
        'min_samples_per_cluster': 40,
        'max_samples': params['dynamics']['max_samples'],
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': VERBOSE,
    'agent': agent,
    'gui_on': params['gui'],
    'algorithm': algorithm,
    'num_samples': SAMPLE_COUNT,
    'exp_name': params['exp_name'],
}

common['info'] = generate_experiment_info(config)
