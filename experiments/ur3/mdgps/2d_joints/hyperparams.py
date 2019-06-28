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
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
# Cost functions
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_ft import CostFT
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
# Dynamics
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
# Policy
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
## Init policy
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.load_lin_gauss_init import load_traj_lqr
# Policy representation
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS, \
    JOINT_VELOCITIES, ACTION, END_EFFECTOR_POINT_VELOCITIES, \
    TORQUE
from gps.algorithm.cost.cost_utils import RAMP_LINEAR
from gps.gui.config import generate_experiment_info
from gps.utility.general_utils import get_ee_points

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/experiments/ur3/mdgps/2d_joints/'

stream = open(EXP_DIR + 'params.yaml', 'r')
params = yaml.load(stream)

EE_POINTS = np.array([[0.0, 0.0, 0.0]])
NUM_EE_POINTS = 2

#Hyperparamters to be tuned for optimizing policy learning on the specific robot
UR_GAINS = np.array([1.0, 1.0])

DOFs = 2 # Degrees of freedom

SENSOR_DIMS = {
    JOINT_ANGLES: DOFs,                    # Read from robot
    JOINT_VELOCITIES: DOFs,                # Actions
    END_EFFECTOR_POINTS: NUM_EE_POINTS,
    END_EFFECTOR_POINT_VELOCITIES: NUM_EE_POINTS,
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

## X dim is fixed
# Target pose #
eeposes = [
    [-0.13101034,  0.37818616,  0.50852045], #up
    [-0.13100248,  0.38417792,  0.49198265], #down
    [-0.13100047,  0.40919604,  0.49196294], #down-right
    [-0.13100211,  0.40921119,  0.51694602], #up-right
]

init_q = [
    np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0]),  #1
    np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0]),  #1
    np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0]),  #1
    np.deg2rad([90.0, -30.0, -60.0, 0.0, 90.0, 0.0]),  #1
    [1.58132096e+00, -3.73642151e-02, -4.66492183e-01,  1.73612724e-04,
        1.57063075e+00, -5.05136730e-05],
]

indices = [1, 2]
task_dims = [1, 2] # y and z

# Set up each condition.
for i in xrange(common['conditions']):
   
    state_space = sum(SENSOR_DIMS.values()) - SENSOR_DIMS[ACTION]
    print "state_space", state_space

    # Initialized to start position and inital velocities are 0
    x0 = np.zeros(state_space)
    x0[:DOFs] = [init_q[i][index] for index in indices]

    # Initialize target end effector position
    ee_tgt = [eeposes[i][index] for index in indices]

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
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'slowness': SLOWNESS,
    'reset_slowness': RESET_SLOWNESS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
                      END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
    'smooth_noise_var': params['noise_var'],
    'noise_var_reduction': False,
    'image': False,
    'eep_diff': True,
    'ft_sensor': False,
    'data_dir': common['data_dir'],
    'exp_name': params['exp_name'],
    'experiment_name': common['experiment_name'],
    'indices': indices,
    'dimensions': task_dims,
    'limits': [np.pi/128, np.pi/128],
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': ITERATIONS,
    'kl_step': params['mdgps']['kl_step'],
    'min_step_mult': params['mdgps']['min_step_mult'],
    'max_step_mult': params['mdgps']['max_step_mult'],
    'policy_sample_mode': 'replace',
    # 'sample_on_policy': True,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'traj_file': [EXP_DIR + "traj0.npy", EXP_DIR + "traj1.npy"],
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
costs = {
    'type': CostState,
    'l1': params['cost']['l1'],
    'l2': params['cost']['l2'],
    'alpha': params['cost']['alpha'],
    'wp_final_multiplier':
    params['cost']['final_w'],  # Weight multiplier on final time step.
    'data_types': {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(2),
            'target_state': np.zeros(2),
        }
    },
}


# Combines the cost functions in 'costs' to produce a single cost function
algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, costs],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': agent['obs_include'],
        'sensor_dims': SENSOR_DIMS,
        'n_layers': 2,
    },
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': params['policy']['iterations'],
    'network_model': tf_network,
    'use_gpu': True
}
if not os.path.exists(algorithm['policy_opt']['weights_file_prefix']):
    os.makedirs(algorithm['policy_opt']['weights_file_prefix'])


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}


config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': VERBOSE,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': params['gui'],
    'algorithm': algorithm,
    'num_samples': SAMPLE_COUNT,
    'exp_name': params['exp_name'],
}

common['info'] = generate_experiment_info(config)
