""" Hyperparameters for Box2d Point Mass task with PIGPS."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.point_mass_world import PointMassWorld
# Algorithm
from gps.algorithm.algorithm_mdgps_pilqr import AlgorithmMDGPSPILQR
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
# from gps.algorithm.algorithm_pigps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
# Cost functions
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
# Dynamics
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
# Policy
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network

from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2

from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython

import yaml

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/experiments/box2d/pointmass/mdgpspilqr/'

stream = open(EXP_DIR + 'params.yaml', 'r')
params = yaml.load(stream)

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2
}

ITERATIONS = params['iterations'] # Typically 10.

common = {
    'experiment_name': params['exp_global_name'],
    'experiment_dir': EXP_DIR,
    'data_dir': EXP_DIR + 'data_files/',
    'data_files_dir': EXP_DIR + 'data_files/' + params['exp_name'] +'/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': params['conditions'],
    # 'log_only_policy': True
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([5, 20, 0]),
    "world" : PointMassWorld,
    'render' : False,
    'x0': [np.array([0, 5, 0, 0, 0, 0]),
           np.array([0, 10, 0, 0, 0, 0]),
           np.array([10, 5, 0, 0, 0, 0]),
           np.array([10, 10, 0, 0, 0, 0]),
        ],
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': params['steps'],
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise_var': params['noise_var'],
}

# algorithm = {
#     'type': AlgorithmPIGPS,
#     'conditions': common['conditions'],
#     'policy_sample_mode': 'replace',
#     'sample_on_policy': True,
#     'iterations': ITERATIONS,
# }

algorithm = {
    'type': AlgorithmMDGPSPILQR,
    # 'step_rule': 'const',
    'conditions': common['conditions'],
    'iterations': ITERATIONS,
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
    'kl_step': np.linspace(0.9, 0.1, 100),
    'max_step_mult': np.linspace(10.0, 5.0, 100),
    'min_step_mult': np.linspace(0.01, 0.5, 100),
    'max_mult': np.linspace(5.0, 2.0, 100),
    'min_mult': np.linspace(0.1, 0.5, 100),
}

# algorithm = {
#     'type': AlgorithmTrajOptPI2,
#     'conditions': common['conditions'],
#     'iterations': ITERATIONS,
# }

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': params['init_var'],
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([5e-5, 5e-5])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
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
    'type': TrajOptPILQR,
    'covariance_damping': params['traj_opt']['covariance_damping'],
    'kl_threshold': params['traj_opt']['kl_threshold'],
    'min_temperature': params['traj_opt']['min_temperature'],
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': SENSOR_DIMS,
    },
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': params['policy']['iterations'],
    'network_model': tf_network,
    'use_gpu': True
}
if not os.path.exists(algorithm['policy_opt']['weights_file_prefix']):
    os.makedirs(algorithm['policy_opt']['weights_file_prefix'])

algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

# algorithm['policy_opt'] = {}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': params['samples'],
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': params['gui'],
    'algorithm': algorithm,
    'dQ': algorithm['init_traj_distr']['dQ'],
}

common['info'] = generate_experiment_info(config)
