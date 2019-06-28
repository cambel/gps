""" Default configuration and hyperparameters for agent objects. """
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

# Agent
AGENT = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
}

AGENT_UR_ROS = {
    'trial_timeout': 4,
    'ft_sensor': False,
    'use_gripper': False,
    'init_seq': None,
    'ext_ee': [0,0,0,1,0,0,0],
    'control': 'joints',
    'space_type': 'joint_space',
    'real_robot': False,
    'robot_urdf': 'ur3_robot',
    'num_ee_points': 1,
}

AGENT_BAXTER = {
    'joint_names': ['right_s0','right_s1','right_e0','right_e1','right_w0','right_w1','right_w2'], 
    'controller': 'torque',
    'image': False,
    'gripper': False,
    'hz': 20,
    'target_offset': [[0.0, 0.0, 0.05], [0., 0.04, -0.05], [0.0, -0.04, -0.05]],
    'eep_diff': False, # if True then End effector points will be saved as the difference btw gripper and object
    'grasp_threashold': 2.5,
    'gripper': True,
}

# AgentMuJoCo
AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
    'meta_include': []
}

AGENT_BOX2D = {
    'render': True,
}
