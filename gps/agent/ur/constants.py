""" Common constants for URROS projects """
import numpy as np

# Topics for the robot publisher and subscriber.
JOINT_PUBLISHER_REAL  = '/vel_based_pos_traj_controller/command'
JOINT_PUBLISHER_SIM  = '/arm_controller/command'
JOINT_SUBSCRIBER = '/arm_controller/state'
JOINT_STATE_SUBSCRIBER = 'joint_states'
FT_SUBSCRIBER_REAL = '/wrench'
FT_SUBSCRIBER_SIM = '/ft_sensor/raw'

# 'SLOWNESS' is how far in the future (in seconds) position control extrapolates
# when it publishs actions for robot movement.  1.0-10.0 is fine for simulation.
SLOWNESS = 10.
# 'RESET_SLOWNESS' is how long (in seconds) we tell the robot to take when
# returning to its start configuration.
RESET_SLOWNESS = 1

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

# Set constants for links
BASE_LINK = 'base_link'
SHOULDER_LINK = 'shoulder_link'
UPPER_ARM_LINK = 'upper_arm_link'
FOREARM_LINK = 'forearm_link'
WRIST_1_LINK = 'wrist_1_link'
WRIST_2_LINK = 'wrist_2_link'
WRIST_3_LINK = 'wrist_3_link'
EE_LINK = 'ee_link'

# Set to identity unless you want the goal to have a certain orientation.
EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 

# Only edit these when editing the robot joints and links. 
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK, EE_LINK]

# Hyperparamters to be tuned for optimizing policy learning on the specific robot.
UR_GAINS = np.array([1, 1, 1, 1, 1, 1])

# Set end effector constants
INITIAL_JOINTS = [0, -np.pi/2, np.pi/2, 0, 0, 0]
