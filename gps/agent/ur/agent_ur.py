""" Agent for the UR robot controlled by joint angles """
import copy
import logging
import numpy as np  # Used pretty much everywhere.
import rospy  # Needed for nodes, rate, sleep, publish, and subscribe.
import timeit
import sys

from gps.sample.sample import Sample
from gps.agent.agent_ros import AgentROS
from gps.agent.config import AGENT_UR_ROS  # Parameters needed for config in __init__.
from gps.utility.general_utils import get_ee_points  # For getting points and velocities.
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,\
                              END_EFFECTOR_POINT_JACOBIANS, END_EFFECTOR_POINT_VELOCITIES,\
                              TORQUE, TORQUE_JACOBIAN

from gps.agent.utils.gazebo_spawner import GazeboModels
from gps.agent.ur.arm import Arm

try:
    from ur_control.controllers import GripperController
except ImportError:
    print "gripper not available"
    
from pyquaternion import Quaternion

LOGGER = logging.getLogger(__name__)


class AgentUR(AgentROS):
    """Connects the UR actions and GPS algorithms."""

    def __init__(self, hyperparams):
        """Initialized UR robot"""

        config = copy.deepcopy(AGENT_UR_ROS)
        config.update(hyperparams)
        AgentROS.__init__(self, config)

        # Init robot controllers
        # Init robot controllers
        self.arm = Arm(
            ft_sensor=self._hyperparams['ft_sensor'], 
            real_robot=self._hyperparams['real_robot'],
            robot_urdf=self._hyperparams['robot_urdf'])

        # Keep track of the previous joints configuration
        # for computing the End Effector velocity
        self._previous_joints = None

        # Used for enforcing the period specified in hyperparameters.
        self.dt = self._hyperparams['dt']

        # set up Gazebo Models
        if self._hyperparams['use_gripper']:
            self.gripper = GripperController()
            self.gripper.open()
        if 'models' in self._hyperparams:
            self.sim = GazeboModels(self._hyperparams['models'], 'ur3_gazebo')

        self._dist_list = []
        if 'data_dir' in self._hyperparams:
            self._log_file = open(
                hyperparams['data_dir'] + hyperparams['experiment_name'] +
                '.log', 'a+')
            self._log_file.write(hyperparams['exp_name'] + " \n")

        self.indices = None
        if 'indices' in self._hyperparams:
            self.indices = self._hyperparams['indices']

    def end(self):
        if 'data_dir' in self._hyperparams:
            l = self._dist_list
            self._log_file.write("  " + str(l) + "\n")
            self._log_file.close()

    def reset(self, condition):
        """Not necessarily a helper function as it is inherited.
        Reset the agent for a particular experiment condition.
        condition: An index into hyperparams['reset_conditions']."""
        self.condition = condition

        if self._hyperparams['init_seq'] is not None:
            self._hyperparams['init_seq'](self.arm, self.gripper, None,
                                          condition)

        else:
            if self._hyperparams['use_gripper']:
                self.gripper.open()

            # Set the rest position as the initial position from agent hyperparams.
            reset_conditions = self._hyperparams['reset_conditions'][condition]
            action = reset_conditions[JOINT_ANGLES]

            if 'waypoints' in reset_conditions \
                and reset_conditions['waypoints']:
                for a in action:
                    self.arm.set_joint_positions(
                        position=a, wait=True,
                        t=self._hyperparams['reset_slowness'])
                    self._previous_joints = a
            else:
                self.arm.set_joint_positions(
                    position=action, wait=True,
                    t=self._hyperparams['reset_slowness'])
                self._previous_joints = action

            if 'models' in self._hyperparams:
                self.sim.reset_model(condition)

    def act(self, action, condition):
        action /= 100.0
        if 'limits' in self._hyperparams:
            action = np.array([action[i] if action[i] <= self._hyperparams['limits'][i] \
                        else self._hyperparams['limits'][i] for i in range(self.dU)])

        if self.indices:
            initial_q = copy.copy(self._hyperparams['reset_conditions'][condition][JOINT_ANGLES])
            initial_q = self.arm.joint_angles()
            for (index, _action) in zip(self.indices, action):
                initial_q[index] += _action
            action = initial_q
        else:
            action = self.arm.joint_angles() + action

        # Do not exceed joint limits
        if np.isnan(np.sum(action)) or np.isinf(np.sum(action)):
            print "action overflow?", action
            raise ValueError("Invalid actions requested")

        action[(action > 2 * np.pi)] = 2 * np.pi
        
        self.arm.set_joint_positions_flex(
            position=action, t=self._hyperparams['slowness'])

        rospy.sleep(self.dt)

    def _get_state(self, condition, attributes):
        state = {}
        joint_angles = self.arm.joint_angles()

        if JOINT_ANGLES in attributes:
            q = joint_angles
            if self.dU == 2:
                q = np.array([joint_angles[i] for i in self.indices])
            state.update({JOINT_ANGLES: q})
        if JOINT_VELOCITIES in attributes:
            qv = self.arm.joint_velocities()
            if self.dU == 2:
                qv = np.array([qv[i] for i in self.indices])
            state.update({JOINT_VELOCITIES: qv})
        if END_EFFECTOR_POINTS in attributes:
            self.ee_points, ee_velocities = \
                self._get_points_and_vels(joint_angles, condition, debug=False)

            state.update({END_EFFECTOR_POINTS: self.ee_points})
            if END_EFFECTOR_POINT_VELOCITIES in attributes:
                state.update({END_EFFECTOR_POINT_VELOCITIES: ee_velocities})

        if JOINT_ANGLES in attributes and END_EFFECTOR_POINTS in attributes:
            ee_jacobians = self.eepts_jacobians(
                self._hyperparams['end_effector_points'],
                joint_angles)
            state.update({END_EFFECTOR_POINT_JACOBIANS: ee_jacobians})

        if self._hyperparams['ft_sensor']:
            # collect ft sensor measure
            ft = self.arm.get_ee_wrench()
            state.update({TORQUE: ft})

            if JOINT_ANGLES in attributes:
                ft_jacobians = self.arm.kinematics.jacobian(
                    joint_angles)
                state.update({TORQUE_JACOBIAN: ft_jacobians})

        return state

    def get_state(self, condition):
        return self._get_state(condition, self.x_data_types)

    def get_observations(self, condition, sample=None, feature_fn=None):
        return self._get_state(condition, self.obs_data_types)

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        sample = AgentROS.sample(self, policy, condition, verbose, save, noisy)
        self.arm.set_joint_positions(position=self.arm.joint_angles(), wait=True,
                        t=self._hyperparams['reset_slowness'])
        # Display meters off from goal.
        dist_mean = np.round(np.mean(np.abs(self.ee_points)), 3)
        self._dist_list.append(dist_mean)

        dist = np.round((self.ee_points), 2)
        self.color_log.warning('Distance from Goal ' + str(dist) + ' cm | mean ' + str(dist_mean))
        return sample

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        """
        return AgentROS._init_sample(self, condition, feature_fn=feature_fn)

    def _get_points_and_vels(self, joint_angles, condition, debug=False):
        """
        Helper function that gets the cartesian positions
        and velocities from ROS."""

        if self._previous_joints is None:
            self._previous_joints = self._hyperparams['reset_conditions'][
                condition][JOINT_ANGLES]

        # Current position
        ee_pos_now = self.__get_ee_pose(joint_angles)
        # print ee_pos_now
        # Last position
        ee_pos_last = self.__get_ee_pose(self._previous_joints)
        self._previous_joints = joint_angles  # update

        # Use the past position to get the present velocity.
        velocity = (ee_pos_now - ee_pos_last) / self.dt

        # Shift the present poistion by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        if self.dU == 2:
            ee_pos_now = [ee_pos_now[i] for i in self.indices]
            position = (np.asarray(ee_pos_now) - np.asarray(
                self._hyperparams['ee_points_tgt'][condition])) * 100.
            position = np.squeeze(np.asarray(position))
            velocity = np.array([velocity[i] for i in self.indices])
        else:
            position = (np.asarray(ee_pos_now) - np.asarray(
                self._hyperparams['ee_points_tgt'][condition]))[0] * 100.
            position = np.squeeze(np.asarray(position))

        if debug:
            print 'VELOCITY:', velocity
            print 'POSITION:', position

        return position, velocity

    def __get_ee_pose(self, joint_angles):
        EE_POINTS = np.array(self._hyperparams['end_effector_points'])

        pose = self.arm.kinematics.forward_position_kinematics(
            joint_angles)  #[x, y, z, ax, ay, az, w]
        translation = np.array(pose[:3]).reshape(1, 3)
        rot = Quaternion(np.roll(pose[3:], 1)).rotation_matrix

        return np.ndarray.flatten(
            get_ee_points(EE_POINTS,
                          np.array(translation).reshape((1, 3)), rot).T)

    def eepts_jacobians(self, eepts, joint_angles=None):
        "Compute Jacobians if multiple eepts are used"
        temp_jacobian_ = self.arm.kinematics.jacobian(joint_angles)
        _pose = self.arm.kinematics.forward_position_kinematics(joint_angles)
        rot = Quaternion(np.roll(_pose[3:], 1)).rotation_matrix

        n_actuator = temp_jacobian_.shape[1]
        n_eep = eepts.flatten().shape[0]

        point_jacobians_ = np.zeros(n_eep * n_actuator).reshape(
            n_eep, n_actuator)
        point_jacobians_rot_ = np.zeros(n_eep * n_actuator).reshape(
            n_eep, n_actuator)

        for i in range(n_eep / 3):
            site_start = i * 3
            ovec = np.array(eepts)[i, :]

            for j in range(3):
                for k in range(n_actuator):
                    point_jacobians_[site_start + j, k] = temp_jacobian_[j, k]
                    point_jacobians_rot_[site_start +
                                         j, k] = temp_jacobian_[j + 3, k]

            #  Compute site Jacobian.
            ovec = np.asarray(ovec * rot)[0]

            for k in range(n_actuator):
                point_jacobians_[site_start, k] += point_jacobians_rot_[
                    site_start + 1, k] * ovec[2] - point_jacobians_rot_[
                        site_start + 2, k] * ovec[1]
                point_jacobians_[
                    site_start +
                    1, k] += point_jacobians_rot_[site_start + 2, k] * ovec[
                        0] - point_jacobians_rot_[site_start, k] * ovec[2]
                point_jacobians_[
                    site_start +
                    2, k] += point_jacobians_rot_[site_start, k] * ovec[
                        1] - point_jacobians_rot_[site_start + 1, k] * ovec[0]

        return point_jacobians_
