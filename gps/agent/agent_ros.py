import abc
import copy  # Only used to copy the agent config data.
import numpy as np  # Used pretty much everywhere.
import timeit

from gps.agent.agent import Agent  # GPS class needed to inherit from.
from gps.agent.agent_utils import setup, generate_noise  # setup used to get hyperparams in init and generate_noise to get noise in sample.
from gps.sample.sample import Sample  # Used to build a Sample object for each sample taken.
from gps.proto.gps_pb2 import ACTION, NOISE

import gps.agent.utils.log as utils


class AgentROS(Agent):
    """Connects the Robot actions and GPS algorithms."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        """Initialized Agent.
        hyperparams: Dictionary of hyperparameters."""

        # Pull parameters from hyperparams file.
        config = {}
        config.update(hyperparams)
        Agent.__init__(self, config)
        conditions = self._hyperparams['conditions']
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        self.x0 = self._hyperparams['x0']
        self.color_log = utils.TextColors()

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """This is the main method run when the Agent object is called by GPS.
        Draws a sample from the environment, using the specified policy and
        under the specified condition.
        If "save" is True, then append the sample object of type Sample to
        self._samples[condition]."""

        # Reset the arm to initial configuration at start of each new trial.
        self.reset(condition)

        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features

        U = np.zeros([self.T, self.dU])

        # Generate noise to be used in the policy object to compute next state.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Execute the trial.
        new_sample = self._init_sample(condition, feature_fn)

        start_time = timeit.default_timer()
        for time_step in range(self.T):
            state = new_sample.get_X(t=time_step)
            obs = new_sample.get_obs(t=time_step)
            # Pass the state vector through the policy to get a vector of
            # actions to go to next.
            action = policy.act(state, obs, time_step, noise[time_step])
            U[time_step, :] = action

            if (time_step + 1) < self.T:
                self.act(action, condition)

                # update state
                self._set_sample(
                    new_sample, condition, time_step, feature_fn=feature_fn)

        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)

        if verbose:
            end_time = timeit.default_timer() - start_time
            self.color_log.warning(
                "iter time: %s segs" % np.round(end_time, 2))

        # Save the sample to the data structure. This is controlled by gps_main.py.
        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, condition, -1, feature_fn=feature_fn)
        return sample

    def _set_sample(self, sample, condition, time_step, feature_fn=None):
        state_data = self.get_state(condition)
        obs_data = self.get_observations(condition, sample, feature_fn)

        for feature, value in state_data.iteritems():
            sample.set(feature, value, time_step + 1)

        for feature, value in obs_data.iteritems():
            if feature in self.obs_data_types:
                sample.set(feature, value, time_step + 1)

    def set_hyperparameters(self, hyperparameters):
        self._hyperparams.update(hyperparameters)

    def end(self):
        """
        end of training
        """
        pass

    @abc.abstractmethod
    def get_state(self, condition):
        """
        Get all the necessary information for the defined state
        Return dict
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_observations(self, condition, sample=None, feature_fn=None):
        """
        Get all the necessary information for the defined observations
        Return dict
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def act(self, action, condition):
        """
        Execute action in corresponding world/environment.
        """
        raise NotImplementedError("Must be implemented in subclass.")
