import argparse
from gps.utility.data_logger import DataLogger
import imp
import os
import random
import timeit
import rospy
import sys
import logging
import yaml
import numpy as np
try:
    from gps.agent.baxter import gazebo_spawner
except ImportError:
    print "not baxter module"

import signal

def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def run_multi_test(config, tests_from, tests_to, repetitions, itr=0):
    data_files_dir = '/media/cambel/Data/documents/gps/experiments/paper-08-18/'
    # /media/cambel/Data/documents/gps/experiments/paper-08-18/02/data_files/policy_itr_00.pkl
    data_logger = DataLogger()

    conditions = config['common']['conditions']
    policy_prefix = 'policy_itr_'
    policy_name = 'data_files/' + policy_prefix + '%02d.pkl' % itr

    print tests_from, tests_to
    for i in xrange(tests_from, tests_to):
        np.save("/tmp/gps_test.npy", np.array([i]))
        policy_file = data_files_dir + str(i).zfill(2) + "/" + policy_name
        print "Current policy", policy_file
        policy_opt = data_logger.unpickle(policy_file)
        pol = policy_opt.policy

        agent = config['agent']['type'](config['agent'])

        grasp_count = 0
        for cond in list(range(conditions)):
            reps = 0
            if repetitions is None:
                reps = config['num_samples']
            else:
                reps = repetitions

            for i in range(reps):
                gp = agent.execute(pol, cond, verbose=(i < config['verbose_trials']), noisy=False)
                if gp:
                    grasp_count += 1
        print "test:", i, grasp_count

def load_hyperparemeters(exp_name):
    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-2]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    # Check if hyperparams file exist
    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    return imp.load_source('hyperparams', hyperparams_file)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run a given policy on an Agent')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='run policy N')
    parser.add_argument('-r', '--repetitions', metavar='R', type=int,
                        help='run policy for R iterations')
    parser.add_argument('-a', '--algorithm', metavar='A', type=int,
                        help='run policy in algorithm A')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-t', '--test', action='store_true',
                        help='test several policies')
    parser.add_argument('-e', '--start', metavar='E', type=int,
                        help='start experiments from folder')
    parser.add_argument('-f', '--finish', metavar='F', type=int,
                        help='finish experiments with folder')
    parser.add_argument('-m', '--model', action='store_true',
                        help='run with models')
    parser.add_argument('--robot', action='store_true',
                        help='real robot')
    parser.add_argument(
        '-q',
        '--ros',
        action='store_true',
        help='ros agent?')
    args = parser.parse_args()

    exp_name = args.experiment
    run_policy_N = args.policy
    run_policy_A = args.algorithm
    repetitions = args.repetitions
    exp_from = args.start
    exp_to = args.finish

    data_logger = DataLogger()

    if os.path.exists('/home/cambel/gps/logging.yaml'):
        with open('/home/cambel/gps/logging.yaml', 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    # Load hyperparams file
    hyperparams = load_hyperparemeters(exp_name)
    config = hyperparams.config

    if args.ros or args.robot:
        rospy.init_node('run_pol')
        
    # Random seed for experiments
    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    conditions = config['common']['conditions']

    if args.test:
        run_multi_test(config, exp_from, exp_to, repetitions, 0)
        return

    data_files_dir = hyperparams.config['common']['experiment_dir'] + 'data_files/'

    pol = None
    algorithm = None
    if run_policy_N is not None:
        policy_prefix = 'policy_itr_'
        policy_file = data_files_dir + policy_prefix + '%02d.pkl' % run_policy_N
        print '%02d.pkl' % run_policy_N
        policy_opt = data_logger.unpickle(policy_file)
        pol = policy_opt.policy

    elif run_policy_A is not None:
        algorithm_prefix = 'algorithm_itr_'

        algorithm_file = data_files_dir + algorithm_prefix + '%02d.pkl' % run_policy_A
        print algorithm_file
        algorithm = data_logger.unpickle(algorithm_file)
        if hasattr(algorithm, 'policy_opt'):
            pol = algorithm.policy_opt.policy
        else:
            algorithm = algorithm.cur

    start_time = timeit.default_timer()  # TIMER

    if args.model:
        grasp_count = 0
        for model in ["can", "ball", "block", "cup", "nut", "mechanical_part"]:
            models = config['agent']['models']
            for cond in list(range(conditions)):
                models[cond][0].name = model
            print "" 
            print "test:", model
            run(config, algorithm, repetitions, pol, real_robot=False)
         
            gazebo_spawner.delete_model(model)
    else:
        run(config, algorithm, repetitions, pol, real_robot=args.robot)

    elapsed = timeit.default_timer() - start_time  # TIMER
    print 'CB time: iteration', elapsed/60., "min" # TIMER

def run(config, algorithm, repetitions, pol, real_robot=False):
    conditions = config['common']['conditions']
    if real_robot:
        config['agent'].update({'real_robot':True})
    agent = config['agent']['type'](config['agent'])
    
    for cond in list(range(conditions)):
        grasp_count = 0
        if type(algorithm) == list:
            pol = algorithm[cond].traj_distr
        reps = 0
        if repetitions is None:
            reps = config['num_samples']
        else:
            reps = repetitions
            
        for i in range(reps):
            gp = agent.sample(pol, cond, verbose=(i < config['verbose_trials']), noisy=False)
            if gp:
                grasp_count += 1
        print "condition", cond ,":", grasp_count, "/", config['num_samples'] if repetitions  is None else repetitions
    
if __name__ == "__main__":
    main()
