import argparse
import copy
import imp

import logging
import os
import sys
import pprint
import random
import timeit
import yaml
import rospy
import numpy as np

from datetime import datetime
from gps.utility.data_logger import DataLogger
from gps.gps_main import GPSMain

pp = pprint.PrettyPrinter(indent=2)

def load_exp_dir():
    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-2]) + '/'
    global exp_dir
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'


def load_hyperparemeters():
    hyperparams_file = exp_dir + 'hyperparams.py'

    # Check if hyperparams file exist
    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    return imp.load_source('hyperparams', hyperparams_file)


def read_file():
    stream = open(exp_dir + 'testing_params.yaml', 'r')
    params = yaml.load(stream)
    return params


def parse_params():
    training_params = read_file()
    exp_global_name = 'mean_error' + '_' + datetime.strftime(
        datetime.now(), '%m-%d-%y_%H-%M')

    _params = copy.deepcopy(training_params['common'])
    _params.update({'exp_name': 'default'})
    _params.update({'exp_global_name': exp_global_name})
    return _params


def default_condition():
    _params = parse_params()
    return [get_default_params(_params)]


def create_individual_conditions():
    _params = parse_params()
    num_tests = _params['num_tests']

    if num_tests > 1:
        conditions = read_training_params(_params, num_tests)
    else:
        # Add one condition with only default params
        conditions = [get_default_params(_params)]

    print "Conditions: #", len(conditions)
    global verbose
    if verbose:
        pp.pprint(conditions)

    return conditions


def create_dependent_conditions():
    _params = parse_params()
    num_tests = _params['num_tests']

    conditions = []
    if num_tests > 1:
        variables = extract_variables(_params, num_tests)
        default = get_default_params(_params)
        for i in range(num_tests):
            c = copy.deepcopy(default)
            build_condition(c, variables, i)
            conditions.append(c)
    else:
        # Add one condition with only default params
        conditions.append(get_default_params(_params))

    print "Conditions: #", len(conditions)
    global verbose
    if verbose:
        pp.pprint(conditions)

    return conditions


def build_condition(c, variables, itr):
    for key, value in variables.iteritems():
        if isinstance(value, dict):
            build_condition(c[key], value, itr)
            c.update({'exp_name': key + '~' + c[key]['exp_name']})
        else:
            if 'exp_name' not in c or c['exp_name'] == 'default':
                c.update({'exp_name': key + ':' + str(value[itr])})
            else:
                c.update({
                    'exp_name':
                    c['exp_name'] + '@' + key + ':' + str(value[itr])
                })
            c.update({key: value[itr]})


def extract_variables(params, num_tests):
    variables = {}
    # iterate over parameters that will be tested
    # add as many condition as num_tests defined
    for key, value in params.iteritems():
        if isinstance(value, list) and len(value) == 3:
            test_vals = np.linspace(value[0], value[1], num=num_tests)
            if isinstance(value[0], int):
                test_vals = test_vals.astype(int)
            variables.update({key: test_vals})

        if isinstance(value, dict):
            test_vals = extract_variables(value, num_tests)
            if test_vals:
                variables.update({key: test_vals})

    return variables


def read_training_params(params, num_tests):
    conditions = []
    # iterate over parameters that will be tested
    # add as many condition as num_tests defined
    for key, value in params.iteritems():
        if isinstance(value, list) and len(value) == 3:
            for i in range(num_tests):
                test_val = np.linspace(value[0], value[1], num=num_tests)[i]
                if isinstance(value[0], int):
                    test_val = int(test_val)

                name = key + ':' + str(test_val)
                create_condition(test_val, key, name, params, conditions)
        elif isinstance(value, dict):
            sub_conditions = read_training_params(value, num_tests)
            for subc in sub_conditions:
                name = key + '~' + subc.pop('exp_name')
                create_condition(subc, key, name, params, conditions)

    return conditions


def create_condition(test_val, key, name, params, conditions):
    default_params = get_default_params(params)
    default_params.update({key: test_val})

    default_params.update({'exp_name': name})

    conditions.append(default_params)


def get_default_params(params):
    def_params = copy.deepcopy(params)
    for key, value in params.iteritems():
        if type(value) == list and len(value) == 3:
            def_params.update({key: value[2]})
        elif isinstance(value, dict):
            def_params.update({key: get_default_params(value)})
    return def_params


def write_params(params):
    with open(exp_dir + 'params.yaml', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
        outfile.close()


def train(conditions):
    start_time = timeit.default_timer()

    for i in range(len(conditions)):
        cur_var = conditions[i]['exp_name'].split('~')[0]

        print "Global conditions ", (i + 1), "of", len(conditions)
        write_params(conditions[i])

        # Load hyperparams file
        hyperparams = load_hyperparemeters()

        # Random seed for experiments
        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        hyperparams.config['agent'].update({'real_robot': real_robot})

        gps = GPSMain(hyperparams.config, False)
        gps.run()

        end_time = timeit.default_timer() - start_time
        print "Params Training time: ", end_time / 60., "min"


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(
        description='Run a given policy on an Agent')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument(
        '-i',
        '--individual',
        action='store_true',
        help='run test with individual params')
    parser.add_argument(
        '-d',
        '--dependent',
        action='store_true',
        help='run test with dependent params')
    parser.add_argument(
        '-r',
        '--repetitions',
        metavar='R',
        type=int,
        help='run policy for R iterations')
    parser.add_argument(
        '-q',
        '--ros',
        action='store_true',
        help='ros agent?')
    parser.add_argument('--robot', action='store_true', help="use real robot")
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='debug print outs')

    args = parser.parse_args()

    global exp_name
    exp_name = args.experiment
    repetitions = args.repetitions
    global real_robot
    real_robot = args.robot

    # Logging config
    global verbose
    if args.verbose is not None:
        verbose = True
    else:
        verbose = False

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if args.ros:
        rospy.init_node('gps_agent_ur_ros_node')

    if os.path.exists('/home/cambel/gps/logging.yaml'):
        with open('/home/cambel/gps/logging.yaml', 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    load_exp_dir()

    if args.individual:
        print "Individual tests"
        conditions = create_individual_conditions()
    elif args.dependent:
        print "Dependent tests"
        conditions = create_dependent_conditions()
    else:
        conditions = default_condition()

    if repetitions is not None:
        start_time = timeit.default_timer()
        for i in range(repetitions):
            print "repetitions:", (i + 1), "of", repetitions
            cs = copy.deepcopy(conditions)
            [c.update({'exp_name': c['exp_name'] + str(i)}) for c in cs]
            train(cs)
            end_time = timeit.default_timer() - start_time
        print "Reps time: ", end_time / 60., "min"
    else:
        train(conditions)


if __name__ == "__main__":
    main()
