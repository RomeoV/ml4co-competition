import argparse
import pathlib
import csv
import json

import ecole as ec

import numpy as np
import random
import ConfigSpace.hyperparameters as CSH

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
#from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.initial_design.random_configuration_design import RandomConfigurations

# generate parameters in correct format for SMAC from file
def getParamsFromFile(paramfile):
    file = open(paramfile,'r').readlines()

    params = []

    i = 0
    while i < len(file):
        line = file[i].split()

        paramname = line[0]
        paramtype = line[1]
        default = line[3][1:-1]

        if paramtype == 'categorical':
            params += [CSH.CategoricalHyperparameter(paramname, choices=line[2][1:-1].split(','), default_value=default)]
        else:
            bounds = line[2][1:-1].split(',')

            if paramtype == 'real':

                if float(default) > 1e+100:
                    i += 1
                    continue
                
                # SMAC cannot handle too large limits, so change them to something smaller
                if float(bounds[0]) <= -1e+100:
                    bounds[0] = -1e+100
                if float(bounds[1]) >= 1e+100:
                    bounds[1] = 1e+100

                params += [CSH.UniformFloatHyperparameter(paramname, float(bounds[0]), float(bounds[1]), default_value=float(default))]
            
            elif paramtype == 'integer':
                params += [CSH.UniformIntegerHyperparameter(paramname, lower=int(bounds[0]), upper=int(bounds[1]), default_value=int(default))]
        i += 1

    return params

# runs ecole and returns primal-dual integral as a reward
def runEcole(settings, instance):

    print("New Ecole run with instance ", instance)

    # read the instance's initial primal and dual bounds from JSON file
    with open(pathlib.PosixPath(instance).with_name(pathlib.PosixPath(instance).stem).with_suffix('.json')) as f:
        instance_info = json.load(f)

    # set up the reward function parameters for that instance
    initial_primal_bound = instance_info["primal_bound"]
    initial_dual_bound = instance_info["dual_bound"]
    objective_offset = 0

    TimeLimitPrimalDualIntegral().set_parameters(
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
            objective_offset=objective_offset)

    # start a new episode
    env.reset(instance)

    # get the next action from the given settings
    action = {k: settings[k] for k in settings}

    for k in action.keys():
        if action[k] == 'TRUE':
            action[k] = True
        elif action[k] == 'FALSE':
            action[k] = False

    # apply the action and collect the reward
    _, _, reward, _, _ = env.step(action)

    return reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='Problem benchmark to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    parser.add_argument(
        '-t', '--timelimit',
        help='Instance time limit (in seconds).',
        default=300,
        type=float,
    )
    parser.add_argument(
        '-i', '--ninstances',
        help='Number of training instances.',
        default=10,
        type=int,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random seed for instance selection.',
        default=0,
        type=int,
    )
    parser.add_argument(
        '-e', '--nevaluations',
        help='Number of function evaluations of SMAC.',
        default=10,
        type=int,
    )
    args = parser.parse_args()

    # collect the instance files
    if args.problem == 'item_placement':
        instances_path = pathlib.Path(f"../../instances/1_item_placement/train/")
        results_file = pathlib.Path(f"results/config/1_item_placement.csv")
    elif args.problem == 'load_balancing':
        instances_path = pathlib.Path(f"../../instances/2_load_balancing/train/")
        results_file = pathlib.Path(f"results/config/2_load_balancing.csv")
    elif args.problem == 'anonymous':
        instances_path = pathlib.Path(f"../../instances/3_anonymous/train/")
        results_file = pathlib.Path(f"results/config/3_anonymous.csv")

    print(f"Processing instances from {instances_path.resolve()}")
    instance_files = list(instances_path.glob('*.mps.gz'))

    # randomly choose set of training instances
    indices = [i for i in range(len(instance_files))]
    random.seed(a=args.seed)
    random.shuffle(indices)
    instances = [instance_files[i] for i in indices[:args.ninstances]]

    # write instances to a file
    instancefile = open('instances.txt', "a+")
    for i in range(len(instances)):
        instancefile.write(str(instances[i]) + '\n')
    instancefile.close()

    print(f"Saving results to {results_file.resolve()}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = ['instance', 'seed', 'initial_primal_bound', 'initial_dual_bound', 'objective_offset', 'cumulated_reward']
    with open(results_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    # get primal-dual integral function that is also used in evaluation
    import sys
    sys.path.insert(1, str(pathlib.Path(f"../../common/")))
    from rewards import TimeLimitPrimalDualIntegral
    
    # create ecole environment
    env = ec.environment.Configuring(
          
          # set time limit for each instance
          scip_params = {'limits/time' : args.timelimit},
          
          # pure bandit, no observation
          observation_function = None,

          # minimize the primal-dual integral
          reward_function = TimeLimitPrimalDualIntegral(),

          # collect additional metrics for information purposes
          information_function = {}
    )

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    params = getParamsFromFile('parameters.pcs')
    cs.add_hyperparameters(params)

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (aka primal-dual integral)
                         "runcount-limit": args.nevaluations, # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "false",
                         "instance_file": "instances.txt"
                         })

    # optimize with SMAC
    smac = SMAC4HPO(scenario = scenario, initial_design=RandomConfigurations, tae_runner = runEcole)

    result = smac.optimize()
    result = {k: result[k] for k in result}

    print("\n")
    print("##############################")
    print("Best parameter settings")
    print("##############################")
    print(result)
