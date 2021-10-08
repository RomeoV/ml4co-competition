import argparse
import itertools
import os
import pyscipopt as pyopt
from collections import OrderedDict
import json

"""Config id's 0-64 are all exhaustive combinations of Presolve, Separaring and Heuristic with Emphasis set to off (note, in 
order to make sure there are no interferences beteween emphasis and the others, we don't set emphasis in this case). 
All config ids from 64+ are combinations of presolve, separating, heuristic and emphasis."""

SETTINGS = OrderedDict({
    'OFF': pyopt.SCIP_PARAMSETTING.OFF,
    'DEFAULT': pyopt.SCIP_PARAMSETTING.DEFAULT,
    'FAST': pyopt.SCIP_PARAMSETTING.FAST,
    'AGGRESSIVE': pyopt.SCIP_PARAMSETTING.AGGRESSIVE,
    })

# remember
# DEFAULT = 0
# AGGRESSIVE = 1
# FAST = 2
# OFF = 3

EMPHASIS_SETTINGS = OrderedDict({
'SCIP_PARAMEMPHASIS_DEFAULT': pyopt.SCIP_PARAMEMPHASIS.DEFAULT,
'SCIP_PARAMEMPHASIS_CPSOLVER': pyopt.SCIP_PARAMEMPHASIS.CPSOLVER,
'SCIP_PARAMEMPHASIS_EASYCIP': pyopt.SCIP_PARAMEMPHASIS.EASYCIP,
'SCIP_PARAMEMPHASIS_FEASIBILITY': pyopt.SCIP_PARAMEMPHASIS.FEASIBILITY,
'SCIP_PARAMEMPHASIS_HARDLP': pyopt.SCIP_PARAMEMPHASIS.HARDLP,
'SCIP_PARAMEMPHASIS_OPTIMALITY' : pyopt.SCIP_PARAMEMPHASIS.OPTIMALITY,
'SCIP_PARAMEMPHASIS_COUNTER': pyopt.SCIP_PARAMEMPHASIS.COUNTER,
'SCIP_PARAMEMPHASIS_PHASEFEAS': pyopt.SCIP_PARAMEMPHASIS.PHASEFEAS,
'SCIP_PARAMEMPHASIS_PHASEIMPROVE': pyopt.SCIP_PARAMEMPHASIS.PHASEIMPROVE,
'SCIP_PARAMEMPHASIS_PHASEPROOF' : pyopt.SCIP_PARAMEMPHASIS.PHASEPROOF,
'SCIP_PARAMEMPHASIS_NUMERICS' : pyopt.SCIP_PARAMEMPHASIS.NUMERICS
})



def main():
    os.makedirs('meta_configs', exist_ok=True)
    config_id_to_parameter_setting = {}
    create_heuristic_presolve_separating_configs(config_id_to_parameter_setting=config_id_to_parameter_setting)
    create_emphasis_configs(config_id_to_parameter_setting=config_id_to_parameter_setting)

    with open('meta_configs/config_id_to_parameters.json', 'w') as fp:
        json.dump(config_id_to_parameter_setting, fp)

def create_heuristic_presolve_separating_configs(config_id_to_parameter_setting):
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=3)):
        presolve, heuristic, separating = setting
        print(f'Config {i} \n \
            presolve: {presolve[0]} \n \
            heuristic: {heuristic[0]} \n \
            separating: {separating[0]} \n')
        config_id_to_parameter_setting[i] = [presolve[1], heuristic[1], separating[1], 0]
        set_parameters(i, presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=separating[1])


def create_emphasis_configs(config_id_to_parameter_setting):
    overall_index = 64

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_CPSOLVER"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 3
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve, heuristic, separating = setting
        overall_index += 1
        print(f'Config {overall_index} \n \
                    emphasis: {emphasis_parameter_index} \n \
                    presolve: {presolve[0]} \n \
                    heuristic: {heuristic[0]} \n \
                    separating: {separating[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], heuristic[1], separating[1], emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=separating[1],
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_EASYCIP"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    overall_index += 1
    print(f'Config {overall_index} \n \
                emphasis: {emphasis_parameter_index} \n')
    config_id_to_parameter_setting[overall_index] = [3, 3, 3, emphasis_parameter]

    set_parameters(overall_index,
                   presolve_parameter=None,
                   heuristic_parameter=None,
                   separating_paramter=None,
                   emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_FEASIBILITY"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 1
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve = setting[0]
        overall_index += 1
        print(f'Config {overall_index} \n \
                      emphasis: {emphasis_parameter_index} \n \
                      presolve: {presolve[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], 3, 3, emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=None,
                       separating_paramter=None,
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_HARDLP"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 1
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        separating = setting[0]
        overall_index += 1
        print(f'Config {overall_index} \n \
                          emphasis: {emphasis_parameter_index} \n \
                          separating: {separating[0]} \n')
        config_id_to_parameter_setting[overall_index] = [3, 3, separating[1], emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=None,
                       heuristic_parameter=None,
                       separating_paramter=separating[1],
                       emphasis_parameter=emphasis_parameter)


    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_OPTIMALITY"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 2
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve, heuristic = setting
        overall_index += 1
        print(f'Config {overall_index} \n \
                          emphasis: {emphasis_parameter_index} \n \
                          presolve: {presolve[0]} \n \
                          heuristic: {heuristic[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], heuristic[1], 3, emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=None,
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_COUNTER"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 1
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve = setting[0]
        overall_index += 1
        print(f'Config {overall_index} \n \
                          emphasis: {emphasis_parameter_index} \n \
                          presolve: {presolve[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], 3, 3, emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=None,
                       separating_paramter=None,
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_PHASEFEAS"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 3
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve, heuristic, separating = setting
        overall_index += 1
        print(f'Config {overall_index} \n \
                    emphasis: {emphasis_parameter_index} \n \
                    presolve: {presolve[0]} \n \
                    heuristic: {heuristic[0]} \n \
                    separating: {separating[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], heuristic[1], separating[1], emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=separating[1],
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_PHASEIMPROVE"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 3
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve, heuristic, separating = setting
        overall_index += 1
        print(f'Config {overall_index} \n \
                    emphasis: {emphasis_parameter_index} \n \
                    presolve: {presolve[0]} \n \
                    heuristic: {heuristic[0]} \n \
                    separating: {separating[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], heuristic[1], separating[1], emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=separating[1],
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_PHASEPROOF"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 1
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve = setting[0]
        overall_index += 1
        print(f'Config {overall_index} \n \
                         emphasis: {emphasis_parameter_index} \n \
                         presolve: {presolve[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], 3, 3, emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=None,
                       separating_paramter=None,
                       emphasis_parameter=emphasis_parameter)

    emphasis_parameter_index = "SCIP_PARAMEMPHASIS_NUMERICS"
    emphasis_parameter = EMPHASIS_SETTINGS[emphasis_parameter_index]
    number_of_meta_parameters_to_set = 3
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=number_of_meta_parameters_to_set)):
        presolve, heuristic, separating = setting
        overall_index += 1
        print(f'Config {overall_index} \n \
                    emphasis: {emphasis_parameter_index} \n \
                    presolve: {presolve[0]} \n \
                    heuristic: {heuristic[0]} \n \
                    separating: {separating[0]} \n')
        config_id_to_parameter_setting[overall_index] = [presolve[1], heuristic[1], separating[1], emphasis_parameter]

        set_parameters(overall_index,
                       presolve_parameter=presolve[1],
                       heuristic_parameter=heuristic[1],
                       separating_paramter=separating[1],
                       emphasis_parameter=emphasis_parameter)



def set_parameters(index, presolve_parameter=None, heuristic_parameter=None, separating_paramter=None, emphasis_parameter=None,):
    model = pyopt.Model()
    if emphasis_parameter is not None:
        model.setEmphasis(emphasis_parameter)
    if presolve_parameter is not None:
        model.setPresolve(presolve_parameter)
    if heuristic_parameter is not None:
        model.setHeuristics(heuristic_parameter)
    if separating_paramter is not None:
        model.setSeparating(separating_paramter)


    model.writeParams(filename=f'meta_configs/config-{index}.set',
                      comments=False,
                      onlychanged=True)


if __name__ == '__main__':
    main()
