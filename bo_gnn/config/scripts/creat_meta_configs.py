import argparse
import itertools
import os
import pyscipopt as pyopt
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--nodry', default=False, action='store_true')

SETTINGS = OrderedDict({
    'OFF': pyopt.SCIP_PARAMSETTING.OFF,
    'DEFAULT': pyopt.SCIP_PARAMSETTING.DEFAULT,
    'FAST': pyopt.SCIP_PARAMSETTING.FAST,
    'AGGRESSIVE': pyopt.SCIP_PARAMSETTING.AGGRESSIVE,
    })

def main():
    args = parser.parse_args()
    os.makedirs('../meta_configs', exist_ok=True)
    for i, setting in enumerate(itertools.product(SETTINGS.items(), repeat=3)):
        presolve, heuristic, separating = setting
        print(f'Config {i} \n \
            presolve: {presolve[0]} \n \
            heuristic: {heuristic[0]} \n \
            separating: {separating[0]} \n')

        model = pyopt.Model()
        model.setPresolve(presolve[1])
        model.setHeuristics(heuristic[1])
        model.setSeparating(separating[1])

        if args.nodry:
            model.writeParams(filename=f'../meta_configs/config-{i}.set',
                              comments=False,
                              onlychanged=True)

if __name__ == '__main__':
    main()
