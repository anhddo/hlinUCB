from algo import create_simulation, exp_in_range, run_simulation
import pdb
from datetime import datetime
import argparse
import os
import pickle
import numpy as np
from sys import exit
from joblib import Parallel, delayed
from tqdm import tqdm
import math
from pathlib import Path
from multiprocessing import Pool


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    start = datetime.now()

    argParser.add_argument('-M', type=int, default=50,
                           help='number of learners')
    argParser.add_argument('-d', type=int, default=3, help='dimension')
    argParser.add_argument('-T', type=int, default=100, help='rounds')
    argParser.add_argument('-ds', type=int, default=25, help='action set size')
    argParser.add_argument('-base-scale', type=float, default=1, help='base scale for generating theta list')
    argParser.add_argument('-eps', type=float, default=0)
    argParser.add_argument('-fine-search', action='store_true')
    argParser.add_argument('-jobs', type=int, default=4, help='jobs')
    argParser.add_argument('-runs', type=int, default=5, help='runs')
    argParser.add_argument('-pkl', default="", help='pkl file')
    argParser.add_argument('-o', default="", help='folder name')

    args = argParser.parse_args()



    Path(args.o).mkdir(parents=True, exist_ok=True)
            
    train_parameter = {'M': args.M, 'd': args.d, 'T': args.T, 'action_set_size': args.ds, 'runs': args.runs,\
        'initial_epsilon': args.eps, 'base_scale': args.base_scale, 'folder_name': args.o}
    print(train_parameter)
    simulation_params = create_simulation(**train_parameter)
    print("Total simulations: ", len(simulation_params))
    Parallel(n_jobs=args.jobs)(delayed(run_simulation)(p)
                                for p in simulation_params)

    end = datetime.now()
    time_taken = end - start
    print('Total Time: ', time_taken)
