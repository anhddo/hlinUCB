import os
from joblib import Parallel, delayed
from datetime import datetime
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy.linalg as LA
import pdb
import pickle
from sklearn.preprocessing import normalize

norm = np.linalg.norm


def rand_vec(d, a=-1, b=1, norm=True):
    x = (b-a) * np.random.rand(d) + a
    return x / np.linalg.norm(x) if norm else x


def create_action_set(action_set_size, d):
    A = np.random.rand(action_set_size, d)-0.5
    A = normalize(A)
    return A


def sqrt_beta(lambda_, m2, delta, d, n, L):
    return math.sqrt(lambda_) * m2 + math.sqrt(2 * math.log(1/delta) + d * math.log(1 + n * L**2 / d / lambda_))


def sqrt_beta_dissimilarity(lambda_, m2, delta, d, M, n, L, gamma, epsilon):
    return math.sqrt(lambda_) * m2 + math.sqrt(2 * math.log(1/delta) + d * math.log(1 + n * L**2 / d / lambda_)) \
        + d * 1./gamma * epsilon * math.log(1+M * n)


def sqrt_beta_corruption(delta, d, M, t, epsilon, alpha):
    return 1 + math.sqrt(d * math.log(1+M*t/d) + math.log(1/delta)) + alpha * epsilon * M * t
    # return 1 + math.sqrt(d * math.log(M*t/d/delta))


def oful(theta, action_set_size, T):
    lambda_ = 1
    m2 = 1
    L = 1
    d = len(theta)
    cov = np.identity(d) * lambda_
    b = np.zeros(d)
    regret = []
    delta = 1./T

    for t in (range(T)):
        
        action_set = create_action_set(action_set_size, d)
        inv_cov = np.linalg.inv(cov)
        theta_hat = inv_cov @ b
        sqrt_beta_ = sqrt_beta(lambda_, m2, delta, d, t, L)
        # ||x||_V^-1
        weighted_norm = np.sqrt(
            np.diag(action_set @ inv_cov @ action_set.T))
        # weighted_norm = np.array([np.sqrt(a @ inv_cov @ a.T) for a in action_set])
        # <\hat{\theta}, x> + ||x||_V^-1 * sqrt(Beta)
        ucb = action_set @ theta_hat + sqrt_beta_ * weighted_norm
        select_arm = action_set[np.argmax(ucb)]
        reward = np.dot(theta, select_arm) + np.random.normal(0, 1)
        # least square regression
        cov += np.outer(select_arm, select_arm)
        b += reward * select_arm
        best_action = action_set[np.argmax(action_set @ theta)]
        regret.append(np.dot(theta, best_action - select_arm))
    return regret


def independence_run(theta_list, action_set_size, T):
    regret = []
    # for m in tqdm(range(len(theta_list)), desc='independent learners'):
    for m in range(len(theta_list)):
        regret.append(oful(theta_list[m], action_set_size, T))
    return regret



def calc_dissimilarity(theta_list):
    epsilon_list = [[norm(theta1 - theta2)
                     for theta1 in theta_list] for theta2 in theta_list]
    epsilon = np.max(epsilon_list)
    return epsilon


def dis_lin_ucb(theta_list, action_set_size, T):
    # OFUL start from here
    lambda_ = 1
    M = len(theta_list)
    delta = 1.0 / (M*T)
    d = len(theta_list[0])
    regret = []

    cov = [0 for _ in range(M)]
    b = [np.zeros(d) for _ in range(M)]
    cov_sync = lambda_ * np.identity(d)
    b_sync = np.zeros(d)
    D = math.sqrt(T * math.log(M*T)/d / M)
    t_last = 0
    for t in range(1, T):
        regret_t = 0
        for m in range(len(theta_list)):
            action_set = create_action_set(action_set_size, d)
            cov_m = cov_sync + cov[m]
            b_m = b_sync + b[m]
            inv_cov = np.linalg.inv(cov_m)
            theta_hat = inv_cov @ b_m
            sqrt_beta_ = sqrt_beta(lambda_, 1, delta, d, t, 1)
            weighted_norm = np.sqrt(np.diag(action_set @ inv_cov @ action_set.T))
            # weighted_norm = np.array([np.sqrt(a @ inv_cov @ a.T) for a in action_set])
            ucb = action_set @ theta_hat + sqrt_beta_ * weighted_norm
            select_arm_ind = np.argmax(ucb)
            select_arm = action_set[select_arm_ind]
            reward = np.dot(theta_list[m], select_arm) + np.random.normal(0, 1)

            if math.log(LA.det(cov_m + np.outer(select_arm, select_arm))/LA.det(cov_sync)) * (t-t_last) > D:
                cov_sync += sum(cov)
                b_sync += sum(b)
                t_last = t
                cov = [0 for _ in range(M)]
                b = [np.zeros(d) for _ in range(M)]
            cov[m] += np.outer(select_arm, select_arm)
            b[m] += reward * select_arm
            best_action = action_set[np.argmax(action_set @ theta_list[m])]
            regret_t += np.dot(theta_list[m], best_action - select_arm)
        regret.append(regret_t)
    return regret


def hlin_ucb(theta_list, action_set_size, T):
    epsilon = calc_dissimilarity(theta_list)
    lambda_ = 1
    M = len(theta_list)
    d = len(theta_list[0])
    U = np.identity(d) * lambda_
    regret = []

    epsilon += 1e-7  # to avoid division by zero
    # epsilon/=4
    action_set = [None for _ in range(M)]
    cov = [0 for _ in range(M)]
    b = [np.zeros(d) for _ in range(M)]

    cov_i = [lambda_ * np.identity(d) for _ in range(M)]
    b_i = [0 for _ in range(M)]

    tau = min(T, int(1./(epsilon ** 2)))
    delta_2 = 1.0 / (T-tau + 1)
    cov_sync = lambda_ * np.identity(d)
    b_sync = np.zeros(d)
    alpha = (math.sqrt(d)) / (epsilon * M * T)
    D = math.sqrt(T * math.log(M*T)/d / M)
    delta_1 = 1.0 / (M*tau+1)
    t_last = 0
    for t in range(1, T):
        regret_t = 0
        if t == tau:
            cov_sync = lambda_ * np.identity(d)
            b_sync = np.zeros(d)
            cov = [0 for _ in range(M)]
            b = [np.zeros(d) for _ in range(M)]
            # cov = cov_i
            # b = b_i
        for m in range(len(theta_list)):
            action_set = create_action_set(action_set_size, d)
            cov_m = cov_sync + cov[m]
            b_m = b_sync + b[m]
            inv_cov = np.linalg.inv(cov_m)
            theta_hat = inv_cov @ b_m
            sqrt_beta_d = 0
            if t < tau:
                sqrt_beta_d = sqrt_beta_corruption(
                    delta_1, d, M, t, epsilon, alpha)
            else:
                sqrt_beta_d = sqrt_beta(lambda_, 1, delta_2, d, t, 1)
            weighted_norm = np.sqrt(np.diag(action_set @ inv_cov @ action_set.T))
            # weighted_norm = np.array([np.sqrt(a @ inv_cov @ a.T) for a in action_set])
            ucb = action_set @ theta_hat + sqrt_beta_d * weighted_norm
            select_arm_ind = np.argmax(ucb)
            select_arm = action_set[select_arm_ind]
            reward = np.dot(theta_list[m], select_arm) + np.random.normal(0, 1)

            w = np.minimum(
                1.0, alpha / weighted_norm[select_arm_ind]) if t < tau else 1

            if t < tau and math.log(LA.det(cov_m + w * np.outer(select_arm, select_arm))/LA.det(cov_sync)) * (t-t_last) > D:
                cov_sync += sum(cov)
                b_sync += sum(b)
                t_last = t
                cov = [0 for _ in range(M)]
                b = [np.zeros(d) for _ in range(M)]
            cov[m] += w * np.outer(select_arm, select_arm)
            b[m] += w * reward * select_arm

            # if t < tau:
            #     cov_i[m] += np.outer(select_arm, select_arm)
            #     b_i[m] += reward * select_arm

            best_action = action_set[np.argmax(action_set @ theta_list[m])]
            regret_t += np.dot(theta_list[m], best_action - select_arm)
        regret.append(regret_t)
    return (regret, tau)


def create_theta_list(M, d, epsilon, base_scale):
    A = np.random.rand(M, d)-0.5
    A = normalize(A) * epsilon/2

    x = rand_vec(d) * base_scale
    B = np.tile(x, M).reshape(M, d)
    A = A + B
    eps = calc_dissimilarity(A)
    return A, eps


def get_type_training(M, d, T, initial_epsilon, base_scale):
    test_type = ""
    e1 = 1. / math.sqrt(M*T)
    e2 = d / math.sqrt(T)
    param = [M, d, initial_epsilon*1.1, (1-initial_epsilon) * base_scale]
    # Randomly and choose the closet generate theta list
    theta_list_set0 = Parallel(n_jobs=20)(delayed(create_theta_list)(*param) for i in range(1000))
    eps_list = [eps for _, eps in theta_list_set0]
    theta_list_set = [theta_list for theta_list, _ in theta_list_set0]
    abs_eps_list = [abs(eps - initial_epsilon) for eps in eps_list]
    theta_list = theta_list_set[np.argmin(abs_eps_list)]

    epsilon = calc_dissimilarity(theta_list)
    if epsilon < 1./math.sqrt(M * T):
        # print("regime small " , epsilon/e1)
        test_type = "(i)"
    elif epsilon < 1./math.sqrt(T):
        # print("regime in between 1 ",epsilon/e1,",",epsilon/e2)
        test_type = "(ii)"
    elif epsilon < d/math.sqrt(T):
        # print("regime in between 2 ",epsilon/e1,",",epsilon/e2)
        test_type = "(iii)"
    else:
        # print("regime large" , epsilon/e2)
        test_type = "(iv)"
    return test_type, epsilon, theta_list



def create_simulation(M=1, d=1, T=1, action_set_size=50, initial_epsilon=0.0, base_scale=1, folder_name='', runs=1):
    test_type, epsilon, theta_list = get_type_training(
        M, d, T, initial_epsilon, base_scale)
    tau = min(T, math.floor(1/((epsilon+1e-5)**2)))
    regret_runs = []
    simulation_parameter = []
    print("epsilon: ", epsilon)
    for i in range(runs):
        for algorithm in ["ind", "dis", "hlin"]:
            params = {"initial_epsilon": initial_epsilon, "base_scale": base_scale,
                    "action_set_size": action_set_size, "T": T, "M": M, "d": d, "epsilon": epsilon,
                    "regret_runs": regret_runs, "test_type": test_type,
                    "theta_list": theta_list, "tau": tau, "run": i,
                    "algorithm": algorithm, "folder_name": folder_name
                    }
            simulation_parameter.append(params)

    return simulation_parameter

def run_simulation(params):
    start = datetime.now()
    if params["algorithm"] == "ind":
        regret = independence_run(params["theta_list"], params["action_set_size"], params["T"])
        regret = np.sum(regret, axis=0)
    elif params["algorithm"] == "dis":
        regret = dis_lin_ucb(params["theta_list"], params["action_set_size"], params["T"])
    elif params["algorithm"] == "hlin":
        (regret, _) = hlin_ucb(params["theta_list"], params["action_set_size"], params["T"])
    params.update({"regret": regret})
    file_name = "{}_{}_{}".format(params['initial_epsilon'],params["algorithm"], params["run"])
    pickle.dump(params, open("{}/{}.pkl".format(params["folder_name"], file_name), "wb"))

    end = datetime.now()
    time_taken = end - start
    print('Finish task {} in {}'.format(params['run'], time_taken))




def exp_in_range(a, b, n_exp, M, d, T, ds, folder_name, runs=1, max_base=1):
    parameter = []
    for epsilon in np.linspace(a, b, num=n_exp, endpoint=False):
        for base_scale in np.linspace(0, max_base, num=5):
            parameter.append(
                [M, d, T, ds, epsilon, base_scale, folder_name, runs])
            # create_simulation(M, d, T, ds, epsilon, base_scale, folder_name)
    return parameter
