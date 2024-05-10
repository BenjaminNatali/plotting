#!/usr/bin/env python
# coding: utf-8
import pdb
import sys
import shutil
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import multiprocessing as mpp
from multiprocessing import Pool
import functools
#from joblib import Parallel, delayed
import re
#from main_checkpoint import experiment_completed
from collections import Counter
import heapq

file = "log.txt"
FONT_SIZE = 30
plt.rcParams.update({'font.size': 14})


def get_core_param(n_ex, methods_list, Path3):
    core_param = []

    for method in os.listdir(Path3):
        if method in methods_list:
            for i in range(1, n_ex + 1):
                core_param.append((i, method))
    if len(core_param) < 1:
        print("There is not any data folders related to the methods")
        sys.exit(1)
    return core_param


def load_data(core_param, Path3, evaluation):
    print("Number " + str(core_param[0]) + " is loading... \n")
    method_Path = os.path.join(Path3, core_param[1])
    loss = []
    if not evaluation:
        with open(os.path.join(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1])),"results", "td_errors_DQN.json")) as f:
            td_errors_DQN = json.load(f)
        td = dict(sorted(td_errors_DQN.items(), key=lambda item: int(item[0]))).items()
        data = list(td)
        loss = np.array([x[1] for x in data])

    if evaluation:
        test1 = [os.path.abspath(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), "evaluation", P)) for P in os.listdir(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), "evaluation"))]
        #os.listdir(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), "evaluation"))
    else:
        test1 = [os.path.abspath(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), P)) for P in os.listdir(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1])))]
        #os.listdir(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1])))

    max_file_size_txt = 0
    max_file_size_json = 0
    training_file = []
    validation_file = ""
    output_file = []
    file_size_txt = []
    file_size_json = []
    for item1 in test1:
        if item1.endswith(".txt"):
            training_file.append(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), item1))
        elif item1.endswith(".json"):
            output_file.append(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), item1))
    actions = []
    previous_actions = []
    rewards = []
    qs = []
    if len(training_file)>1:
        #idx = np.argmin(file_size_txt)
        #training_file.pop(idx)
        dfs = []
        steps = []
        for train_file in training_file:
            nRowsRead = None  # specify 'None' if want to read whole file
            df = pd.read_csv(train_file, delimiter='"info:"', nrows=nRowsRead, index_col=False)
            #pdb.set_trace()
            steps.append(json.loads(df.columns[1])["step"])
            dfs.append(df)
        dfs_sorted = []
        while len(steps)>0:
            idx = np.argmin(steps)
            dfs_sorted.append(dfs.pop(idx))
            steps.pop(idx)

        #idx = np.argmin(file_size_json)
        #output_file.pop(idx)
        output_file.sort()
        _pd = []
        for output in output_file:
            _pd.append(pd.read_json(os.path.join(output), lines=True))
        final_df = []#pd.DataFrame(columns=["unnamed",""])

        for i in range( dfs_sorted[0].shape[1]-1):

            if i<len(_pd[0]):
                for df in dfs_sorted:
                    final_df.append(df.columns[i+1])
                for output in _pd:

                    action = output['policy_batches'][i]['default_policy']['actions'][0]
                    actions.append(action)
                    previous_actions.append(output['policy_batches'][i]['default_policy']['prev_actions'][0])
                    rewards.append(output['policy_batches'][i]['default_policy']['rewards'][0])
                    qs.append(output['policy_batches'][i]['default_policy']['q_values'][0][action])
        final_df=pd.DataFrame(final_df)
       # p=os.path.dirname(training_file[0])
       # if evaluation:
       #     final_df.to_csv(p + '/validation_{}.txt'.format(final_df.shape[0]), index=False, sep='\t')
       # else:
       #     final_df.to_csv(p + '/training_{}.txt'.format(final_df.shape[0]), index=False, sep='\t')

    else:
        #path_txt = os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), validation_file)
        nRowsRead = None  # specify 'None' if want to read whole file
        final_df = pd.read_csv(training_file[0], delimiter='"info:"', nrows=nRowsRead, index_col=False)
        path_output = output_file[0] #os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), output_file)
        if evaluation:
            with open(os.path.join(path_output)) as f:
                for line in f:
                    out = json.loads(line)
                    actions = out['policy_batches']['default_policy']['actions']
                    previous_actions = out['policy_batches']['default_policy']['prev_actions']
                    rewards = out['policy_batches']['default_policy']['rewards']
                    for idx, action in enumerate(actions):
                        qs.append(out['policy_batches']['default_policy']['q_values'][idx][action])
        else:
            with open(os.path.join(path_output)) as f:
                for line in f:
                    out = json.loads(line)
                    action = out['policy_batches']['default_policy']['actions'][0]
                    actions.append(action)
                    previous_actions.append(out['policy_batches']['default_policy']['prev_actions'][0])
                    rewards.append(out['policy_batches']['default_policy']['rewards'][0])
                    qs.append(out['policy_batches']['default_policy']['q_values'][0][action])


    '''if os.path.exists(path_setup):
        with open(path_setup) as f:
            setup = json.load(f)
    else:
        setup = None
    return df, loss, setup, core_param[1]'''
    return final_df, loss, actions, previous_actions, rewards, qs, core_param[1]

def plot_loss(result_folder_dir, scenario_name, losses, existent_methods):
    plt.figure(figsize=(16, 8))
    
    for idx, method in enumerate(existent_methods):
       '''if len(losses[idx][0]) > 1:
            transparency = 1 - 0.2 * idx
            I = []
            equal_length = all(len(ele) == len(losses[idx][0]) for ele in losses[idx])
            if not equal_length:
                length = min([len(losses[idx][i]) for i in range(len(losses[idx]))])
                for id, loss in enumerate(losses[idx]):
                    losses[idx][id] = loss[0:length]
            ave_loss = np.mean(losses[idx], axis=0)
            if n_steps_per_fit > 1:
                ave_loss = np.repeat(ave_loss, n_steps_per_fit)
            for i in range(initial_replay_size, initial_replay_size + len(ave_loss)):
                I.append(i)

            plt.plot(I, ave_loss, label=method, alpha=transparency)'''
       I=[]
       for i in range(len(losses[idx][0])):
           I.append(i)
       plt.plot(I, losses[idx][0], label=method)
    plt.xlabel('steps')
    # naming the y axis
    plt.ylabel('DQN td-error')
    plt.legend()
    #plt.show()
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_DQNloss.pdf', dpi=200)


def getEnergyConsumptionInterval1(interval, step, max_step, EC):
    _interval = min(interval, max_step - step)
    energy_consumption = [EC[i + step] for i in range(_interval)]

    return sum(energy_consumption)


# fixing the limit of latency violation to have a clear plot otherwise we could have violation of 1oooms that
# will let the figure not be clear for latency such as 40ms
def shrink(value, limit):
    if value > limit:
        return limit
    else:
        return value


# function to get the cumulative violation. Moving average
def getViolationInterval1(interval, step, max_step, t_max, lt):
    if step < (max_step // 3):
        tmax = t_max[0]

    elif step >= (max_step // 3) and step < (2 * max_step // 3):
        tmax = t_max[step + 100]

    elif step >= (2 * max_step // 3):
        tmax = t_max[(2 * max_step // 3) + 100]

    count = len([i for i in lt[(step - 1):step + interval] if i > tmax])

    return (count * 100.) / interval


# function to get the cumulative action
def getCumulativeAction(interval, step, max_step, action):
    import pdb
    _interval = min(interval, max_step - step)
    try:
        count_action = [i for i in range(_interval - 1) if
                        action[i + step + 1] != action[i + step] and action[i + step + 1] != 0]
    except:
        pdb.set_trace()

    return len(count_action)


def getScenarioName(path):
    var = ""
    scenario = ""
    setup = ""

    if "Var_Weights" in path:
        scenario = "Var_Weights"
    elif "Var_Tmax" in path:
        scenario = "Var_Tmax"

    if "ConstantLr" in path:
        setup = "ConstantLr"
    else:
        setup = "DecayLr"



    # print("{}{}_{}".format(setup,scenario,var),file=file)

    return "{}_{}".format(setup, scenario)


def countViolation(result_folder_dir, scenario_name, interval, max_step, t_max, latencies, existent_methods):
    counts = []
    plt.figure(figsize=(16, 8))
    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2 * idx
        I = []
        percentage_method = []
        count_method = []
        for i in range(max_step // interval):
            percentage_ave = []
            count_ave = []
            for n_exp in range(len(latencies[idx])):
                percentage_ave.append((len([j for j in latencies[idx][n_exp][i * interval:i * interval + interval] if
                                            j > t_max[i * interval + 1]]) / interval) * 100)
                count_ave.append(len([j for j in latencies[idx][n_exp][i * interval:i * interval + interval] if
                                      j > t_max[i * interval + 1]]))
            percentage_ave_method = np.mean(np.asarray(percentage_ave), 0)
            percentage_method.append(percentage_ave_method)
            count_method.append(np.mean(np.asarray(count_ave), 0))
            I.append(i + 1)
        plt.plot(I, percentage_method, label=method, alpha=transparency)
        counts.append(sum(count_method))
        name = result_folder_dir + "/counts_" + method + ".npy"
        if not os.path.exists(name):
            np.save(name, sum(count_method))
        name = result_folder_dir + "/percentage_ave_" + method + ".npy"
        if not os.path.exists(name):
            np.save(name, percentage_method)
    plt.xlabel('steps/' + str(interval))
    # naming the y axis
    plt.ylabel('Violation (%)')
    plt.legend()
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_violations_comparision.pdf', dpi=200)

    plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-0.5, len(existent_methods) - 0.5)
    width = min(1 - 1 / (len(existent_methods) + 1), 0.1)
    bars = plt.bar(existent_methods, counts, width=width)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .1, str(round((yval / max_step) * 100, 2)) + "%")
    plt.xlabel("Name of methods", fontsize=14)
    plt.ylabel("Total number of violations", fontsize=14)
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_total_violations.pdf', dpi=200)


def plot_metrics_seprately_without_range(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation, cum_action,
                                         r,
                                         qmax, cum_EC, q_diff, qt, method):
    if method == "SARSA":
        color = "orange"
    elif method == "Q":
        color = "blue"
    else:
        color = "green"
    fig, axs = plt.subplots(1, 7, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.3)
    if len(r) > 0:
        plt.subplot(7, 1, 1)
        plt.plot(np.convolve(np.array(r), np.ones(100) / 100., "same"),
                 label=method, color=color)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('reward')

    if len(qmax) > 0:
        plt.subplot(7, 1, 2)
        plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(qmax[mstart:mstep + mstart]), np.ones(1) / 1., "same"),
                 label=method, color=color)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Q_max')

    plt.subplot(7, 1, 3)
    plt.plot(x[mstart:mstep + mstart - 1],
             np.convolve(np.array(cum_violation[mstart:mstep + mstart - 1]), np.ones(1) / 1., "same"), label=method,
             color=color)
    plt.ylabel('Moving average Violation (%)')
    plt.xlabel('Timestep')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(x[mstart:mstep + mstart - 1], cum_action[mstart:mstep + mstart - 1], label=method, color=color)
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average number of actions')

    # giving a title to my graph
    plt.legend()

    plt.subplot(7, 1, 5)
    plt.plot(x[mstart:mstep + mstart - 1], cum_EC[mstart:mstep + mstart - 1], label=method, color=color)
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average Energy consumption')
    plt.legend()

    if len(qt) > 0:
        plt.subplot(7, 1, 6)
        plt.plot(qt, label=method, color=color)
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('q')
        plt.legend()

    if "Double" in method or "DQN" in method and len(q_diff)>0:
        plt.subplot(7, 1, 7)
        plt.plot(x[mstart:mstep + mstart], q_diff[mstart:mstep + mstart], label=method, color=color)
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('differences of qs')
        plt.legend()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_metrics_' + method + '_without_range.pdf', dpi=200)
    print("Metrics of " + method + " plotted and saved successfully!!")


def plot_metrics_seprately(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation, cum_action, r,
                           qmax, cum_EC, q_diff, qt, method, r_range, qmax_range, cum_violation_range, cum_action_range,
                           cum_EC_range, q_diff_range, qt_range):
    if method == "SARSA":
        color = "orange"
    elif method == "Q":
        color = "blue"
    else:
        color = "green"
    fig, axs = plt.subplots(1, 7, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.3)
    if len(r) > 0:
        plt.subplot(7, 1, 1)
        plt.plot(np.convolve(np.array(r), np.ones(100) / 100., "same"),
                 label=method, color=color)
        plt.ylim(r_range[0], r_range[1])
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('reward')

    if len(qmax) > 0:
        plt.subplot(7, 1, 2)
        plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(qmax[mstart:mstep + mstart]), np.ones(1) / 1., "same"),
                 label=method, color=color)
        plt.ylim(qmax_range[0], qmax_range[1])
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Q_max')

    plt.subplot(7, 1, 3)
    plt.plot(x[mstart:mstep + mstart - 1],
             np.convolve(np.array(cum_violation[mstart:mstep + mstart - 1]), np.ones(1) / 1., "same"), label=method,
             color=color)
    plt.ylim(cum_violation_range[0], cum_violation_range[1])
    plt.ylabel('Moving average Violation (%)')
    plt.xlabel('Timestep')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(x[mstart:mstep + mstart - 1], cum_action[mstart:mstep + mstart - 1], label=method, color=color)
    plt.ylim(cum_action_range[0], cum_action_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average number of actions')

    # giving a title to my graph
    plt.legend()

    plt.subplot(7, 1, 5)
    plt.plot(x[mstart:mstep + mstart - 1], cum_EC[mstart:mstep + mstart - 1], label=method, color=color)
    plt.ylim(cum_EC_range[0], cum_EC_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average Energy consumption')
    plt.legend()

    if len(qt) > 0:
        plt.subplot(7, 1, 6)
        plt.plot(qt, label=method, color=color)
        plt.ylim(qt_range[0], qt_range[1])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('q')
        plt.legend()

    if "Double" in method or "DQN" in method and len(q_diff)>0:
        plt.subplot(7, 1, 7)
        plt.plot(x[mstart:mstep + mstart], q_diff[mstart:mstep + mstart], label=method, color=color)
        plt.ylim(q_diff_range[0], q_diff_range[1])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('differences of qs')
        plt.legend()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_metrics_' + method + '.pdf', dpi=200)
    print("Metrics of " + method + " plotted and saved successfully!!")


def plot_metrics(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation, cum_action, r, cum_EC, cum_EC_phone,
                 qt, existent_methods):

    print("Plotting metrics...")
    # fig = plt.figure(figsize=(35,16))
    plt.figure(figsize=(15, 20))

    #fig, axs = plt.subplots(1, 5, figsize=(15, 20))

    plt.subplots_adjust(hspace=0.3)
    # qmax = 0.25 * (q_max + 1) - 1.5
    min_r_methods = []
    max_r_methods = []
    min_qmax_methods = []
    max_qmax_methods = []
    min_violation_methods = []
    max_violation_methods = []
    min_action_methods = []
    max_action_methods = []
    min_EC_methods = []
    max_EC_methods = []
    min_EC_phone_methods = []
    max_EC_phone_methods = []
    min_q_diff_methods = []
    max_q_diff_methods = []
    min_qt_methods = []
    max_qt_methods = []

    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2 * idx
        plt.subplot(6, 1, 1)
        _reward = np.mean(r[idx], 0)
        plt.plot(_reward, label=method, alpha=transparency)

        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('reward')

        plt.subplot(6, 1, 2)

        conv_violation_method = np.convolve(np.array(cum_violation[idx][mstart:mstep + mstart]), np.ones(1) / 1.,
                                            "same")
        # conv_violation_method = cum_violation
        min_violation_methods.append(min(conv_violation_method))
        max_violation_methods.append(max(conv_violation_method))
        plt.plot(x[mstart:mstep + mstart - window_violation], conv_violation_method[mstart:mstep + mstart - window_violation], label=method,
                 alpha=transparency)
        plt.ylabel('Moving average Violation (%)')
        plt.xlabel('Timestep')
        plt.legend()

        plt.subplot(6, 1, 3)
        min_action_methods.append(min(cum_action[idx][mstart:mstep + mstart]))
        max_action_methods.append(max(cum_action[idx][mstart:mstep + mstart]))
        plt.plot(x[mstart:mstep + mstart - window_violation], cum_action[idx][mstart:mstep + mstart - window_violation], label=method,
                 alpha=transparency)
        plt.xlabel('Timestep')
        plt.ylabel('Moving average number of actions')
        plt.legend()

        plt.subplot(6, 1, 4)
        plt.plot(x[mstart:mstep + mstart - window_violation], cum_EC[idx][mstart:mstep + mstart - window_violation], label=method, alpha=transparency)
        min_EC_methods.append(min(cum_EC[idx][mstart:mstep + mstart]))
        max_EC_methods.append(max(cum_EC[idx][mstart:mstep + mstart]))
        # naming the x axis
        plt.xlabel('Timestep')
        plt.ylabel('Moving average EC SEW')
        # naming the x axis
        plt.legend()

        plt.subplot(6, 1, 5)
        plt.plot(x[mstart:mstep + mstart - window_violation], cum_EC_phone[idx][mstart:mstep + mstart - window_violation], label=method, alpha=transparency)
        plt.xlabel('Timestep')
        plt.ylabel('Moving average EC Phone')
        plt.legend()
        if len(qt[idx]) > 0:
            plt.subplot(6, 1, 6)
            _qt = np.mean(qt[idx], 0)
            # conv_qt_method = np.convolve(np.array(qt[idx][mstart:mstep + mstart]), np.ones(1) / 1., "same")
            # min_qt_methods.append(min(conv_qt_method))
            # max_qt_methods.append(max(conv_qt_method))
            min_qt_methods.append(min(qt[idx]))
            max_qt_methods.append(max(qt[idx]))
            plt.plot(_qt, label=method, alpha=transparency)
            plt.legend()
            plt.xlabel('Timestep')
            plt.ylabel('q')



    # function to show the plot
    # plt.show()
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_metrics.pdf', dpi=200)
    '''Min_reward = min(min_r_methods)
    Max_reward = max(max_r_methods)
    if len(min_qmax_methods)>0:
        Min_qmax = min(min_qmax_methods)
        Max_qmax = max(max_qmax_methods)
    Min_cum_violation = min(min_violation_methods)
    Max_cum_violation = max(max_violation_methods)
    Min_cum_action = min(min_action_methods)
    Max_cum_action = max(max_action_methods)
    Min_cum_EC = min(min_EC_methods)
    Max_cum_EC = max(max_EC_methods)
    if len(min_q_diff_methods) > 0:
        Min_q_diff = min(min_q_diff_methods)
        Max_q_diff = max(max_q_diff_methods)
    else:
        Min_q_diff = 0
        Max_q_diff = 0
    if len(min_qt_methods)>0:
        Min_qt = min(min_qt_methods)
        Max_qt = max(min_qt_methods)
    print("Metrics plotted and saved successfully!!")'''

    '''for idx, method in enumerate(existent_methods):
        plot_metrics_seprately(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation[idx], cum_action[idx],
                               r[idx], qmax[idx],
                               cum_EC[idx], q_diff[idx], qt[idx], method, [Min_reward, Max_reward]
                               , [Min_qmax, Max_qmax], [Min_cum_violation, Max_cum_violation],
                               [Min_cum_action, Max_cum_action], [Min_cum_EC, Max_cum_EC], [Min_q_diff, Max_q_diff],
                               [Min_qt, Max_qt])
        plot_metrics_seprately_without_range(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation[idx],
                                             cum_action[idx],
                                             r[idx], qmax[idx], cum_EC[idx], q_diff[idx], qt[idx], method)'''


def plot_intervals_separately(result_folder_dir, scenario_name, x, lt, t_max, method,max_step, plot_name):
    print("Plotting violation for some intervals")
    mstart_list = [0, (max_step // 3) - mstep_interval, (2 * max_step // 3) - mstep_interval, max_step - mstep_interval]

    mstart1 = mstart_list[0]
    mstart2 = mstart_list[1]
    mstart3 = mstart_list[2]
    mstart4 = mstart_list[3]
    mstep = mstep_interval
    if method == "SARSA":
        color = "orange"
    elif method == "Q":
        color = "blue"
    else:
        color = "green"
    fig = plt.figure(figsize=(16, 16))
    plt.subplot(4, 1, 1)
    plt.plot(x[mstart1:mstep + mstart1], lt[mstart1:mstep + mstart1], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart1:mstep + mstart1], t_max[mstart1:mstep + mstart1], label="Max_latency")
    # plt.ylim(latency_range[0], latency_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(x[mstart2:mstep + mstart2], lt[mstart2:mstep + mstart2], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart2:mstep + mstart2], t_max[mstart2:mstep + mstart2], label="Max_latency")
    # plt.ylim(latency_range[0], latency_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(x[mstart3:mstep + mstart3], lt[mstart3:mstep + mstart3], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart3:mstep + mstart3], t_max[mstart3:mstep + mstart3], label="Max_latency")
    # plt.ylim(latency_range[0], latency_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)

    # giving a title to my graph
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(x[mstart4:mstep + mstart4], lt[mstart4:mstep + mstart4], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart4:mstep + mstart4], t_max[mstart4:mstep + mstart4], label="Max_latency")
    # plt.ylim(latency_range[0], latency_range[1])
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)

    # giving a title to my graph
    plt.legend()
    # plt.savefig('workload_trace_taxi_{}'.format(x),format='pdf',dpi=600)
    # function to show the plot
    # plt.show()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_' + plot_name + '_' + method + '.pdf', dpi=200)
    print("Violation of " + method + " for some intervals plotted and saved successfully")


def plot_for_intervals(result_folder_dir, scenario_name, x, lt, t_max, existent_methods,max_step, plot_name):
    print("Plotting violation for some intervals")
    mstart_list = [0, (max_step // 3) - mstep_interval, (2 * max_step // 3) - mstep_interval, max_step - mstep_interval]

    mstart1 = mstart_list[0]
    mstart2 = mstart_list[1]
    mstart3 = mstart_list[2]
    mstart4 = mstart_list[3]
    mstep = mstep_interval
    min_lt_methods = []
    max_lt_methods = []
    fig = plt.figure(figsize=(16, 16))
    t_max_shown = False
    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2 * idx
        min_lt_methods.append(min(lt[idx][mstart:mstep + mstart]))
        max_lt_methods.append(max(lt[idx][mstart:mstep + mstart]))
        plt.subplot(4, 1, 1)
        plt.plot(x[mstart1:mstep + mstart1], lt[idx][mstart1:mstep + mstart1], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart1:mstep + mstart1], t_max[mstart1:mstep + mstart1], label="Max_latency")

        # naming the x axis

        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(x[mstart2:mstep + mstart2], lt[idx][mstart2:mstep + mstart2], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart2:mstep + mstart2], t_max[mstart2:mstep + mstart2], label="Max_latency")
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(x[mstart3:mstep + mstart3], lt[idx][mstart3:mstep + mstart3], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart3:mstep + mstart3], t_max[mstart3:mstep + mstart3], label="Max_latency")
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        # giving a title to my graph
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(x[mstart4:mstep + mstart4], lt[idx][mstart4:mstep + mstart4], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart4:mstep + mstart4], t_max[mstart4:mstep + mstart4], label="Max_latency")
            t_max_shown = True
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)

        # giving a title to my graph
        plt.legend()
    # plt.savefig('workload_trace_taxi_{}'.format(x),format='pdf',dpi=600)
    # function to show the plot
    # plt.show()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_' + plot_name + '.pdf', dpi=200)
    print("Violation for some intervals plotted and saved successfully")
    # Min_latency = min(min_lt_methods)
    # Max_latency = max(max_lt_methods)
    for idx, method in enumerate(existent_methods):
        plot_intervals_separately(result_folder_dir, scenario_name, x, lt[idx], t_max, method,max_step, plot_name)


def action_distribution(result_folder_dir, scenario_name, conf_distribution, existent_methods, setups):
    Color = {0: "r", 1: "b", 2: "g", 3: "c", 4: "m"}
    for method_idx, method in enumerate(existent_methods):
        ############## configs distribution among all steps ###############
        fig = plt.figure(figsize=(16, 10))
        my_counter = Counter(conf_distribution[method_idx])
        all_config = len(my_counter)
        x = []
        y = []
        for i in range(all_config):
            x.append(i + 1)
            y.append(my_counter[i + 1])

        fig = plt.figure(figsize=(16, 10))
        # plt.rcParams.update({'font.size': 18})
        # creating the bar plot
        plt.xticks(x, rotation=0)
        bars = plt.bar(x, y, color='maroon',
                       width=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / max_step, 1)))

        plt.xlabel("Configuration")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")

        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_all_distribution_config_' + method + '.pdf',
                    dpi=300)

        ############## configs distribution for each interval ###############

        fig = plt.figure(figsize=(30, 10))
        setup = setups[method_idx]
        if "Var_Tmax" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Tmax"]
        elif "Var_Weight" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Weight"]
        else:
            print("Error: name_scenario must be 'Var_Tmax' or 'Var_Weight'")
            sys.exit(1)
        inter = max_step // len(variation_list)
        list_interval = list(range(1, max_step, inter))
        start_interval_idx = 0
        lable_list = []
        for idx, interval in enumerate(list_interval):
            x = []
            y = []
            lable_list.append("interval" + str(idx + 1))
            if idx != len(list_interval) - 1:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:list_interval[idx + 1] - 1])
            else:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:max_step])
            for i in range(all_config):
                x.append(i + 1)
                y.append(my_counter_interval[i + 1])
            start_interval_idx = interval
            plt.xticks(x, rotation=0)
            bars = plt.bar([n + (idx * 0.2) for n in x], y, color=Color[idx],
                           width=0.2, label=lable_list[idx])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / inter, 1)))
        plt.margins(x=0.005)
        plt.xlabel("Configurations")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")
        plt.legend()
        plt.savefig(
            result_folder_dir + '/plots_' + scenario_name + '_distribution_config_each_interval_' + method + '.pdf',
            dpi=300)


def action_distribution_large_configs(result_folder_dir, scenario_name, conf_distribution, existent_methods, setups,
                                      common_configs_num):
    Color = {0: "r", 1: "b", 2: "g", 3: "c", 4: "m"}
    for method_idx, method in enumerate(existent_methods):
        ############## configs distribution among all steps ###############
        fig = plt.figure(figsize=(16, 10))
        my_counter = Counter(conf_distribution[method_idx])
        all_config = len(my_counter)
        x = []
        x_tick = []
        y = []

        my_top_counter = my_counter.most_common(common_configs_num)
        i = 0
        for counter in my_top_counter:
            x.append(i)
            x_tick.append(counter[0])
            y.append(counter[1])
            i += 1

        fig = plt.figure(figsize=(16, 10))
        # plt.rcParams.update({'font.size': 18})
        # creating the bar plot
        plt.xticks(ticks=x, labels=x_tick)
        bars = plt.bar(x, y, color='maroon',
                       width=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / max_step, 1)))

        plt.xlabel("Configuration")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")

        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_all_distribution_config_' + method + '.pdf',
                    dpi=300)

        ############## configs distribution for each interval ###############

        fig = plt.figure(figsize=(50, 10))
        setup = setups[method_idx]
        if "Var_Tmax" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Tmax"]
        elif "Var_Weight" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Weight"]
        else:
            print("Error: name_scenario must be 'Var_Tmax' or 'Var_Weight'")
            sys.exit(1)
        inter = max_step // len(variation_list)
        list_interval = list(range(1, max_step, inter))
        start_interval_idx = 0
        lable_list = []
        x_tick = []
        x_total = []
        for idx, interval in enumerate(list_interval):

            lable_list.append("interval" + str(idx + 1))
            if idx != len(list_interval) - 1:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:list_interval[idx + 1] - 1])
            else:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:max_step])

            x = []
            y = []

            my_top_counter = my_counter_interval.most_common(common_configs_num)
            i = 0
            for counter in my_top_counter:
                x.append(i + (common_configs_num * idx))
                x_total.append(i + (common_configs_num * idx))
                x_tick.append(counter[0])
                y.append(counter[1])
                i += 1

            start_interval_idx = interval

            bars = plt.bar(x, y, color=Color[idx],
                           width=0.3, label=lable_list[idx])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / inter, 1)))
        plt.margins(x=0.005)
        plt.xticks(ticks=x_total, labels=x_tick)
        plt.xlabel("Configurations")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")
        plt.legend()
        plt.savefig(
            result_folder_dir + '/plots_' + scenario_name + '_distribution_config_each_interval_' + method + '.pdf',
            dpi=300)


def extract_data_and_plot(data, window_violation, actions, previous_actions, max_step, mstart, mstep, r, path, losses, qt,
                          existent_methods):

    scenario_name = getScenarioName(path)
    result_folder = ""
    for method in existent_methods:
        result_folder = method + "_" + result_folder
    result_folder_dir = directory + "/" + result_folder  # os.getcwd() + "/" + result_folder
    if not os.path.exists(result_folder_dir):
        os.mkdir(result_folder_dir)


    print("Extracting data and plotting...")
    t_max = []
    cum_action = []
    latencies = []
    mean_cum_violations = []
    lt = []
    cum_violation = []
    cum_EC = []
    cum_EC_phone = []
    action = []
    EC = []
    EC_phone = []
    q_diff = []
    conf_distribution = []
    lt_shrinked_one_instance = []
    energy_cons_one_instance = []
    energy_cons_phone_one_instance = []
    setups = []
    # pdb.set_trace()
    number_steps = [max_step]
    for method_idx in range(len(existent_methods)):
        mm=[]
        mm.append(data[method_idx][0].shape[1])
        mm.append(data[method_idx][0].shape[0])
        number_steps.append(max(mm))
        if len(losses[method_idx][0]) > 0:
            plot_loss(result_folder_dir, scenario_name, losses, existent_methods)
    max_step = min(number_steps)-1
    mstep = min(number_steps)-1
    for method_idx in range(len(existent_methods)):
        #rate_5G = []
        #rate_WIFI = []
        #l_cloud = []
        t_max = []
        action_method = []
        conf_distribution_method = []
        x = []
        previous_config = 0
        for i in range(1, max_step + 1):

            x.append(i)
            if evaluation:
                dt_method = data[method_idx][0].columns[i]
            else:
                dt_method = data[method_idx][0][0][i]
            dt_method = dt_method.replace("{None}", "null")
            dt_method = dt_method.replace("None", "null")

            data_json = json.loads('{"info":' + dt_method + '}')

            #rate_5G.append(data_json["info"]["5G"])
            #rate_WIFI.append(data_json["info"]["wifi"])
            #l_cloud.append(data_json["info"]["l_cloud"])
            t_max.append(data_json["info"]["others"]["T_max"])
            act = data_json["info"]["action"]
            keys = list(act.keys())
            key = keys[0]
            act_v = act[key]
            if "config" in keys:
                action_method.append(int(act_v))  # act[key]["config"]
                conf_distribution_method.append(int(act_v))
                previous_config = int(act_v)
            else:
                action_method.append(0)
                conf_distribution_method.append(previous_config)
            '''if actions[method_idx][0][i-1]==previous_actions[method_idx][0][i-1]:
                action_method.append(0)
                conf_distribution_method.append(previous_actions[method_idx][0][i-1])
            else:
                action_method.append(actions[method_idx][0][i-1])  # act[key]["config"]
                conf_distribution_method.append(actions[method_idx][0][i-1])'''

        max_time_to_shrink = max(t_max) + 5
        cum_action_method = [getCumulativeAction(window_violation, t, mstep, action_method) for t in range(mstep)]
        cum_action.append(cum_action_method)
        action.append(action_method)
        conf_distribution.append(conf_distribution_method)
        latencies_method = []
        cum_violation_method = []
        energy_cons_method = []
        energy_cons_phone_method = []
        for t in range(0, len(data[method_idx])):
            # data[t].shape[1]
            latency_exp = []
            energy_cons_exp = []
            energy_cons_phone_exp = []
            for j in range(1, max_step + 1):
                if evaluation:
                    df = data[method_idx][t].columns[j]
                else:
                    df = data[method_idx][t][0][j]
                #df = data[method_idx][t][0][j]
                df = df.replace("{None}", "null")
                df = df.replace("None", "null")
                _data_json = json.loads('{"info":' + df + '}')
                latency_exp.append(_data_json["info"]["others"]["latency"])
                energy_cons_exp.append(_data_json["info"]["others"]["energy"])
                energy_cons_phone_exp.append(_data_json["info"]["others"]["energy_phone"])
            cum_violation_exp = [getViolationInterval1(window_violation, t + 1, max_step, t_max, latency_exp) for t in
                                 range(mstep)]
            latencies_method.append(latency_exp)
            cum_violation_method.append(cum_violation_exp)
            energy_cons_method.append(energy_cons_exp)
            energy_cons_phone_method.append(energy_cons_phone_exp)
        mean_cum_violation_method = np.mean(np.asarray(cum_violation_method), 0)
        mean_cum_violations.append(mean_cum_violation_method)
        latencies.append(latencies_method)
        lt_method_shrinked_one_instance = [shrink(value, max_time_to_shrink) for value in latencies_method[0]]
        lt_shrinked_one_instance.append(lt_method_shrinked_one_instance)
        print("Getting moving average violation...")
        lt_method_shrinked = [[shrink(value, max_time_to_shrink) for value in latencies_method[idx_exp]] for idx_exp in
                              range(len(latencies_method))]
        lt.append(lt_method_shrinked)
        # cum_violation.append([getViolationInterval1(window_violation, t + 1, max_step, t_max, lt_method_shrinked) for t in range(mstep)])

        energy_cons_one_instance.append(energy_cons_method[0])
        energy_cons_phone_one_instance.append(energy_cons_phone_method[0])
        energy_cons_mean = np.mean(np.asarray(energy_cons_method), 0)
        energy_cons_phone_mean = np.mean(np.asarray(energy_cons_phone_method), 0)
        EC.append(energy_cons_mean)
        EC_phone.append(energy_cons_phone_mean)
        print("Getting moving average energy consumption...")
        cum_EC_method = [getEnergyConsumptionInterval1(window_violation, t, mstep, energy_cons_mean) for t in
                         range(mstep)]
        cum_EC.append(cum_EC_method)
        cum_EC_phone_method = [getEnergyConsumptionInterval1(window_violation, t, mstep, energy_cons_phone_mean) for t in range(mstep)]
        cum_EC_phone.append(cum_EC_phone_method)

        name = result_folder_dir + "/cum_EC_" + existent_methods[method_idx] + ".npy"
        if not os.path.exists(name):
            np.save(name, cum_EC_method)
        name = result_folder_dir + "/cum_EC_phone" + existent_methods[method_idx] + ".npy"
        if not os.path.exists(name):
            np.save(name, cum_EC_phone_method)

        name = result_folder_dir + "/mean_cum_violation_" + existent_methods[method_idx] + ".npy"
        if not os.path.exists(name):
            np.save(name, mean_cum_violation_method)

        name = result_folder_dir + "/cum_action_" + existent_methods[method_idx] + ".npy"
        if not os.path.exists(name):
            np.save(name, cum_action_method)

    my_counter = Counter(conf_distribution[0])
    all_config = len(my_counter)
    '''if all_config > common_configs_numbers:
        action_distribution_large_configs(result_folder_dir, scenario_name, conf_distribution, existent_methods, setups,
                                          common_configs_numbers)
    else:
        action_distribution(result_folder_dir, scenario_name, conf_distribution, existent_methods, setups)'''

    plot_metrics(result_folder_dir, scenario_name, mstart, mstep, x, mean_cum_violations, cum_action, r, cum_EC,cum_EC_phone,
                  qt, existent_methods)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 14})
    plot_for_intervals(result_folder_dir, scenario_name, x, lt_shrinked_one_instance, t_max, existent_methods,max_step,
                       'Latency(ms)')
    plot_for_intervals(result_folder_dir, scenario_name, x, conf_distribution, [], existent_methods,max_step, 'ConfigID')
    plot_for_intervals(result_folder_dir, scenario_name, x, energy_cons_one_instance, [], existent_methods,max_step,
                       'Norm-Energy SEW')
    plot_for_intervals(result_folder_dir, scenario_name, x, energy_cons_phone_one_instance, [], existent_methods, max_step,'Norm-Energy phone')

    interval = window_violation
    countViolation(result_folder_dir, scenario_name, interval, mstep, t_max, lt, existent_methods)

    print("Data extraction and plotting done successfully")


def make_all_plots(dir, n_experiments, window_violation, mstart):
    print("Making all the plots")

    for var_folder in os.listdir(dir):  # for each folder with results
        if os.path.isdir(os.path.join(dir, var_folder)):
            Path1 = os.path.join(dir, var_folder)
            for lr_folder in os.listdir(Path1):    # each learning strategy
                if os.path.isdir(os.path.join(Path1, lr_folder)):
                    Path2 = os.path.join(Path1, lr_folder)
                    extended_state = True
                    for Var in os.listdir(Path2):   # var weights and var Tmax
                        if "var_weights" in Var.lower() or "var_tmax" in Var.lower():
                            extended_state = False
                            Var_scenario = Var
                            break
                    if not extended_state:
                        if os.path.isdir(os.path.join(Path2, Var)):
                            Path3 = os.path.join(Path2, Var)
                    else:
                        Path3 = Path2

                    r = []
                    qt = []
                    all_data_files = []
                    all_setups = []
                    core_params = get_core_param(n_experiments, methods, Path3)
                    out = list([load_data(core_params[0], Path3, evaluation)])   # onlly 1 exp
                    #out = Parallel(n_jobs=-1)(
                     #   delayed(load_data)(core_param, Path3) for core_param in core_params)
                    outs = {}
                    for i in out:   # each algo
                        *key, _ = i[-1]
                        outs.setdefault(tuple(key), []).append(i)
                    all_outs = list(outs.values())
                    actions = []
                    previous_actions = []
                    losses = []
                    existent_methods = []
                    for data_method in all_outs:
                        existent_methods.append(data_method[0][6])
                        data_file = []
                        loss = []
                        action = []
                        previous_action = []
                        reward = []
                        q = []
                        for idx, data in enumerate(data_method):
                            method = data[6]
                            data_file.append(data[0])
                            method_Path = os.path.join(Path3, method)
                            loss.append(data[1])
                            action.append(data[2])
                            previous_action.append(data[3])
                            reward.append(data[4])
                            q.append(data[5])
                        all_data_files.append(data_file)
                        losses.append(loss)
                        actions.append(action)
                        previous_actions.append(previous_action)
                        r.append(reward)
                        qt.append(q)
                    max_step = max(data_file[0].shape[0], data_file[0].shape[1]) #en(actions[0][0])
                    mstep = max(data_file[0].shape[0], data_file[0].shape[1]) #len(actions[0][0])
                    #pdb.set_trace()
                    extract_data_and_plot(all_data_files, window_violation, actions, previous_actions, max_step, mstart, mstep,
                                          r, Path3, losses, qt, existent_methods)
    # print("All the plots made and save successfully",file=file)


methods = ["DQN"]
#methods = ["DQN"]
directory = "/home/natali/rllib2/output_data/bn_rayfed/no_fed/09_05_2024_13_43_56/hamta_logs/" #"/Users/hamtasedghani/Downloads/PMAIEDGE-extended_state_dqn/Applications2/app1/logs_valid/hamta"
n_experiments = 1
window_violation = 1000
max_step = 15000
mstart = 0
common_configs_numbers = 5
mstep_interval = 2000
n_steps_per_fit = 1
initial_replay_size = 0
is_completed = True
evaluation=False

make_all_plots(directory, n_experiments, window_violation, mstart)

