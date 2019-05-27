import os
import multiprocessing
import subprocess
import numpy as np
import shlex
import time
import math
import socket

lock = multiprocessing.Lock()

def work(cmd):
    lock.acquire()
    time.sleep(20)
    lock.release()
    # return subprocess.call(cmd, shell=False)
    return subprocess.call(shlex.split(cmd), shell=False)

if __name__ == '__main__':

    hostname = socket.gethostname()

    agents_knowledge_state = 2

    if hostname == 'DESKTOP-LA8NF7N':  # HOME
        print('DESKTOP-LA8NF7 (HOME)')
        machine_name = 'HOME'
        n_processors = 5
        config_set = 0
    elif hostname == 'ercument-lab':  # LAB-1
        print('ercument-lab (LAB-1)')
        machine_name = 'LAB-1'
        n_processors = 6
        config_set = 1
    elif hostname == 'DESKTOP-8A3QAR8':  # LAB-2
        print('DESKTOP-8A3QAR8 (LAB-2)')
        machine_name = 'LAB-2'
        n_processors = 6
        config_set = 2

    # ==================================================================================================================

    base_command = 'python dqn-ps.py'

    i_command = 0
    commands = []

    seeds_train = [101, 102, 103]

    seeds_ps = [
        [301],        # 0: HOME
        [401],        # 1: LAB-1
        [501],        # 2: LAB-2
    ]

    seeds_test = [
        [301, 302, 303],        # 0: HOME
        [401, 402, 403],        # 1: LAB-1
        [501, 502, 503, 504],   # 2: LAB-2
    ]

    seeds = seeds_test

    # ==================================================================================================================
    # Parameter Sweep

    for seed in seeds[config_set % 10]:
        commands.append(
            base_command \
            + ' ' + '--seed ' + str(seed) \
            + ' ' + '--agents-knowledge-state ' + str(agents_knowledge_state) \
            + ' ' + '--machine-name ' + str(machine_name) \
            + ' ' + '--process-index ' + str(i_command % n_processors)
        )
        i_command += 1

    for thresholds in [(10.0, 3.0, 1.0)]:
        for budget in [1000, 2000, 5000, 10000, 20000, np.inf]:
            for seed in seeds[config_set % 10]:
                commands.append(
                    base_command \
                    + ' ' + '--seed ' + str(seed) \
                    + ' ' + '--use-teaching' \
                    + ' ' + '--agents-knowledge-state ' + str(agents_knowledge_state) \
                    + ' ' + '--budget-ask ' + str(budget) \
                    + ' ' + '--budget-give ' + str(budget) \
                    + ' ' + '--advice-asking-mode ' + str(3) \
                    + ' ' + '--advice-giving-mode ' + str(0) \
                    + ' ' + '--threshold-ask ' + str(thresholds[0]) \
                    + ' ' + '--threshold-give ' + str(thresholds[1]) \
                    + ' ' + '--importance-threshold-give ' + str(thresholds[2]) \
                    + ' ' + '--machine-name ' + str(machine_name) \
                    + ' ' + '--process-index ' + str(i_command % n_processors)
                )
                i_command += 1

    for thresholds in [(10.0, 3.0, 1.0)]:
        for budget in [1000, 2000, 5000, 10000, 20000, np.inf]:
            for seed in seeds[config_set % 10]:
                commands.append(
                    base_command \
                    + ' ' + '--seed ' + str(seed) \
                    + ' ' + '--use-teaching' \
                    + ' ' + '--agents-knowledge-state ' + str(agents_knowledge_state) \
                    + ' ' + '--budget-ask ' + str(budget) \
                    + ' ' + '--budget-give ' + str(budget) \
                    + ' ' + '--advice-asking-mode ' + str(3) \
                    + ' ' + '--advice-giving-mode ' + str(1) \
                    + ' ' + '--threshold-ask ' + str(thresholds[0]) \
                    + ' ' + '--threshold-give ' + str(thresholds[1]) \
                    + ' ' + '--importance-threshold-give ' + str(thresholds[2]) \
                    + ' ' + '--machine-name ' + str(machine_name) \
                    + ' ' + '--process-index ' + str(i_command % n_processors)
                )
                i_command += 1

    # ==================================================================================================================

    print(commands)
    print('There are {} commands.'.format(len(commands)))

    n_cycles = int(math.ceil(len(commands) / n_processors))

    print('There are {} cycles.'.format(n_cycles))

    for i_cycle in range(n_cycles):
        pool = multiprocessing.Pool(processes=n_processors)

        start = (n_processors*i_cycle)
        end = start + n_processors

        print('start and end:', start, end)

        if end > len(commands):
            end = len(commands)

        print('start and end:', start, end)

        print(pool.map(work, commands[(n_processors*i_cycle):(n_processors*i_cycle) + n_processors]))
