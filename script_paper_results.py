import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

import paper_results_utils
import utils
import datagen
import datagen_rw
import evalfun
import reference_algo
import paper_results_utils as putils

SCENARIODIR = os.path.abspath("scenario_data_paper")
RESULTDIR = os.path.abspath("result_data_paper")
EXPORTDIR = os.path.abspath("paper_export")

BATCH_SIZE = 500
BASENAME = "base"

if not os.path.isdir(SCENARIODIR):
    os.mkdir(SCENARIODIR)

if not os.path.isdir(RESULTDIR):
    os.mkdir(RESULTDIR)

if not os.path.isdir(EXPORTDIR):
    os.mkdir(EXPORTDIR)


def get_lam_mu_grid(min, max, resolution):
    lam_log_space = torch.linspace(min, max, resolution)
    mu_log_space = torch.linspace(min, max, resolution)
    return lam_log_space, mu_log_space


def gen_base_scenario_param():
    num_timesteps = 100

    graph_param = {
        "num_nodes": 15,
        "num_edges": 26,
        "min_distance": 0.35,
    }
    sampling_param = {
        "flow_distr": {"rank": 5, "scale": 0.5},
        "anomaly_distr": {"prob": 0.005, "amplitude": 1.0},
        "noise_distr": {"variance": 0.05},
        "observation_prob": 0.9,
    }

    param_dict = {"num_timesteps": num_timesteps,
                  "graph_param": graph_param,
                  "sampling_param": sampling_param}

    return param_dict


def part1():
    base_scenario_set_name = BASENAME
    base_param = gen_base_scenario_param()
    putils.generate_data(base_scenario_set_name, SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **base_param)

    # gridsearch
    base_scenario_set = torch.load(os.path.join(SCENARIODIR, base_scenario_set_name + "_test.pt"))

    rank = 10
    lam_log_space, mu_log_space = get_lam_mu_grid(-6, 2, 33)
    grid_search_name = "BSCA_100iter_on_{}".format(base_scenario_set_name)
    putils.gridsearch(base_scenario_set, grid_search_name, RESULTDIR, lam_log_space, mu_log_space, rank, inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bcsa")
    grid_search_name = "BBCD_100iter_on_{}".format(base_scenario_set_name)
    putils.gridsearch(base_scenario_set, grid_search_name, RESULTDIR, lam_log_space, mu_log_space, rank, inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bbcd")
    grid_search_name = "BBCDr_100iter_on_{}".format(base_scenario_set_name)
    putils.gridsearch(base_scenario_set, grid_search_name, RESULTDIR, lam_log_space, mu_log_space, rank,
                      inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bbcd_r")

    grid_search_name = "BSCA_100iter_on_{}".format(base_scenario_set_name)
    putils.show_results_gridsearch(RESULTDIR, grid_search_name, rank, layers_to_show=[5, 10, 100])
    grid_search_name = "BBCD_100iter_on_{}".format(base_scenario_set_name)
    putils.show_results_gridsearch(RESULTDIR, grid_search_name, rank, layers_to_show=[5, 10, 100])
    grid_search_name = "BBCDr_100iter_on_{}".format(base_scenario_set_name)
    putils.show_results_gridsearch(RESULTDIR, grid_search_name, rank, layers_to_show=[5, 10, 100])

    ## Comparison BSCA and BCD

    # From gridsearch experiments
    # base_scenario_set_small = base_scenario_set.return_subset(torch.arange(0, 100))
    best_lam_log_10iter = torch.tensor(-0.25)
    best_mu_log_10iter = torch.tensor(-1.5)

    lam_bsca = torch.exp(best_lam_log_10iter)
    mu_bsca = torch.exp(best_mu_log_10iter)

    lam_bbcd = torch.exp(torch.tensor(-0.25))
    mu_bbcd = torch.exp(torch.tensor(-1.75))

    lam_bbcd_r = torch.exp(torch.tensor(-0.25))  # torch.exp(torch.tensor(-0.25))
    mu_bbcd_r = torch.exp(torch.tensor(-1.75))  # torch.exp(torch.tensor(-1.75))

    base_scenario_set_reduced = base_scenario_set.return_subset(torch.arange(100))  # because otherwise heavy on the RAM
    num_iter = 1000
    itidx = torch.arange(num_iter + 1)
    bsca_path = os.path.join(EXPORTDIR, "bsca_r{}_{}iter.txt".format(rank, num_iter))
    if not os.path.isfile(bsca_path):
        print("BSCA running")
        torch.manual_seed(0)
        s = time.time()
        P_alg1, Q_alg1, A_alg1 = reference_algo.bsca_incomplete_meas(base_scenario_set_reduced, lam_bsca, mu_bsca, rank, num_iter=num_iter, init="randsc", return_im_steps=True)
        e = time.time()
        print("BSCA elapsed time: {}".format(e - s))
        obj = reference_algo.network_anomalography_obj(base_scenario_set_reduced, P_alg1, Q_alg1, A_alg1, lam_bsca, mu_bsca)
        auc = utils.exact_auc(A_alg1, base_scenario_set_reduced["A"])
        result = torch.stack([itidx, obj, auc], dim=-1).numpy()
        np.savetxt(bsca_path, result, header="it\tobj\tauc  ### lam: {}, mu: {}, elapsed_time: {}".format(lam_bsca, mu_bsca, e - s), delimiter="\t")

    bbcd_path = os.path.join(EXPORTDIR, "bbcd_r{}_{}iter.txt".format(rank, num_iter))
    if not os.path.isfile(bbcd_path):
        print("BBCD running")
        torch.manual_seed(0)

        s = time.time()
        P_alg2, Q_alg2, A_alg2 = reference_algo.batch_bcd_incomplete_meas(base_scenario_set_reduced, lam_bbcd, mu_bbcd, rank, num_iter=num_iter, init="randsc", return_im_steps=True)
        e = time.time()
        print("BCD elapsed time: {}".format(e - s))
        obj = reference_algo.network_anomalography_obj(base_scenario_set_reduced, P_alg2, Q_alg2, A_alg2, lam_bbcd, mu_bbcd)
        auc = utils.exact_auc(A_alg2, base_scenario_set_reduced["A"])

        itidx = torch.arange(num_iter+1)
        result = torch.stack([itidx, obj, auc], dim=-1).numpy()
        np.savetxt(bbcd_path, result, header="it\tobj\tauc  ### lam: {}, mu: {}, elapsed_time: {}".format(lam_bbcd, mu_bbcd, e - s), delimiter="\t")

    else:
        print("Comparison results found.")

    bbcd_r_path = os.path.join(EXPORTDIR, "bbcdr_r{}_{}iter.txt".format(rank, num_iter))
    if not os.path.isfile(bbcd_r_path):
        print("BBCDr running")
        torch.manual_seed(0)

        s = time.time()
        P_alg3, Q_alg3, A_alg3 = reference_algo.batch_bcd_incomplete_meas(base_scenario_set_reduced, lam_bbcd_r, mu_bbcd_r, rank, num_iter=num_iter, init="randsc", return_im_steps=True, order="PQA")
        e = time.time()
        print("BCDr elapsed time: {}".format(e - s))
        obj = reference_algo.network_anomalography_obj(base_scenario_set_reduced, P_alg3, Q_alg3, A_alg3, lam_bbcd_r, mu_bbcd_r)
        auc = utils.exact_auc(A_alg3, base_scenario_set_reduced["A"])

        itidx = torch.arange(num_iter+1)
        result = torch.stack([itidx, obj, auc], dim=-1).numpy()
        np.savetxt(bbcd_r_path, result, header="it\tobj\tauc  ### lam: {}, mu: {}, elapsed_time: {}".format(lam_bbcd, mu_bbcd, e - s), delimiter="\t")

    else:
        print("Comparison results found.")

    result_bsca = np.loadtxt(bsca_path, skiprows=1)
    result_bbcd = np.loadtxt(bbcd_path, skiprows=1)
    result_bbcdr = np.loadtxt(bbcd_r_path, skiprows=1)
    idx = result_bsca[:, 0]
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(idx, result_bsca[:, 1], label="BSCA")
    axs[0].plot(idx, result_bbcd[:, 1], label="BBCD")
    axs[0].plot(idx, result_bbcdr[:, 1], label="BBCDr")
    axs[0].set_title("Objective")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Objective Val.")
    axs[0].set_yscale("log")

    axs[1].plot(idx, result_bsca[:, 2], label="BSCA")
    axs[1].plot(idx, result_bbcd[:, 2], label="BBCD")
    axs[1].plot(idx, result_bbcdr[:, 2], label="BBCDr")
    axs[1].set_title("AUC")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("AUC")


def part2():
    base_scenario_set_name = BASENAME
    resolution = 10
    rank = 10
    num_iter = 100
    batch_size = 100  # rough values suffice
    lam_mu_log_init = np.array([-3, -3], dtype=np.float32)  # from previous test for 10
    # lam_mu_log_init = np.array([-0.25, -1.5], dtype=np.float32)  # from previous test for 10
    # lam_mu_log_init = np.array([-2.0, -3.25], dtype=np.float32)  # from previous test for 100

    sweep_name = "anomaly_amplitude"
    print("# " + sweep_name)
    anomaly_amplitude_range = torch.logspace(np.log10(0.2), np.log10(5.0), resolution)

    path_aamp = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_aamp):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in anomaly_amplitude_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["anomaly_distr"]["amplitude"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([anomaly_amplitude_range[:, None], result], axis=-1)
        np.savetxt(path_aamp, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_aamp))


    sweep_name = "anomaly_probability"
    print("# " + sweep_name)
    anomaly_prob_range = torch.logspace(np.log10(0.001), np.log10(0.05), resolution)

    path_aprob = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_aprob):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in anomaly_prob_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["anomaly_distr"]["prob"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([anomaly_prob_range[:, None], result], axis=-1)
        np.savetxt(path_aprob, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_aprob))


    sweep_name = "normal_traffic_scale"
    print("# " + sweep_name)
    normal_traffic_scale_range = torch.logspace(np.log10(0.1), np.log10(10.0), resolution)

    path_scale = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_scale):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in normal_traffic_scale_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["flow_distr"]["scale"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([normal_traffic_scale_range[:, None], result], axis=-1)
        np.savetxt(path_scale, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_scale))


    sweep_name = "normal_traffic_rank"
    print("# " + sweep_name)
    normal_traffic_rank_range = list(range(1, 11))

    path_rank = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_rank):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in normal_traffic_rank_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["flow_distr"]["rank"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([np.array(normal_traffic_rank_range)[:, None], result], axis=-1)
        np.savetxt(path_rank, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_rank))

    ###
    sweep_name = "noise_var"
    print("# " + sweep_name)
    noise_var_range = torch.logspace(np.log10(0.001), np.log10(0.4), resolution)

    path_nvar = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_nvar):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in noise_var_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["noise_distr"]["variance"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([noise_var_range[:, None], result], axis=-1)
        np.savetxt(path_nvar, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_nvar))

    ###
    sweep_name = "observation_probability"
    print("# " + sweep_name)
    observation_probability_range = torch.linspace(0.33, 1.0, resolution)

    path_obsprob = os.path.join(EXPORTDIR, "SWEEP_{}_{}.txt".format(base_scenario_set_name, sweep_name))
    if not os.path.isfile(path_obsprob):
        lam_mu_log = lam_mu_log_init.copy()
        result = []
        for val in observation_probability_range:
            print(sweep_name, val)
            scenario_param = gen_base_scenario_param()
            scenario_param["sampling_param"]["observation_prob"] = val
            lam_mu_log, auc = putils.bsca_param_search(batch_size, scenario_param, rank, num_iter, lam_mu_log)
            result.append(np.concatenate([lam_mu_log, auc]))
        result = np.stack(result, axis=0)
        result = np.concatenate([observation_probability_range[:, None], result], axis=-1)
        np.savetxt(path_obsprob, result, header="value/lam_log/mu_log/auc", delimiter="\t")
    else:
        print("{} already computed.".format(path_obsprob))

    print("## Regression Analysis")
    # investigated value and lam/mu in log-space
    # anomaly amplitudes
    data = np.loadtxt(path_aamp, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0, 1]])
    mu_res = scipy.stats.linregress(data[:, [0, 2]])
    print("# Anomaly amplitude")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    # anomaly probability
    data = np.loadtxt(path_aprob, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0,1]])
    mu_res = scipy.stats.linregress(data[:, [0,2]])
    print("# Anomaly probability")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    # anomaly amplitudes
    data = np.loadtxt(path_scale, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0,1]])
    mu_res = scipy.stats.linregress(data[:, [0,2]])
    print("# Normal traffic scale:")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    # Normal traffic rank
    data = np.loadtxt(path_rank, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0,1]])
    mu_res = scipy.stats.linregress(data[:, [0,2]])
    print("# Normal traffic rank:")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    # Noise variance
    data = np.loadtxt(path_nvar, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0,1]])
    mu_res = scipy.stats.linregress(data[:, [0,2]])
    print("# Noise variance:")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    # Observation probability
    data = np.loadtxt(path_obsprob, skiprows=1)
    # data[:, 0] = np.log(data[:, 0])
    data[:, [1, 2]] = np.exp(data[:, [1, 2]])
    lam_res = scipy.stats.linregress(data[:, [0, 1]])
    mu_res = scipy.stats.linregress(data[:, [0, 2]])
    print("# Observation probability:")
    print("LAMBDA: a1={}, b1={}, Pearson={}".format(lam_res.slope, lam_res.intercept, lam_res.rvalue))
    print("MU: a2={}, b2={}, Pearson={}".format(mu_res.slope, mu_res.intercept, mu_res.rvalue))
    a = mu_res.slope / lam_res.slope
    b = mu_res.intercept - a * lam_res.intercept
    print("MU = f(LAMBDA): a={}, b={}".format(a, b))

    return


def part3():
    print("## USC-RPCA Initial tests ##")
    print("## w/o skip connections ##")
    num_layers_range = list(range(1, 8))
    rank = 10
    base_scenario_set_name = BASENAME
    # base_scenario_set = torch.load(os.path.join(SCENARIODIR, base_scenario_set_name + "_test.pt"))
    for ly in num_layers_range:
        putils.training_parametrized(RESULTDIR, base_scenario_set_name, ly, rank, skip_connections=False, param_nw=False, shared_weights=False)
        putils.analyze_model(RESULTDIR,
                             putils.get_run_name(base_scenario_set_name, ly, rank, skip_connections=False, param_nw=False, shared_weights=False),
                             base_scenario_set_name + "_test", auc_over_layers=True, training_stats=False)

    print("## with skip connections ##")
    num_layers_range = list(range(1, 8))
    base_scenario_set_name = BASENAME
    # base_scenario_set = torch.load(os.path.join(SCENARIODIR, base_scenario_set_name + "_test.pt"))
    for ly in num_layers_range:
        putils.training_parametrized(RESULTDIR, base_scenario_set_name, ly, rank, skip_connections=True, param_nw=False, shared_weights=False)
        putils.analyze_model(RESULTDIR,
                             putils.get_run_name(base_scenario_set_name, ly, rank, skip_connections=True,
                                                 param_nw=False, shared_weights=False),
                             base_scenario_set_name + "_test", auc_over_layers=True, training_stats=False)

    print("## Training Stats for 5 layers no Skip")
    run_name = putils.get_run_name(base_scenario_set_name, 5, 10, skip_connections=False, param_nw=False, shared_weights=False)
    putils.analyze_model(RESULTDIR, run_name, base_scenario_set_name + "_test", auc_over_layers=False, training_stats=True)


def gen_varstat_scenario_param():
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["amplitude"] = [0.2, 5.0]
    scenario_param["sampling_param"]["anomaly_distr"]["prob"] = [0.001, 0.01]
    scenario_param["sampling_param"]["flow_distr"]["scale"] = [0.1, 10.0]
    scenario_param["sampling_param"]["flow_distr"]["rank"] = [1, 10]
    scenario_param["sampling_param"]["noise_distr"]["variance"] = [0.001, 0.4]
    scenario_param["sampling_param"]["observation_prob"] = [0.33, 1.0]
    return scenario_param


def gen_varstat_scenario_set():
    # complete variable stats
    base_set_name = BASENAME
    scenario_param = gen_varstat_scenario_param()
    paper_results_utils.generate_data(base_set_name + "_allstatvar", SCENARIODIR, 500, BATCH_SIZE,
                                      **scenario_param)


def part4():
    num_layers = 5
    rank = 10
    vanilla_num_layers = 10
    base_set_name = BASENAME

    # Anomaly Amplitude param
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["amplitude"] = 0.2
    paper_results_utils.generate_data(base_set_name + "_aamp02", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["amplitude"] = 5.0
    paper_results_utils.generate_data(base_set_name + "_aamp5", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["amplitude"] = [0.2, 5.0]
    scenario_name_aamp = base_set_name + "_aamp02_5"
    paper_results_utils.generate_data(base_set_name + "_aamp02_5", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)


    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_aamp, "num_layers": num_layers, "rank": rank, "skip_connections": False, "param_nw": False,
                                 "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    unroll1 = {"data_name": scenario_name_aamp, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": False,
                     "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll1)
    unroll2 = {"data_name": scenario_name_aamp, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": True,
                     "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_aamp, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)
    unroll4 = {"data_name": scenario_name_aamp, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_aamp02" + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_aamp5" + "_test", auc_over_layers=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_aamp02" + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_aamp5" + "_test", auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_aamp02" + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_aamp5" + "_test", auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_aamp02" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_aamp5" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_aamp02" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_aamp5" + "_test",
                         auc_over_layers=True, training_stats=False)

    # Anomaly probability param
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["prob"] = 0.001
    paper_results_utils.generate_data(base_set_name + "_ap001", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["prob"] = 0.05
    paper_results_utils.generate_data(base_set_name + "_ap05", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                          **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["anomaly_distr"]["prob"] = [0.001, 0.05]
    scenario_name_ap = base_set_name + "_ap001_05"
    paper_results_utils.generate_data(base_set_name + "_ap001_05", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                          **scenario_param)

    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_ap, "num_layers": num_layers, "rank": rank, "skip_connections": False,
                         "param_nw": False,
                         "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    unroll1 = {"data_name": scenario_name_ap, "num_layers": num_layers, "rank": rank, "skip_connections": True,
                   "param_nw": False,
                   "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll1)
    unroll2 = {"data_name": scenario_name_ap, "num_layers": num_layers, "rank": rank, "skip_connections": True,
                   "param_nw": True,
                   "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_ap, "num_layers": num_layers, "rank": rank, "skip_connections": False,
                   "param_nw": False,
                   "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)

    unroll4 = {"data_name": scenario_name_ap, "num_layers": num_layers, "rank": rank, "skip_connections": False,
                   "param_nw": True,
                   "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_ap001" + "_test",
                             auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_ap05" + "_test",
                             auc_over_layers=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_ap001" + "_test",
                             auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_ap05" + "_test",
                             auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_ap001" + "_test",
                             auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_ap05" + "_test",
                             auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_ap001" + "_test",
                             auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_ap05" + "_test",
                             auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_ap001" + "_test",
                             auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_ap05" + "_test",
                             auc_over_layers=True, training_stats=False)

    # Flow scale
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["scale"] = 0.1
    paper_results_utils.generate_data(base_set_name + "_fscale01", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["scale"] = 10.0
    paper_results_utils.generate_data(base_set_name + "_fscale10", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["scale"] = [0.1, 10]
    scenario_name_fscale = base_set_name + "_fscale01_10"
    paper_results_utils.generate_data(scenario_name_fscale, SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_fscale, "num_layers": num_layers, "rank": rank, "skip_connections": False, "param_nw": False,
                                 "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    # unroll1 = {"data_name": scenario_name_fscale, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": False,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll1)
    # unroll2 = {"data_name": scenario_name_fscale, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": True,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_fscale, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)
    unroll4 = {"data_name": scenario_name_fscale, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_fscale01" + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_fscale10" + "_test", auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_fscale01" + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_fscale2" + "_test", auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_fscale01" + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_fscale2" + "_test", auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_fscale01" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_fscale10" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_fscale01" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_fscale10" + "_test",
                         auc_over_layers=True, training_stats=False)

    # Flow rank
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["rank"] = 1
    paper_results_utils.generate_data(base_set_name + "_r1", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["rank"] = 10
    paper_results_utils.generate_data(base_set_name + "_r10", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["flow_distr"]["rank"] = [1, 10]
    scenario_name_rank = base_set_name + "_r1_10"
    paper_results_utils.generate_data(scenario_name_rank, SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_rank, "num_layers": num_layers, "rank": rank, "skip_connections": False, "param_nw": False,
                                 "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    # unroll1 = {"data_name": scenario_name_rank, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": False,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll1)
    # unroll2 = {"data_name": scenario_name_rank, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": True,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_rank, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)
    unroll4 = {"data_name": scenario_name_rank, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_r1" + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_r10" + "_test", auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_r1" + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_r10" + "_test", auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_r1" + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_r10" + "_test", auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_r1" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_r10" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_r1" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_r10" + "_test",
                         auc_over_layers=True, training_stats=False)

    ## Noise param
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["noise_distr"]["variance"] = 0.001
    paper_results_utils.generate_data(base_set_name + "_n001", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["noise_distr"]["variance"] = 0.5
    paper_results_utils.generate_data(base_set_name + "_n4", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["noise_distr"]["variance"] = [0.001, 0.5]
    scenario_name_noise = base_set_name + "_n001_4"
    paper_results_utils.generate_data(scenario_name_noise, SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_noise, "num_layers": num_layers, "rank": rank,
                     "skip_connections": False, "param_nw": False,
                     "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    unroll1 = {"data_name": scenario_name_noise, "num_layers": num_layers, "rank": rank, "skip_connections": True,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll1)
    unroll2 = {"data_name": scenario_name_noise, "num_layers": num_layers, "rank": rank, "skip_connections": True,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_noise, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)
    unroll4 = {"data_name": scenario_name_noise, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_n001" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_n4" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_n001" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_n4" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_n001" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_n4" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_n001" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_n4" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_n001" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_n4" + "_test",
                         auc_over_layers=True, training_stats=False)

    ## Observation prob
    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["observation_prob"] = 0.33
    paper_results_utils.generate_data(base_set_name + "_op033", SCENARIODIR, BATCH_SIZE, BATCH_SIZE, **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["observation_prob"] = 1.0
    paper_results_utils.generate_data(base_set_name + "_op1", SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    scenario_param = gen_base_scenario_param()
    scenario_param["sampling_param"]["observation_prob"] = [0.33, 1.0]
    scenario_name_obsprob = base_set_name + "_op033_1"
    paper_results_utils.generate_data(scenario_name_obsprob, SCENARIODIR, BATCH_SIZE, BATCH_SIZE,
                                      **scenario_param)

    # Vanilla (surrogate for normal BSCA)
    vanilla_param = {"data_name": scenario_name_obsprob, "num_layers": num_layers, "rank": rank,
                     "skip_connections": False, "param_nw": False,
                     "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)

    # Learned
    # unroll1 = {"data_name": scenario_name_obsprob, "num_layers": num_layers, "rank": rank, "skip_connections": True,
    #            "param_nw": False,
    #            "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll1)
    # unroll2 = {"data_name": scenario_name_obsprob, "num_layers": num_layers, "rank": rank, "skip_connections": True,
    #            "param_nw": True,
    #            "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": scenario_name_obsprob, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)
    unroll4 = {"data_name": scenario_name_obsprob, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    # Analysis
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_op033" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), base_set_name + "_op1" + "_test",
                         auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_op05" + "_test",
    #                      auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), base_set_name + "_op1" + "_test",
    #                      auc_over_layers=True, training_stats=False)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_op05" + "_test",
    #                      auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), base_set_name + "_op1" + "_test",
    #                      auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_op033" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), base_set_name + "_op1" + "_test",
                         auc_over_layers=True, training_stats=False)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_op033" + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), base_set_name + "_op1" + "_test",
                         auc_over_layers=True, training_stats=False)


def part5():
    num_layers = 5
    rank = 10

    # reduced training data set size
    # best with mixed set...

    varstat_data_name = BASENAME + "_allstatvar"

    # Variable stats
    gen_varstat_scenario_set()
    vanilla_param = {"data_name": varstat_data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False, "param_nw": False,
                                 "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param)
    # Learned
    # unroll1 = {"data_name": varstat_data_name, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": False,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll1)
    # unroll2 = {"data_name": varstat_data_name, "num_layers": num_layers, "rank": rank, "skip_connections": True, "param_nw": True,
    #                  "shared_weights": False}
    # putils.training_parametrized(RESULTDIR, **unroll2)
    unroll3 = {"data_name": varstat_data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3)

    unroll4 = {"data_name": varstat_data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4)

    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), varstat_data_name + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll1), varstat_data_name + "_test", auc_over_layers=True, training_stats=True)
    # putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll2), varstat_data_name + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), varstat_data_name + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), varstat_data_name + "_test", auc_over_layers=True, training_stats=True)

    training_data_name = BASENAME + "_allstatvar_training"
    training_data_path = os.path.join(SCENARIODIR, training_data_name + ".pt")
    training_data = torch.load(training_data_path)

    # training_data_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 1000]
    training_data_sizes = [50, 100, 150, 200, 300, 400]
    for sz in training_data_sizes:
        training_data_reduced_name = training_data_name + "_sz{}".format(sz)
        training_data_reduced_path = os.path.join(SCENARIODIR, training_data_reduced_name + ".pt")
        if not os.path.isfile(training_data_reduced_path):
            training_data_reduced = training_data.return_subset(torch.arange(0, sz))
            torch.save(training_data_reduced, training_data_reduced_path)
        else:
            print("{} already exists.".format(training_data_reduced_path))

        param = {}
        putils.training_parametrized(RESULTDIR, BASENAME + "_allstatvar", num_layers, rank, skip_connections=False, param_nw=True, shared_weights=False, ovl_training_data_name=training_data_reduced_name, fixed_steps_per_epoch=10)
        putils.analyze_model(RESULTDIR, putils.get_run_name(training_data_reduced_name, num_layers, rank,
                                                            skip_connections=False, param_nw=True, shared_weights=False),
                             BASENAME + "_allstatvar" + "_test", auc_over_layers=True, training_stats=True)


def part6():
    # real data
    split = 7
    ano_amplitude = [1.0, 2]
    obs_prob = [0.5, 1]
    sampling_param = {"anomaly_distr": {"amplitude": ano_amplitude, "prob": 0.005, "len": 2}, "observation_prob": obs_prob}
    rpath = os.path.abspath(os.path.join("abilene", "A"))

    data_name = "abilene2_obs{}_s{}_aamp{}".format(obs_prob, split, ano_amplitude)

    # Training data
    train_data_path = os.path.join(SCENARIODIR, data_name + "_training.pt")
    if not os.path.isfile(train_data_path):
        fpaths = [os.path.abspath(os.path.join("abilene", "X{:02d}.gz".format(i))) for i in range(1, 25, 2)]
        utils.set_rng_seed(0)
        ss = datagen_rw.abilene_dataset(rpath, fpaths, sampling_param, split=split)
        torch.save(ss, train_data_path)

    test_data_path = os.path.join(SCENARIODIR, data_name + "_test.pt")
    if not os.path.isfile(test_data_path):
        fpaths = [os.path.abspath(os.path.join("abilene", "X{:02d}.gz".format(i))) for i in range(2, 26, 2)]
        utils.set_rng_seed(0)
        ss = datagen_rw.abilene_dataset(rpath, fpaths, sampling_param, split=split)
        torch.save(ss, test_data_path)

    # Training
    num_layers = 5
    rank = 10

    vanilla_param = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False, "param_nw": False,
                                 "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param, fixed_steps_per_epoch=10)

    # Learned
    unroll3 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3, fixed_steps_per_epoch=10)

    unroll4 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4, fixed_steps_per_epoch=10)

    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), data_name + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), data_name + "_test", auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), data_name + "_test", auc_over_layers=True, training_stats=True)


def gen_big_scenario_set():
    set_name = "big_allstatvar"
    scenario_param = gen_varstat_scenario_param()
    scenario_param["graph_param"]["num_nodes"] = 20
    scenario_param["graph_param"]["num_edges"] = 46
    paper_results_utils.generate_data(set_name, SCENARIODIR, BATCH_SIZE//2, BATCH_SIZE//2,
                                      **scenario_param)
    # reduced batch-size since more flows per scenario


def part7():
    # Generalization to different-sized graph
    num_layers = 5
    rank = 10

    data_name = "big_allstatvar"
    gen_big_scenario_set()

    vanilla_param = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
                     "param_nw": False,
                     "shared_weights": True}
    putils.training_parametrized(RESULTDIR, **vanilla_param, fixed_steps_per_epoch=10)
    unroll3 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll3, fixed_steps_per_epoch=10)

    unroll4 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}
    putils.training_parametrized(RESULTDIR, **unroll4, fixed_steps_per_epoch=10)

    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), data_name + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), data_name + "_test", auc_over_layers=True,
                         training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), data_name + "_test", auc_over_layers=True,
                         training_stats=True)

    # Generalization test
    data_name = "base_allstatvar"

    vanilla_param = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
                     "param_nw": False,
                     "shared_weights": True}
    unroll3 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": False,
               "shared_weights": False}

    unroll4 = {"data_name": data_name, "num_layers": num_layers, "rank": rank, "skip_connections": False,
               "param_nw": True,
               "shared_weights": False}

    data_name = "big_allstatvar"
    putils.analyze_model(RESULTDIR, putils.get_run_name(**vanilla_param), data_name + "_test",
                         auc_over_layers=True, training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll3), data_name + "_test", auc_over_layers=True,
                         training_stats=True)
    putils.analyze_model(RESULTDIR, putils.get_run_name(**unroll4), data_name + "_test", auc_over_layers=True,
                         training_stats=True)


def part8():
    # run_name = "base_allstatvar_ly5_r10_sw"
    # run_name = "base_allstatvar_ly5_r10"
    run_name = "base_allstatvar_ly5_r10_paramnw"
    test_data_name = "base_allstatvar_test"

    report_path = os.path.join(RESULTDIR, run_name + ".pt")
    report = torch.load(report_path)

    model_dict = report["model_dict"]
    model_kw = report["model_kw"]

    import unrolled_bsca
    model = unrolled_bsca.BSCAUnrolled(**model_kw)
    model.load_state_dict(model_dict)
    model.eval()  # IMPORTANT

    test_data_path = os.path.join(SCENARIODIR, test_data_name + ".pt")
    test_data_set = torch.load(test_data_path)
    num_layers = model.num_layers

    P, Q, A = model(test_data_set)
    import evalfun
    eval_dict = evalfun.detector_single_class_auc_approx(test_data_set, A, batch_mean=True)
    pd_roc = eval_dict["pd_roc"]
    pfa_roc = eval_dict["pfa_roc"]
    fig, ax = plt.subplots(1, 1)
    ax.plot(pfa_roc[:, -1], pd_roc[:, -1])
    ax.set_title(run_name)
    plt.show()
    print("Done")


# part1()
# part2()
# part3()
# part4()
# part5()
# part6()
part7()
# part8()
