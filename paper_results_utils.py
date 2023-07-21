import os
import numpy as np
import torch
import utils
import datagen
import evalfun
import reference_algo
import unrolled_bsca
import training
import scipy
import matplotlib.pyplot as plt

SCENARIODIR = os.path.abspath("scenario_data_paper")
EXPORTDIR = os.path.abspath("paper_export")


def generate_data(name, dir, training_size=0, test_size=0, num_timesteps=None, graph_param=None, sampling_param=None, nflow_version="exp+exp"):
    utils.set_rng_seed(0)
    training_data_path = os.path.join(dir, name + "_training.pt")
    test_data_path = os.path.join(dir, name + "_test.pt")

    if training_size > 0 and (not os.path.isfile(training_data_path)):
        training_data = datagen.generate_synthetic_nw_scenarios(batch_size=training_size, num_timesteps=num_timesteps,
                                                                graph_param=graph_param, sampling_param=sampling_param, nflow_version=nflow_version)

        torch.save(training_data, training_data_path)
        print("Training data saved.")
    else:
        print("Training data already exists.")

    if test_size > 0 and (not os.path.isfile(test_data_path)):
        test_data = datagen.generate_synthetic_nw_scenarios(batch_size=test_size, num_timesteps=num_timesteps,
                                                                graph_param=graph_param, sampling_param=sampling_param, nflow_version=nflow_version)

        torch.save(test_data, test_data_path)
        print("Test data saved.")
    else:
        print("Test data already exists.")


def gridsearch(test_data, result_name, result_dir, lam_log_space, mu_log_space, rank, inv_layers=None, num_iter=100, init="randsc", alg="bsca"):
    # file_name = "gridsearch_ref_r20_ON_r2_10iter"
    if inv_layers is None:
        inv_layers = [10, 100]

    print("## Gridsearch ##{}".format(result_name))
    result_path = os.path.join(result_dir, "gridsearch_r{}_".format(rank) + result_name + ".pt")

    if os.path.isfile(result_path):
        print("Result already exists.")
        return
    # result_path = os.path.join(result_dir, result_name + ".pt")

    test_data = test_data.return_subset(list(range(250)))  # reducing by half to save time

    auc_results = torch.zeros(len(lam_log_space), len(mu_log_space), len(inv_layers), dtype=torch.float)

    for lam_idx in range(len(lam_log_space)):
        for mu_idx in range(len(mu_log_space)):
            print("Lam Idx {} | Mu Idx {}".format(lam_idx, mu_idx))
            lam = torch.exp(lam_log_space[lam_idx])
            mu = torch.exp(mu_log_space[mu_idx])

            torch.manual_seed(0)  # fixing initialization
            if alg == "bsca":
                A_ref = reference_algo.bsca_incomplete_meas(test_data, lam, mu, rank, num_iter, init=init)[2]
            elif alg == "bbcd":
                A_ref = reference_algo.batch_bcd_incomplete_meas(test_data, lam, mu, rank, num_iter, init=init)[2]
            elif alg == "bbcd_r":
                A_ref = reference_algo.batch_bcd_incomplete_meas(test_data, lam, mu, rank, num_iter, init=init, order="PQA")[2]
            else:
                raise ValueError
            A_ref = A_ref[inv_layers]

            stats_ref = evalfun.detector_single_class_auc_approx(test_data, A_ref, batch_mean=True)
            auc_ref = stats_ref["auc"]

            auc_results[lam_idx, mu_idx] = auc_ref

    results = {"auc": auc_results, "rank": rank, "lam_log": lam_log_space, "mu_log": mu_log_space,
               "result_name": result_name, "layers": inv_layers, "init": init}

    torch.save(results, result_path)
    print("Saved results")
    return


def show_results_gridsearch(result_dir, result_name, rank, layers_to_show=None):
    result_path = os.path.join(result_dir, "gridsearch_r{}_".format(rank) + result_name + ".pt")
    print(result_path)
    results = torch.load(result_path)
    auc = results["auc"]
    lam_log_space = results["lam_log"]
    mu_log_space = results["mu_log"]
    layers = results["layers"]
    rank = results["rank"]

    LAM, MU = torch.meshgrid(lam_log_space, mu_log_space)

    zmin = 0.5
    zmax = 1
    if layers_to_show is None:
        layers_to_show = layers
        layers_to_show_idx = list(range(len(layers)))
    else:
        assert(all([e in layers for e in layers_to_show]))
        layers_to_show_idx = [layers.index(e) for e in layers_to_show]

    ## Absolut maximum AUC
    auc = auc.numpy()
    max_idx = np.unravel_index(auc.argmax(), auc.shape)
    lam_log_max = lam_log_space[max_idx[0]]
    mu_log_max = mu_log_space[max_idx[1]]
    print("Absolute best achieved AUC \n Iteration {}: Best AUC: {}, log(lambda)={}, best log(mu)={}".format(layers[max_idx[2]],
                                                                               auc.max(),
                                                                               lam_log_max,
                                                                               mu_log_max))

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array([0] + layers), np.concatenate([np.array([0.5]), auc[max_idx[0], max_idx[1]]]))
    ax.vlines(layers[max_idx[2]], 0.5, 1.0, colors="r")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("AUC")
    ax.set_title("Max AUC run")
    ax.set_ylim([0.5, 1.0])

    fig, axs = plt.subplots(1, len(layers_to_show))
    for i in range(len(layers_to_show)):
        idx = layers_to_show_idx[i]
        if len(layers_to_show) > 1:
            ax_obj = axs[i]
        else:
            ax_obj = axs
        c = ax_obj.pcolormesh(LAM, MU, auc[:, :, idx], cmap="seismic", vmin=zmin, vmax=zmax)
        ax_obj.set_xlabel("log(lambda)")
        ax_obj.set_ylabel("log(mu)")
        auc_temp = auc[:, :, idx]
        ax_obj.set_title("{} Iterations, AUC_max={}".format(layers_to_show[i], auc_temp.max()))
        max_idx = np.unravel_index(auc_temp.argmax(), auc_temp.shape)
        ax_obj.scatter(lam_log_space[max_idx[0]], mu_log_space[max_idx[1]])
        print("Iteration {}: Best AUC: {}, log(lambda)={}, best log(mu)={}".format(layers_to_show[i],
                                                                                auc_temp.max(),
                                                                          lam_log_space[max_idx[0]],
                                                                          mu_log_space[max_idx[1]]))

        export_path = os.path.join(result_dir, "gridsearch_r{}_".format(rank) + result_name + "_iter{}".format(layers_to_show[i]) + ".txt")
        np.savetxt(export_path, _export_for_heatmap(lam_log_space, mu_log_space, auc_temp), delimiter="\t")
    # fig.suptitle("Grid Search Reference Alg. R6 on R6 Data")
    fig.suptitle("Grid Search R{} {}".format(rank, result_name))
    # fig.suptitle("Grid Search Reference Alg. R20 on R2 Data")
    fig.tight_layout()
    fig.colorbar(c, ax=axs)
    plt.show()


    return


def bsca_param_search(batch_size, scenario_param, rank, num_iter, lam0_mu0_log_numpy):
    utils.set_rng_seed(0)
    data = datagen.generate_synthetic_nw_scenarios(batch_size=batch_size, **scenario_param,
                                                   nflow_version="exp+exp")
    lam0_mu0_log = lam0_mu0_log_numpy

    def run_bsca(lam_mu_log):
        lam = torch.tensor(lam_mu_log[0]).exp().type(torch.float32)  # force float32 due to scipy minimize
        mu = torch.tensor(lam_mu_log[1]).exp().type(torch.float32)  # force float32 due to scipy minimize
        torch.manual_seed(0)  # fixing initialization
        A_ref = reference_algo.bsca_incomplete_meas(data, lam, mu, rank, num_iter, init="randsc", return_im_steps=False)[2]
        # A_ref = A_ref[-1]
        auc = evalfun.detector_single_class_auc_approx(data, A_ref, batch_mean=True)["auc"]
        print("Sample", auc, lam, mu)
        return -auc.numpy()

    res = scipy.optimize.minimize(fun=run_bsca, x0=lam0_mu0_log, method="Nelder-Mead", options={"xatol": 0.01})
    assert(res.success)
    lam_mu_log_opt = res.x
    obj = -np.array([res.fun])

    return lam_mu_log_opt, obj


def get_run_name(data_name, num_layers, rank, skip_connections, param_nw, shared_weights):
    run_name = data_name + "_ly{}_r{}".format(num_layers, rank)
    if shared_weights:
        run_name = run_name + "_sw"
    if param_nw:
        run_name = run_name + "_paramnw"
    if skip_connections:
        run_name = run_name + "_skip"
    return run_name


def training_parametrized(result_dir, data_name, num_layers, rank, skip_connections=False, param_nw=False, shared_weights=False, ovl_training_data_name=None, fixed_steps_per_epoch=None):

    if ovl_training_data_name:
        training_data_name = os.path.join(SCENARIODIR, ovl_training_data_name)
    else:
        training_data_name = os.path.join(SCENARIODIR, data_name + "_training")
    test_data_name = os.path.join(SCENARIODIR, data_name + "_test")

    # run_name = data_name + "_ly{}_r{}".format(num_layers, rank)
    # if param_nw:
    #     run_name = run_name + "_param_nw"
    # if skip_connections:
    #     run_name = run_name + "_skip"
    if ovl_training_data_name:
        run_name = get_run_name(ovl_training_data_name, num_layers, rank, skip_connections, param_nw, shared_weights)
    else:
        run_name = get_run_name(data_name, num_layers, rank, skip_connections, param_nw, shared_weights)
    run_path = os.path.join(result_dir, run_name + ".pt")
    if os.path.isfile(run_path):
        print("Run {} already exists.".format(run_path))
        return

    nn_model_class = unrolled_bsca.BSCAUnrolled
    nn_model_kw = {"num_layers": num_layers, "rank": rank, "init": "randsc", "param_nw": param_nw,
                   "shared_weights": shared_weights, "layer_param": {"skip_connections": skip_connections}}
    if not param_nw:
        nn_model_kw["layer_param"]["init_val"] = -3.0

    if not param_nw:
        num_epochs = 2500
        batch_size = 50
        opt_kw = {"lr": 0.02, "weight_decay": 0.0}
        sched_kw = {"milestones": [500, 2250]}  # epoch
    else:
        num_epochs = 2500
        batch_size = 50
        opt_kw = {"lr": 0.001, "weight_decay": 0.01}
        sched_kw = {"milestones": [500, 2250]}  # epoch
        # sched_kw = {"milestones": [2000, 2250]}  # epoch
    report = training.training_simple_new(run_name, training_data_name, test_data_name, nn_model_class, nn_model_kw,
                                      num_epochs, batch_size, opt_kw, sched_kw,
                                      loss_type="approxauc2_homotopy", fixed_steps_per_epoch=fixed_steps_per_epoch)

    torch.save(report, run_path)
    print("Run {} saved.".format(run_path))


def analyze_model(result_dir, run_name, test_data_name, auc_over_layers=True, training_stats=False):
    # run_name = training_data_name + "_ly{}_r{}".format(num_layers, rank)
    report_path = os.path.join(result_dir, run_name + ".pt")
    report = torch.load(report_path)

    model_dict = report["model_dict"]
    model_kw = report["model_kw"]

    model = unrolled_bsca.BSCAUnrolled(**model_kw)
    model.load_state_dict(model_dict)
    model.eval()  # IMPORTANT
    num_layers = model.num_layers

    if auc_over_layers:
        export_path_auclayers = os.path.join(EXPORTDIR, "{}_ON_{}_auc.txt".format(run_name, test_data_name))
        if not os.path.isfile(export_path_auclayers):
            test_data_path = os.path.join(SCENARIODIR, test_data_name + ".pt")
            test_data_set = torch.load(test_data_path)

            with torch.autograd.no_grad():
                torch.manual_seed(0)
                P_model, Q_model, A_model = model(test_data_set)[0:3]

            auc_model = utils.exact_auc(A_model.abs(), test_data_set["A"])
            lam = model.lam_val.numpy()
            mu = model.mu_val.numpy()

            if lam.ndim == 2:
                lam = np.log(lam.mean(axis=-1))
            else:
                lam = np.log(lam)
            if mu.ndim == 2:
                mu = np.log(mu.mean(axis=-1))
            else:
                mu = np.log(mu)

            lam = np.concatenate([np.array([-999.0]), lam])
            mu = np.concatenate([np.array([-999.0]), mu])

            export_data_auc = np.stack([np.arange(num_layers+1), auc_model.numpy(), lam, mu], axis=-1)

            np.savetxt(export_path_auclayers, export_data_auc, delimiter="\t", header="Layer\tAUC\tlam\tmu")
        else:
            print("{} already exists".format(export_path_auclayers))

    if training_stats:
        export_path_tsteps = os.path.join(EXPORTDIR, "{}_tloss.txt".format(run_name))
        if not os.path.isfile(export_path_tsteps):
            training_loss = report["training_loss"].numpy()
            export_tsteps = np.stack([np.arange(len(training_loss)), training_loss], axis=-1)
            np.savetxt(export_path_tsteps, export_tsteps, delimiter="\t", header="Layer\ttraining_data_loss")
        else:
            pass

        export_path_tepochs = os.path.join(EXPORTDIR, "{}_tepochs.txt".format(run_name))
        if not os.path.isfile(export_path_tepochs):
            test_loss = report["test_loss"].numpy()
            anomaly_l2 = report["test_anomaly_l2"].numpy()
            auc = report["test_det_eval"]["auc"].numpy()
            epochs = np.arange(len(test_loss))

            lam = report["reg_param"]["lam"]
            mu = report["reg_param"]["mu"]

            if lam.ndim == 3:
                lam_log = lam.mean(dim=-1).log()
            else:
                lam_log = lam.log()

            if mu.ndim == 3:
                mu_log = mu.mean(dim=-1).log()
            else:
                mu_log = mu.log()

            export_data = [epochs, test_loss, auc, anomaly_l2[:, -1]]
            export_data_header = "epoch\ttest_loss\tauc\tanomaly_l2"
            for l in range(num_layers):
                export_data.append(lam_log[..., l])
                export_data.append(mu_log[..., l])
                export_data_header = export_data_header + "\tlam{}".format(l+1) + "\tmu{}".format(l+1)

            export_data = np.stack(export_data, axis=-1)

            np.savetxt(export_path_tepochs, export_data, delimiter="\t", header=export_data_header)
        else:
            pass


def _export_for_heatmap(x, y, z):
    size = len(x) * len(y)
    assert(size == z.size)
    filemat = []
    for i in range(len(x)):
        for j in range(len(y)):
            filemat.append(np.array([x[i], y[j], z[i, j]]))
    filemat = np.stack(filemat, axis=0)
    return filemat
