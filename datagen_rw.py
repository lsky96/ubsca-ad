import os
import numpy as np
import torch
import pywt

import utils
import datagen


def _abilene_flows_considered():
    self_flows = np.arange(12) * 13
    # since ATLAng and ATLA-M5 are at the same location, we remove flows to ATLAng to keep the links 0(1) and 1(2)
    atlang_flows = np.concatenate((np.arange(12, 24), np.arange(12)*12 + 1))

    flows_to_remove = np.concatenate((self_flows, atlang_flows))

    flows_to_use = [f for f in range(144) if f not in flows_to_remove]

    return np.array(flows_to_use)


def _abilene_get_routing_matrix(path):
    values_raw = np.loadtxt(path, skiprows=1, delimiter=" ", usecols=(2,3,4))
    values_raw = values_raw.astype(int)

    temp = values_raw[values_raw[:, 0] <= 30]  # only keep internal links

    R = np.zeros((30, 144))
    for i in range(temp.shape[0]):
        e = temp[i, 0] - 1
        f = temp[i, 1] - 1
        R[e, f] = 1

    flow_idx_considered = _abilene_flows_considered()
    R = R[:, flow_idx_considered]

    return R


def _abilene_read_flow(path):
    values_raw = np.loadtxt(path, delimiter=" ", usecols=list(range(1, 721)))
    real_od_idx = np.arange(144) * 5
    real_od = values_raw[:, real_od_idx]

    flow_idx_considered = _abilene_flows_considered()
    flows = real_od[:, flow_idx_considered]
    Z = flows.T

    return Z


def _smoothed_flow(Z):
    # Z.shape = (F, T)
    # follows Kasai 2016
    # w = pywt.Wavelet("db5")   # Daubechies-5
    num_flows = Z.shape[0]
    num_time = Z.shape[1]

    Z_smoothed = np.zeros_like(Z)
    for f in range(num_flows):
        acoeff = pywt.downcoef("a", Z[f, :], "db5", mode="smooth", level=5)
        Z_smoothed[f, :] = pywt.upcoef("a", acoeff, "db5", level=5, take=num_time)

    return Z_smoothed


def abilene_dataset(rpath, fpaths, sampling_param, split=1):
    # fpaths - list
    # sampling_param = {anomaly_distr={amplitude, prob, length}}
    ano_amplitude = sampling_param["anomaly_distr"]["amplitude"]
    ano_prob = sampling_param["anomaly_distr"]["prob"]
    ano_len = sampling_param["anomaly_distr"]["len"]
    observation_prob = sampling_param["observation_prob"]

    R = _abilene_get_routing_matrix(rpath)
    num_directed_edges = R.shape[0]
    num_timesteps = 2016

    batch_size = len(fpaths)
    Z = []
    A_ind = []
    for i in range(batch_size):
        Zraw = _abilene_read_flow(fpaths[i])
        Zraw = Zraw / Zraw.mean()  # normalizing flow
        Zsmooth = _smoothed_flow(Zraw)  # see Kasai 2016
        Znoise_var = (Zraw - Zsmooth).var(axis=-1)  # see Kasai 2016
        Zfinal = np.random.randn(*Zraw.shape) * np.sqrt(Znoise_var)[:, None] + Zsmooth
        Zfinal = torch.tensor(Zfinal.astype(np.float32))
        Z.append(Zfinal)

        # Anomalies
        ano_indicator = torch.rand_like(Zfinal) <= ano_prob
        ano_indicator_idx = torch.nonzero(ano_indicator)
        num_anomalies = ano_indicator_idx.shape[0]
        for i in range(num_anomalies):
            f, t = ano_indicator_idx[i]
            ano_indicator[f, t:min([t+ano_len, num_timesteps])] = 1
        # Afinal = ano_indicator * Zfinal * ano_amplitude
        # Afinal = ano_indicator * ano_amplitude
        A_ind.append(ano_indicator)

    R = torch.tensor(R.astype(np.float32))
    Z = torch.stack(Z)
    A_ind = torch.stack(A_ind)

    if split != 1:
        split_size = num_timesteps // split
        Z = torch.split(Z, split_size, dim=-1)
        Z = torch.cat(Z, dim=0)
        A_ind = torch.split(A_ind, split_size, dim=-1)
        A_ind = torch.cat(A_ind, dim=0)
        # N = torch.split(N, split_size, dim=-1)
        # N = torch.cat(N, dim=0)
        # Omega = torch.split(Omega, split_size, dim=-1)
        # Omega = torch.cat(Omega, dim=0)
        num_timesteps = num_timesteps // split
        batch_size = batch_size * split

    R = R.expand(batch_size, num_directed_edges, Z.shape[-2])
    N = torch.zeros((batch_size, num_directed_edges, num_timesteps))

    """Anomaly amp"""
    if isinstance(ano_amplitude, list):
        # num_anomalies_pos = anomaly_indicator_pos.sum()
        # num_anomalies_neg = anomaly_indicator_neg.sum()
        # ano_amplitude = torch.rand(num_anomalies_neg + num_anomalies_pos) \
        #                  * (anomaly_distr["amplitude"][1] - anomaly_distr["amplitude"][0]) + anomaly_distr["amplitude"][0]
        # A[anomaly_indicator_pos] = ano_amplitude[:num_anomalies_pos]
        # A[anomaly_indicator_neg] = -ano_amplitude[-num_anomalies_neg:]

        ano_amplitude_sampled = torch.rand(batch_size, 1, 1) \
                         * (ano_amplitude[1] - ano_amplitude[0]) + ano_amplitude[0]
        # A[anomaly_indicator_pos] = 1
        # A[anomaly_indicator_neg] = -1
        A = A_ind * ano_amplitude_sampled
    else:
        A = A_ind * ano_amplitude

    """Observations"""
    if isinstance(observation_prob, list):
        obs_prob_temp = torch.rand(batch_size, 1, 1) * (observation_prob[1] - observation_prob[0]) + observation_prob[0]
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= obs_prob_temp
    else:
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= observation_prob

    ss_dict = {"batch_size": batch_size, "sampling_param": sampling_param, "R": R, "Omega": Omega, "Z": Z, "A": A, "N": N}
    scenario_set = datagen.ScenarioSet(**ss_dict)
    return scenario_set


# p1 = os.path.abspath(os.path.join("abilene", "A"))
# _abilene_get_routing_matrix(p1)
#
# p2 = os.path.abspath(os.path.join("abilene", "X01.gz"))
# Z = _abilene_read_flow(p2)

def test():
    sampling_param = {"anomaly_distr": {"amplitude": 1.0, "prob": 0.005, "len": 2}, "observation_prob": 0.9}

    rpath = os.path.abspath(os.path.join("abilene", "A"))
    fpaths = [os.path.abspath(os.path.join("abilene", "X{:02d}".format(i))) for i in range(9, 25)]
    utils.set_rng_seed(0)
    ss = abilene_dataset(rpath, fpaths, sampling_param, split=56)
    # ss = torch.load(os.path.join("scenario_data_paper", "abilene_test.pt"))

    Z = ss["Z"]
    A = ss["A"]

    import reference_algo

    best_lam_log_10iter = torch.tensor(-0.25)
    best_mu_log_10iter = torch.tensor(-1.5)

    lam_bsca = torch.exp(best_lam_log_10iter)
    mu_bsca = torch.exp(best_mu_log_10iter)
    num_iter = 10
    rank = 10
    P_alg, Q_alg, A_alg = reference_algo.bsca_incomplete_meas(ss, lam_bsca, mu_bsca, rank,
                                                                 num_iter=num_iter, init="randsc", return_im_steps=True)
    torch.manual_seed(0)
    obj = reference_algo.network_anomalography_obj(ss, P_alg, Q_alg, A_alg, lam_bsca, mu_bsca)
    auc = utils.exact_auc(A_alg, ss["A"])
    print(obj)
    print(auc)


    import unrolled_bsca

    # rpr = torch.load(os.path.abspath(os.path.join("result_data_paper", "base_ly5_r10.pt")))
    rpr = torch.load(os.path.abspath(os.path.join("result_data_paper", "abilene_ly5_r10.pt")))
    model_kw = rpr["model_kw"]
    model = unrolled_bsca.BSCAUnrolled(**model_kw)
    model.load_state_dict(rpr["model_dict"])
    model.eval()
    torch.manual_seed(0)
    P_alg, Q_alg, A_alg = model(ss)

    auc1 = utils.exact_auc(A_alg, ss["A"])
    print(model.lam_val, model.mu_val)
    print("Auc1", auc1)

    # rpr = torch.load(os.path.abspath(os.path.join("result_data_paper", "base_allstatvar_ly5_r10_paramnw.pt")))
    rpr = torch.load(os.path.abspath(os.path.join("result_data_paper", "abilene2_obs0.9_s28_ly5_r10_paramnw.pt")))
    model_kw = rpr["model_kw"]
    model = unrolled_bsca.BSCAUnrolled(**model_kw)
    model.load_state_dict(rpr["model_dict"])
    model.eval()
    torch.manual_seed(0)
    P_alg, Q_alg, A_alg = model(ss)

    auc2 = utils.exact_auc(A_alg, ss["A"])
    print(model.lam_val.mean(dim=-1), model.mu_val.mean(dim=-1))
    print("Auc2", auc2)

    print("Done")

# test()