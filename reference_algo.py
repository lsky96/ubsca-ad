import torch
import utils
import datagen

DEBUG = False
PREC_EPS = 1


def network_anomalography_obj(scenario, P, Q, A, lam, mu, batch_mean=True):
    Y, R, Omega = datagen.nw_scenario_observation(scenario)
    # batch_size = scenario["batch_size"]
    obj = _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=batch_mean)
    return obj


def _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=True):
    obj = utils.frob_norm_sq(Omega * (Y - P @ Q.mT - R @ A), dim=(-2, -1)) / 2
    obj += lam * (utils.frob_norm_sq(P, dim=(-2, -1)) + utils.frob_norm_sq(Q, dim=(-2, -1))) / 2
    obj += mu * utils.l1_norm(A, dim=(-2, -1))

    if batch_mean:
        obj = obj.mean(dim=-1)  # only batch size, iteration dimension is kept

    return obj


def bsca_incomplete_meas(scenario_dict, lam, mu, rank, num_iter=10, return_im_steps=True, init="default", order="PQA"):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
    # Init
    if init == "default":
        P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)
    elif init == "detuneq":
        P, Q, A = _bsca_incomplete_meas_init_deterministic_uneq(Y, R, Omega, rank, alpha=1.0)
    elif init == "randsc":
        # print("sigma=0.1")
        P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1)
    elif init == "randn":
        P, Q, A = _bsca_incomplete_meas_init_randn(Y, R, Omega, rank)
    else:
        raise ValueError
    if return_im_steps:
        P_list = [P]
        Q_list = [Q]
        A_list = [A]

    for i in range(num_iter):
        if DEBUG:
            print("Iteration {}".format(i+1))
        P_new, Q_new, A_new = _bsca_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order=order)
        if return_im_steps:
            # P_list.append(P)
            # Q_list.append(Q)
            # A_list.append(A_new)

            # P_list.append(P_new)
            # Q_list.append(Q)
            # A_list.append(A_new)

            P_list.append(P_new)
            Q_list.append(Q_new)
            A_list.append(A_new)

        P, Q, A = P_new, Q_new, A_new

    if return_im_steps:
        P_list = torch.stack(P_list)
        Q_list = torch.stack(Q_list)
        A_list = torch.stack(A_list)
        return P_list, Q_list, A_list
    else:
        return P, Q, A


def _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank):
    # Init
    batch_size = Y.shape[:-2]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))  # abs of Y is dirty fix against negative values

    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)
    P = torch.ones(*batch_size, num_edges, rank, dtype=torch.float) * pval.unsqueeze(-1).unsqueeze(-1)
    Q = torch.ones(*batch_size, num_time_steps, rank, dtype=torch.float) * qval.unsqueeze(-1).unsqueeze(-1)
    A = torch.zeros(*batch_size, num_flows, num_time_steps, dtype=torch.float)

    return P, Q, A


def _bsca_incomplete_meas_init_deterministic_uneq(Y, R, Omega, rank, alpha=1.0):
    print("DO NOT USE THIS!")
    # Init
    P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)

    scaling = (torch.arange(rank) / (rank-1) - 1/2) * alpha
    P = P * torch.exp(scaling)
    Q = Q * torch.exp(scaling.flip(-1))

    return P, Q, A


def _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1):
    # Init
    P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)

    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))
    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)

    P = P + pval.unsqueeze(-1).unsqueeze(-1)*sigma*torch.randn_like(P)
    Q = Q + qval.unsqueeze(-1).unsqueeze(-1)*sigma*torch.randn_like(Q)

    return P, Q, A


def _bsca_incomplete_meas_init_randn(Y, R, Omega, rank):
    # Init
    batch_size = Y.shape[:-2]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))  # abs of Y is dirty fix against negative values

    # This pretends that P and Q were correlated, leadig to the elements of P@Q.T being chi-squared(rank) with mean rank*pval*qval
    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)
    P = torch.randn(*batch_size, num_edges, rank, dtype=torch.float) * pval.unsqueeze(-1).unsqueeze(-1)
    Q = torch.randn(*batch_size, num_time_steps, rank, dtype=torch.float) * qval.unsqueeze(-1).unsqueeze(-1)
    A = torch.zeros(*batch_size, num_flows, num_time_steps, dtype=torch.float)

    return P, Q, A


def _bsca_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order="PQA"):

    if order == "PQA":
        err = Omega * (Y - (R @ A))
        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam, err)
        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam, err)
        A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)
    elif order == "APQ":
        A_new = _bsca_update_A(Y, R, Omega, P, Q, A, mu)
        err = Omega * (Y - (R @ A_new))
        P_new = _bsca_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A_new, lam, err)
    else:
        raise ValueError

    return P_new, Q_new, A_new


def _bsca_update_P(Y, R, Omega, Q, A, lam, err=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = Q.shape[-1]

    if err is None:
        rhs = (Omega * (Y - (R @ A))) @ Q
    else:
        rhs = err @ Q
    rhs = rhs.unsqueeze(-1)  # (*batch, links, rank, 1)

    lhs = Q.mT.unsqueeze(-3) @ (Omega.unsqueeze(-1) * Q.unsqueeze(-3))
    regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), lam.unsqueeze(-1))
    lhs = lhs + torch.eye(rank) * regularizer.unsqueeze(-1).unsqueeze(-1)
    # try:
    P_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)
    # except:
    #     print("This shouldn't occur anymore")

    return P_new


def _bsca_update_Q(Y, R, Omega, P, A, lam, err=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = P.shape[-1]

    if err is None:
        rhs = (Omega * (Y - R @ A)).mT @ P
    else:
        rhs = err.mT @ P
    rhs = rhs.unsqueeze(-1)  # (*batch, time, rank, 1)

    lhs = P.mT.unsqueeze(-3) @ (Omega.mT.unsqueeze(-1) * P.unsqueeze(-3))
    regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), lam.unsqueeze(-1))
    lhs = lhs + torch.eye(rank) * regularizer.unsqueeze(-1).unsqueeze(-1)

    # try:
    Q_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)  # (*batch, time, rank, 1)
    # except:
        # print("wtf")

    return Q_new


def _bsca_update_A(Y, R, Omega, P, Q, A, mu, err=None, soft_thresh_cont_grad=False, return_gamma=False):
    """All matrices have leading batch dimension.
        err = Omega * (Y - (R @ A))"""
    if err is None:
        full_err = Omega * (Y - (R @ A) - (P @ Q.mT))
    else:
        full_err = Omega * (err - (P @ Q.mT))

    # Direction
    A_scale = (R * R).transpose(-2, -1) @ Omega.type(torch.float)
    A_scale_zero = A_scale == 0
    soft_thresh_args = (A_scale * A + R.mT @ full_err, mu.unsqueeze(-1).unsqueeze(-1))
    if soft_thresh_cont_grad:
        BA_temp = utils.soft_thresholding_contgrad(*soft_thresh_args)
    else:
        BA_temp = utils.soft_thresholding(*soft_thresh_args)

    A_scale_safezero = A_scale + A_scale_zero * 1
    BA = BA_temp / A_scale_safezero
    BA[A_scale_zero] = 0  # set direction to 0 where A does not receive information (no connected link measurements for particular a)

    # Step size
    proj_step = R @ (BA - A)
    denom = Omega * proj_step
    denom = torch.sum(denom.square(), dim=(-2, -1))
    nom1 = - full_err * proj_step
    nom1 = torch.sum(nom1, dim=(-2, -1))
    nom2 = mu * (utils.l1_norm(BA, dim=(-2, -1)) - utils.l1_norm(A, dim=(-2, -1)))
    nom = - nom1 - nom2

    denom_zero = denom == 0
    denom[denom_zero] = 1  # avoiding division by 0
    gamma = nom / denom
    gamma[denom_zero] = 0
    gamma = torch.clamp(gamma, min=0, max=1)
    if torch.any(torch.isnan(gamma)) or torch.any(torch.isnan(BA)):
        raise RuntimeError("Gamma or BA was nan")

    # print("Gamma", gamma.mean())

    # Step
    A_new = gamma.unsqueeze(-1).unsqueeze(-1) * BA + (1 - gamma.unsqueeze(-1).unsqueeze(-1)) * A

    if return_gamma:
        return A_new, gamma
    else:
        return A_new


def batch_bcd_incomplete_meas(scenario_dict, lam, mu, rank, num_iter=10, init="default", return_im_steps=True, order="APQ"):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
    # Init
    if init == "default":
        P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)
    elif init == "randsc":
        # print("sigma=0.1")
        P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1)
    elif init == "randn":
        P, Q, A = _bsca_incomplete_meas_init_randn(Y, R, Omega, rank)
    else:
        raise ValueError
    if return_im_steps:
        P_list = [P]
        Q_list = [Q]
        A_list = [A]

    for _ in range(num_iter):
        P, Q, A = _batch_bcd_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order=order)
        if return_im_steps:
            P_list.append(P)
            Q_list.append(Q)
            A_list.append(A)

    if return_im_steps:
        P_list = torch.stack(P_list)
        Q_list = torch.stack(Q_list)
        A_list = torch.stack(A_list)
        return P_list, Q_list, A_list
    else:
        return P, Q, A


def _batch_bcd_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order="APQ"):
    if order == "APQ":
        err = Omega * (Y - (R @ A))
        A_new = _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err)
        err = Omega * (Y - (R @ A_new))
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A_new, lam, err)

    if order == "APQold":
        err = Omega * (Y - (R @ A))
        A_new = _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err)
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A_new, lam, err)

    elif order == "PQA":
        err = Omega * (Y - (R @ A))
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A, lam, err)
        A_new = _batch_bcd_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)

    return P_new, Q_new, A_new


def _batch_bcd_update_P(Y, R, Omega, Q, A, lam, err=None):
    return _bsca_update_P(Y, R, Omega, Q, A, lam, err=None)


def _batch_bcd_update_Q(Y, R, Omega, P, A, lam, err=None):
    return _bsca_update_Q(Y, R, Omega, P, A, lam, err=None)


def _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err=None):
    """All matrices have leading batch dimension.
        err = Omega * (Y - (R @ A))"""
    if err is None:
        full_err = Omega * (Y - (R @ A) - (P @ Q.mT))
    else:
        full_err = Omega * (err - (P @ Q.mT))

    num_flows = R.shape[-1]
    A_new = torch.zeros_like(A)
    for f in range(num_flows):
        Y_f = Omega * (Y - P @ Q.mT - R[..., :, :f] @ A_new[..., :f, :] - R[..., :, f:] @ A[..., f:, :])
        A_new[..., f, :] = utils.soft_thresholding(R[..., :, f].unsqueeze(-2) @ Y_f, mu).squeeze(-2) \
                           / utils.frob_norm_sq(R[..., :, f], dim=-1).unsqueeze(-1)

    return A_new
