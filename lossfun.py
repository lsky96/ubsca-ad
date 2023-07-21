import numpy as np
import torch
import utils
from reference_algo import network_anomalography_obj


NORM_EPS = 1e-6


def unsuper_obj_loss_last_layer(scenario, P, Q, A, lam, mu):
    # P, Q, A are expected to be (iteration, batch_size, *, *)
    assert (len(A.shape) == 4)
    P = P[-1]
    Q = Q[-1]
    A = A[-1]
    return network_anomalography_obj(scenario, P, Q, A, lam, mu)


def super_anomaly_l1loss_last_layer(scenario_dict, A_est):
    return _super_anomaly_l1loss(scenario_dict, A_est, layers=[-1])


def super_anomaly_l1loss_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))  # leaving out init layer
    return _super_anomaly_l1loss(scenario_dict, A_est, layers=layer_list)


def _super_anomaly_l1loss(scenario_dict, A_est, layers=None):
    if layers is None:
        layers = [-1]
    assert (len(A_est.shape) == 4)
    A_true = scenario_dict["A"]
    A_est = A_est[layers]  # remove init
    loss = utils.l1_norm(A_true - A_est, dim=(-2, -1))
    loss = loss.mean(dim=(-2, -1))
    return loss


def super_anomaly_l2loss_last_layer(scenario_dict, A_est):
    return _super_anomaly_l2loss(scenario_dict, A_est, layers=[-1])


def super_anomaly_l2loss_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))  # leaving out init layer
    return _super_anomaly_l2loss(scenario_dict, A_est, layers=layer_list)


def super_anomaly_l2loss_norm_last_layer(scenario_dict, A_est):
    return _super_anomaly_l2loss(scenario_dict, A_est, layers=[-1], normalize=True)


def super_anomaly_l2loss_norm_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))  # leaving out init layer
    return _super_anomaly_l2loss(scenario_dict, A_est, layers=layer_list, normalize=True)


def _super_anomaly_l2loss(scenario_dict, A_est, layers=None, normalize=False):
    if layers is None:
        layers = [-1]
    assert (len(A_est.shape) == 4)
    A_true = scenario_dict["A"]
    A_est = A_est[layers]  # remove init
    diff = A_true - A_est
    if normalize:
        quotient = A_true.clone()
        quotient[A_true == 0] = 1
        diff = diff / quotient
    loss = utils.frob_norm_sq(diff, dim=(-2, -1))
    loss = loss.mean(dim=(-2, -1))
    return loss


def super_anomaly_l05loss_last_layer(scenario_dict, A_est):
    return _super_anomaly_lploss(scenario_dict, A_est, p=0.5, layers=[-1])


def super_anomaly_l05loss_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))  # leaving out init layer
    return _super_anomaly_lploss(scenario_dict, A_est, p=0.5, layers=layer_list)


def _super_anomaly_lploss(scenario_dict, A_est, p=1.0, layers=None):
    EPS = 1e-6
    if layers is None:
        layers = [-1]
    assert (len(A_est.shape) == 4)
    A_true = scenario_dict["A"]
    A_est = A_est[layers]  # remove init
    loss = (((A_true - A_est).abs() + EPS) ** p).sum(dim=(-2, -1))
    loss = loss.mean(dim=(-2, -1))
    return loss


def poly_softauc_cheb_last_layer(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="chebychev")


def poly_softauc_cheb_last_layer_7(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="chebychev", degree=7)


def poly_softauc_cheb_last_layer_9(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="chebychev", degree=9)


def poly_softauc_cheb_last_layer_11(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="chebychev", degree=11)


def poly_softauc_mono_last_layer(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="monotonic")


def poly_softauc_mono_last_layer_7(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="monotonic", degree=7)


def poly_softauc_mono_last_layer_9(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="monotonic", degree=9)


def poly_softauc_mono_last_layer_11(scenario_dict, A_est):
    return _poly_softauc(scenario_dict, A_est, layers=[-1], polynom_type="monotonic", degree=11)


def poly_softauc_cheb_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))
    return _poly_softauc(scenario_dict, A_est, layers=layer_list, polynom_type="chebychev")


def poly_softauc_mono_all_layers(scenario_dict, A_est):
    layer_list = list(range(1, A_est.shape[0]))
    return _poly_softauc(scenario_dict, A_est, layers=layer_list, polynom_type="monotonic")


# def poly_softauc_all_layers(scenario_dict, A_est):
#     layer_list = list(range(1, A_est.shape[0]))
#     return _poly_softauc(scenario_dict, A_est, layers=layer_list)


def _poly_softauc(scenario_dict, A_est, layers=None, polynom_type="monotonic", degree=5, detach_norm=False):
    if layers is None:
        layers = [-1]
    assert(degree % 2 == 1)

    """Coefficent calculation."""
    # degree = 5
    if polynom_type == "monotonic":
        base_coeff = utils.heavyside_monotonic_coeff(degree // 2)
    elif polynom_type == "chebychev":
        base_coeff = utils.heavyside_cheby_coeff(degree)
    else:
        raise ValueError
    k = np.concatenate([np.zeros(1, dtype=int), np.arange(1, degree+1)])
    l = np.arange(degree+1)
    alpha = base_coeff[k].unsqueeze(-1) * utils.binomial_coefficient(k.reshape((-1, 1)), l)\
            * ((-1.0) ** (k.reshape((-1, 1)) - l))

    """AUC"""
    A_true = scenario_dict["A"]
    A_est = A_est[layers].abs()  # remove init, layers must be a list, A_est then has shape (num_layers, batch_size, F, T),
    # also we only need absolute value

    is_anomaly_true = A_true.abs() > 0
    num_anomalies = is_anomaly_true.sum(dim=(-2, -1))
    num_non_anomalies = (A_true.shape[-2] * A_true.shape[-1]) - num_anomalies
    num_anomalies = torch.clamp(num_anomalies, min=1)  # num stability
    num_non_anomalies = torch.clamp(num_non_anomalies, min=1)  # num stability

    if detach_norm:
        A_scale = A_est.detach().max(dim=-1)[0].max(dim=-1)[0] + NORM_EPS  # individual normalization may lead to exploding gradients for the zeros
    else:
        A_scale = A_est.max(dim=-1)[0].max(dim=-1)[0] + NORM_EPS
    A_est_norm = A_est / A_scale.unsqueeze(-1).unsqueeze(-1)  # (layers, batchsize, F, T)

    A_est_norm_exp = A_est_norm ** torch.tensor(l.reshape((-1, 1, 1, 1, 1)))
    A_ONE = (A_est_norm_exp * is_anomaly_true).sum(dim=(-2, -1)) / num_anomalies  # (exp, layers, batch)
    A_ZERO = (A_est_norm_exp * (~is_anomaly_true)).sum(dim=(-2, -1)) / num_non_anomalies  # (exp, layers, batch)

    auc = 0
    for idx in range(1, len(k)):  # constant offset irrelevant
        kk = k[idx]
        temp = alpha[idx, :(kk+1)].unsqueeze(-1).unsqueeze(-1) * A_ONE[:(kk+1)] * (A_ZERO[:(kk+1)].flip(dims=[0]))
        auc += temp.sum(dim=0)

    loss = -auc.mean()

    return loss


def approxauc0_homotopy(scenario_dict, A_est, epoch):
    epoch = torch.tensor(epoch - 500)  # start off
    eps = 50 / torch.clamp(epoch, min=torch.tensor(50), max=torch.tensor(1000))
    return _approx_auc(scenario_dict, A_est, eps, option=0, layers=[-1])


def approxauc2_homotopy(scenario_dict, A_est, epoch):
    epoch = torch.tensor(epoch - 500)  # start off
    beta = torch.clamp(epoch / 10, min=torch.tensor(5), max=torch.tensor(100))
    return _approx_auc(scenario_dict, A_est, beta, option=2, layers=[-1])


def _approx_auc(scenario_dict, A_est, eps, option=0, layers=None, detach_norm=False):
    if layers is None:
        layers = [-1]

    """AUC"""
    A_true = scenario_dict["A"]
    A_est = A_est[layers].abs()  # remove init, layers must be a list, A_est then has shape (num_layers, batch_size, F, T),
    # also we only need absolute value

    is_anomaly_true = A_true.abs() > 0
    # num_anomalies = is_anomaly_true.sum(dim=(-2, -1))
    # num_non_anomalies = (A_true.shape[-2] * A_true.shape[-1]) - num_anomalies
    # num_anomalies = torch.clamp(num_anomalies, min=1)  # num stability
    # num_non_anomalies = torch.clamp(num_non_anomalies, min=1)  # num stability

    if detach_norm:
        A_scale = A_est.detach().max(dim=-1)[0].max(dim=-1)[0] + NORM_EPS  # individual normalization may lead to exploding gradients for the zeros
    else:
        A_scale = A_est.max(dim=-1)[0].max(dim=-1)[0] + NORM_EPS
    A_est_norm = A_est / A_scale.unsqueeze(-1).unsqueeze(-1)  # (layers, batchsize, F, T)

    auc = approx_auc_cls.apply(A_est_norm, is_anomaly_true, eps, option)

    loss = -auc.mean()

    return loss


class approx_auc_cls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, anomalies, eps, option):
        # ctx.save_for_backward(x, mu)
        sh = scores.shape
        assert (len(scores.shape) == 4)
        assert (scores.shape[-3:] == anomalies.shape[-3:])

        num_blocks = scores.shape[0] * scores.shape[1]
        scores = scores.abs().flatten(start_dim=-2).reshape(num_blocks, -1)
        anomalies = anomalies.expand(*sh)
        anomalies = anomalies.flatten(start_dim=-2).reshape(num_blocks, -1)
        if ctx.needs_input_grad[0]:
            grad_scores = torch.zeros_like(scores)

        num_anomalies = anomalies.sum(dim=-1)
        num_non_anomalies = (~anomalies).sum(dim=-1)

        auc = torch.zeros(num_blocks, dtype=torch.float)

        for i in range(num_blocks):
            # print("Block {} of {}".format(i + 1, num_blocks))
            comp = scores[i][anomalies[i]].unsqueeze(-1) - scores[i][~anomalies[i]]
            if option == 0:
                comp_in_interval = torch.logical_and(-eps < comp, comp < eps)
                auc_temp = (comp > eps).sum(dim=(-2, -1)).type(torch.float)
                auc_temp += ((comp + eps) / (2*eps) * comp_in_interval).sum(dim=(-2, -1))
                auc_temp = auc_temp / num_non_anomalies[i] / num_anomalies[i]

                if ctx.needs_input_grad[0]:
                    contribution_ano = comp_in_interval.sum(dim=1)
                    contribution_norm = comp_in_interval.sum(dim=0)
                    grad_scores[i][anomalies[i]] = contribution_ano / num_non_anomalies[i] / num_anomalies[i] / (2*eps)
                    grad_scores[i][~anomalies[i]] = -contribution_norm / num_non_anomalies[i] / num_anomalies[i] / (2*eps)

            elif option == 2:
                comp_sigval = torch.special.expit(eps * comp)
                auc_temp = comp_sigval.sum(dim=(-2, -1))
                auc_temp = auc_temp / num_non_anomalies[i] / num_anomalies[i]

                if ctx.needs_input_grad[0]:
                    comp_sigval_grad = eps * comp_sigval * (1 - comp_sigval)
                    contribution_ano = comp_sigval_grad.sum(dim=1)
                    contribution_norm = comp_sigval_grad.sum(dim=0)
                    grad_scores[i][anomalies[i]] = contribution_ano / num_non_anomalies[i] / num_anomalies[i]
                    grad_scores[i][~anomalies[i]] = -contribution_norm / num_non_anomalies[i] / num_anomalies[i]
            else:
                raise ValueError
            # comp_zero_zero_correction = 0

            auc[i] = auc_temp

        auc = auc.reshape(sh[:2])
        # num_anomalies = num_anomalies.reshape(sh[:2])
        # num_non_anomalies = num_non_anomalies.reshape(sh[:2])

        if ctx.needs_input_grad[0]:
            grad_scores = grad_scores.reshape(*sh)
            ctx.save_for_backward(grad_scores)

        return auc

    @staticmethod
    def backward(ctx, grad_auc):
        grad_scores, = ctx.saved_tensors
        grad_scores = grad_scores * grad_auc.unsqueeze(-1).unsqueeze(-1)

        return grad_scores, None, None, None


def lossfun(data, model_out_P, model_out_Q, model_out_A, reg_param_ll, epoch, option="superl2"):

    if option == "unsuper":
        loss = unsuper_obj_loss_last_layer(data, model_out_P, model_out_Q, model_out_A,
                                                   reg_param_ll["lam"], reg_param_ll["mu"])
    elif option == "superl1":
        loss = super_anomaly_l1loss_last_layer(data, model_out_A)
    elif option == "superl1_all":
        loss = super_anomaly_l1loss_all_layers(data, model_out_A)
    elif option == "superl2":
        loss = super_anomaly_l2loss_last_layer(data, model_out_A)
    elif option == "superl2_all":
        loss = super_anomaly_l2loss_all_layers(data, model_out_A)
    elif option == "superl2norm":
        loss = super_anomaly_l2loss_norm_last_layer(data, model_out_A)
    elif option == "superl2norm_all":
        loss = super_anomaly_l2loss_norm_all_layers(data, model_out_A)
    elif option == "superl05":
        loss = super_anomaly_l05loss_last_layer(data, model_out_A)
    elif option == "superl05_all":
        loss = super_anomaly_l05loss_all_layers(data, model_out_A)

    elif option == "polyauc_mono":
        loss = poly_softauc_mono_last_layer(data, model_out_A)
    elif option == "polyauc_mono7":
        loss = poly_softauc_mono_last_layer_7(data, model_out_A)
    elif option == "polyauc_mono9":
        loss = poly_softauc_mono_last_layer_9(data, model_out_A)
    elif option == "polyauc_mono11":
        loss = poly_softauc_mono_last_layer_11(data, model_out_A)

    elif option == "polyauc_cheb":
        loss = poly_softauc_cheb_last_layer(data, model_out_A)
    elif option == "polyauc_cheb7":
        loss = poly_softauc_cheb_last_layer_7(data, model_out_A)
    elif option == "polyauc_cheb9":
        loss = poly_softauc_cheb_last_layer_9(data, model_out_A)
    elif option == "polyauc_cheb11":
        loss = poly_softauc_cheb_last_layer_11(data, model_out_A)

    elif option == "polyauc_mono_all":
        loss = poly_softauc_mono_all_layers(data, model_out_A)
    elif option == "polyauc_cheb_all":
        loss = poly_softauc_cheb_all_layers(data, model_out_A)

    elif option == "approxauc0_homotopy":
        loss = approxauc0_homotopy(data, model_out_A, epoch)
    elif option == "approxauc2_homotopy":
        loss = approxauc2_homotopy(data, model_out_A, epoch)
    else:
        raise ValueError

    return loss