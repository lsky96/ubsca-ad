import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.special
import random


def set_rng_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)


def soft_thresholding(x, mu):
    """Elementwise soft thresholding."""
    res = torch.abs(x) - mu
    res[res < 0] = 0
    soft_x = torch.sign(x) * res
    return soft_x


class soft_thresholding_contgrad_cls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mu):
        ctx.save_for_backward(x, mu)
        return soft_thresholding(x, mu)

    @staticmethod
    def backward(ctx, grad_xout):
        x, mu = ctx.saved_tensors
        res = torch.abs(x) - mu
        nonzero = res > 0

        grad_x = torch.zeros_like(grad_xout)
        grad_x[nonzero] = grad_xout[nonzero]

        grad_mu = torch.zeros_like(grad_xout)
        grad_mu[nonzero] = -torch.sign(x[nonzero])
        grad_mu[~nonzero] = -(x / mu)[~nonzero] / 100
        grad_mu2 = grad_mu * _broadcast_shape_sum(grad_mu, grad_xout)

        return grad_x, grad_mu2


def _broadcast_shape_sum(small, big):
    """Sums all dimensions of big which small would need to broadcast to. Does not check validity of broadcast."""
    dims = []
    for idx in range(1, len(small.shape)+1):
        if small.shape[-idx] != big.shape[-idx]:
            dims.append(-idx)
    for idx in range(len(small.shape)+1, len(big.shape)+1):
        dims.append(-idx)

    if len(dims) > 0:
        big = torch.sum(big, dim=dims)
    big = big.reshape(small.shape)

    return big


def soft_thresholding_contgrad(x, mu):
    return soft_thresholding_contgrad_cls.apply(x, mu)


def l1_norm(x, dim):
    return torch.sum(torch.abs(x), dim=dim)


def frob_norm_sq(x, dim):
    return torch.sum(torch.abs(x).square(), dim=dim)


def min_diff_machine_precision(x):
    """Find some small number to add to still have difference within machine precision (FP32)."""
    fp_exp = torch.frexp(x)[1]
    min_diff = 2.0 ** (fp_exp - 23)
    return min_diff


def adjust_lightness(color, amount=0.5):
    """From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def heavyside_cheby_coeff(degree):
    def chebychev_poly(n):
        assert(n > 1)
        coeff = torch.zeros(n+1, n+1)
        coeff[0, 0] = 1
        coeff[1, 1] = 1
        for i in range(2, n+1):
            coeff[i] = torch.cat((torch.zeros(1), 2*coeff[i-1, :-1])) - coeff[i-2]

        return coeff

    def heavyside(x):
        y = torch.zeros_like(x)
        y[x > 0] = 1
        y[x == 0] = 1/2
        return y

    jj = torch.arange(degree+1)
    in_cos = ((jj + 1/2)*torch.pi / (degree+1)).unsqueeze(-1)
    cheb_interp_coeff = heavyside(torch.cos(in_cos)) * torch.cos(jj * in_cos)
    cheb_interp_coeff = cheb_interp_coeff.sum(dim=0) * 2 / (degree+1)

    cheb_poly_coeff = chebychev_poly(degree)

    hs_poly_coeff = cheb_poly_coeff * cheb_interp_coeff.unsqueeze(-1)
    hs_poly_coeff = hs_poly_coeff.sum(dim=0)  # sum over cheb polys
    hs_poly_coeff[0] -= cheb_interp_coeff[0] / 2

    return hs_poly_coeff


def heavyside_monotonic_coeff(ord):
    temp = np.array([-1, 0, 1])
    qdash = np.array([-1, 0, 1])
    for i in range(ord-1):
        qdash = np.convolve(qdash, temp)
    q = np.concatenate((np.array([0]), qdash / np.arange(1, len(qdash)+1)))
    q1 = q.sum()
    coeff = torch.tensor(q / 2 / q1)
    coeff[0] = 1/2

    return coeff


def exact_auc(scores, anomalies):
    sh = scores.shape
    assert(len(scores.shape) == 4)
    assert(scores.shape[-3:] == anomalies.shape[-3:])
    anomalies = anomalies.abs() > 0

    num_blocks = scores.shape[0] * scores.shape[1]
    scores = scores.abs().flatten(start_dim=-2).reshape(num_blocks, -1)
    anomalies = anomalies.expand(*sh)
    anomalies = anomalies.flatten(start_dim=-2).reshape(num_blocks, -1)

    num_anomalies = anomalies.sum(dim=-1)
    num_non_anomalies = (~anomalies).sum(dim=-1)

    auc = torch.zeros(num_blocks, dtype=torch.float)
    for i in range(num_blocks):
        # print("Block {} of {}".format(i+1, num_blocks))
        comp = scores[i][anomalies[i]].unsqueeze(-1) > scores[i][~anomalies[i]]
        # comp_zero_zero_correction = 0
        comp_equal_score_correction = (scores[i][anomalies[i]].unsqueeze(-1) == scores[i][~anomalies[i]])
        comp_equal_score_correction = comp_equal_score_correction.sum(dim=(-2, -1)) / 2 / num_non_anomalies[i] / num_anomalies[i]
        auc_temp = comp.sum(dim=(-2, -1)) / num_non_anomalies[i] / num_anomalies[i] + comp_equal_score_correction
        auc[i] = auc_temp
    auc = auc.reshape(sh[:2])
    auc = auc.mean(dim=-1)

    return auc


def auc_diff_distr(scores, anomalies, norm=True):
    sh = scores.shape
    assert (len(scores.shape) == 4)
    assert (scores.shape[-3:] == anomalies.shape[-3:])
    anomalies = anomalies.abs() > 0

    num_blocks = scores.shape[0] * scores.shape[1]
    scores = scores.abs().flatten(start_dim=-2).reshape(num_blocks, -1)
    anomalies = anomalies.expand(*sh)
    anomalies = anomalies.flatten(start_dim=-2).reshape(num_blocks, -1)

    scores = scores / scores.max(dim=-1)[0].unsqueeze(-1)
    comps = []
    ests = []
    exs = []
    for i in range(num_blocks):
        print("Block {} of {}".format(i + 1, num_blocks))
        comp = scores[i][anomalies[i]].unsqueeze(-1) - scores[i][~anomalies[i]]
        guilty = (scores[i][anomalies[i]].unsqueeze(-1) > 0) * (scores[i][~anomalies[i]] == 0)
        est = (scores[i][anomalies[i]] > 0).sum() * (scores[i][~anomalies[i]] == 0).sum()
        est = est / (anomalies[i].sum() * (~anomalies[i]).sum())
        exs.append(guilty.sum() / (anomalies[i].sum() * (~anomalies[i]).sum()))

        ests.append(est)
        comp = comp[guilty]
        comps.append(comp.flatten())

    comps = torch.cat(comps)
    ests = torch.stack(ests)
    exs = torch.stack(exs)
    plt.hist(comps, bins="auto")
    plt.show()
    return


def binomial_coefficient(n, k):
    return scipy.special.comb(n, k, exact=False)