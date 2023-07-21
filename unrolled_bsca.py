import torch
import torch.nn as nn

import datagen
import utils
from reference_algo import _bsca_update_P, _bsca_update_Q, _bsca_update_A
from reference_algo import _bsca_incomplete_meas_init_deterministic
from reference_algo import _bsca_incomplete_meas_init_scaled_randn


class BSCAUnrolled(nn.Module):
    lam_val = []
    mu_val = []
    def __init__(self,
                 num_layers,
                 rank,
                 param_nw=False,  # networks for parameters
                 shared_weights=False,
                 layer_param=None,
                 last_layer_unmodified=False,
                 init="default",
                 A_holdoff=0):
        super().__init__()
        if layer_param is None:
            layer_param = {}

        self.num_layers = num_layers
        self.same_weights_across_layers = shared_weights
        self.rank = rank
        self.param_nw = param_nw
        self.layer_param = {}
        self.last_layer_unmodified = last_layer_unmodified
        self.A_holdoff = A_holdoff
        self.init = init
        self.layers = []

        if self.param_nw:
            iter_class = BSCAUnrolledIteration_ParamNW
        else:
            iter_class = BSCAUnrolledIteration

        if not shared_weights:
            for i_layer in range(self.num_layers):
                if i_layer == num_layers - 1 and self.last_layer_unmodified:
                    self.layers.append(iter_class())
                else:
                    self.layers.append(iter_class(**layer_param))
        else:
            self.layers = [iter_class(**layer_param)] * num_layers

        self.layers = nn.ModuleList(self.layers)

    def forward(self, scenario_dict, return_im_steps=True):
        Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
        # Init
        if self.init == "default":
            P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, self.rank)
        elif self.init == "randsc":
            # alpha is chosen heuristically
            P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, self.rank, sigma=0.1)
        else:
            raise ValueError

        if return_im_steps:
            P_list = [P]
            Q_list = [Q]
            A_list = [A]

        self.lam_val = []
        self.mu_val = []

        if self.param_nw:
            param_nw_out = torch.zeros(scenario_dict["batch_size"], self.layers[0].num_out)

        for l in range(self.num_layers):
            if self.param_nw:
                P, Q, A, param_nw_out = self.layers[l](Y, R, Omega, P, Q, A, param_nw_out)
            else:
                P, Q, A = self.layers[l](Y, R, Omega, P, Q, A, A_update=(l >= self.A_holdoff))

            if return_im_steps:
                P_list.append(P)
                Q_list.append(Q)
                A_list.append(A)

            self.lam_val.append(self.layers[l].lam_val)
            self.mu_val.append(self.layers[l].mu_val)

        self.lam_val = torch.stack(self.lam_val)
        self.mu_val = torch.stack(self.mu_val)

        if return_im_steps:
            P_list = torch.stack(P_list)
            Q_list = torch.stack(Q_list)
            A_list = torch.stack(A_list)
            return P_list, Q_list, A_list
        else:
            return P, Q, A


class BSCAUnrolledIteration(nn.Module):
    lam_val = None
    mu_val = None
    def __init__(self,
                 two_lambda=False,
                 soft_thresh_cont_grad=False,
                 A_postproc=None,
                 skip_connections=False,
                 init_val=-5):
        super().__init__()
        self.two_lambda = two_lambda
        self.soft_thresh_cont_grad = soft_thresh_cont_grad
        self.A_postproc = A_postproc
        self.skip_connections = skip_connections

        self.lam_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))
        self.mu_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))

        if self.two_lambda:
            self.lam2_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))

        if self.skip_connections:
            self.skip_P = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
            self.skip_Q = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
            self.skip_A = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
        else:
            raise ValueError

    def forward(self, Y, R, Omega, P, Q, A, A_update=True):
        lam1 = torch.exp(self.lam_log)
        if self.two_lambda:
            lam2 = torch.exp(self.lam2_log)
        else:
            lam2 = lam1
        mu = torch.exp(self.mu_log)

        self.lam_val = lam1.detach().clone()
        self.mu_val = mu.detach().clone()

        err = Omega * (Y - (R @ A))

        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam1, err)
        if self.skip_connections:
            P_new = P + torch.sigmoid(self.skip_P) * (P_new - P)

        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam2, err)
        if self.skip_connections:
            Q_new = Q + torch.sigmoid(self.skip_Q) * (Q_new - Q)

        if A_update:
            A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err,
                                   soft_thresh_cont_grad=self.soft_thresh_cont_grad)
        else:
            A_new = A

        if self.skip_connections:
            A_new = A + torch.sigmoid(self.skip_A) * (A_new - A)

        return P_new, Q_new, A_new

    def get_regularization_parameters(self, clone=False):
        param_dict = {}
        # if self.two_lambda:
        #     lam = (torch.exp(self.lam_log), torch.exp(self.lam2_log))
        # else:
        #     lam = torch.exp(self.lam_log)
        #
        # mu = torch.exp(self.mu_log)
        if clone:
            param_dict["lam"] = torch.exp(self.lam_log).detach().clone()
            param_dict["mu"] = torch.exp(self.mu_log).detach().clone()
            if self.two_lambda:
                param_dict["lam2"] = torch.exp(self.lam2_log).detach().clone()
        else:
            param_dict["lam"] = torch.exp(self.lam_log)
            param_dict["mu"] = torch.exp(self.mu_log)
            if self.two_lambda:
                param_dict["lam2"] = torch.exp(self.lam2_log)

        return param_dict


class BSCAUnrolledIteration_ParamNW(nn.Module):
    lam_val = None
    mu_val = None

    def __init__(self, two_lambda=False):
        super().__init__()
        self.two_lambda = two_lambda
        self.skip_connections = False
        self.num_in_features = 8

        self.nw_out = ["lam", "mu"]

        if self.two_lambda:
            self.nw_out.append("lam2")

        if self.skip_connections:
            self.nw_out.extend(["skip_P", "skip_Q", "skip_A"])

        self.num_out = len(self.nw_out)
        self.num_in_features = self.num_in_features + self.num_out
        self.param_nw = RegParamMLP([self.num_in_features, self.num_in_features, self.num_in_features, self.num_out], batch_norm_input=True)

    def forward(self, Y, R, Omega, P, Q, A, prev_param_nw_out):
        if self.two_lambda:
            raise NotImplementedError
        err = Omega * (Y - (R @ A))
        # num_timesteps = Y.shape[-1]

        feature_vec = _compile_features_improved(Y, R, Omega, P, Q, A, prev_param_nw_out)
        param_nw_out = self.param_nw(feature_vec)
        lam1 = torch.exp(param_nw_out[..., 0])
        self.lam_val = lam1.detach().clone()

        mu = torch.exp(param_nw_out[..., 1])
        self.mu_val = mu.detach().clone()

        lam2 = lam1

        if self.skip_connections:
            skip_P = param_nw_out[..., 2].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim
            skip_Q = param_nw_out[..., 3].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim
            skip_A = param_nw_out[..., 4].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim

        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam1, err)
        if self.skip_connections:
            P_new = P + torch.sigmoid(skip_P) * (P_new - P)

        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam2, err)
        if self.skip_connections:
            Q_new = Q + torch.sigmoid(skip_Q) * (Q_new - Q)

        A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)
        if self.skip_connections:
            A_new = A + torch.sigmoid(skip_A) * (A_new - A)

        return P_new, Q_new, A_new, param_nw_out

    def get_regularization_parameters(self, clone=False, batch_mean=True):
        # Returns default values to not crash rest of the code.
        param_dict = {}
        param_dict["lam"] = self.lam_val.mean(dim=0)
        param_dict["mu"] = self.mu_val.mean(dim=0)
        if self.two_lambda:
            param_dict["lam2"] = torch.tensor(1.0)

        return param_dict


class RegParamMLP(nn.Module):
    def __init__(self, feat_sizes, batch_norm_input=False):
        super().__init__()
        assert(isinstance(feat_sizes, list))
        assert(len(feat_sizes) > 1)  # At least input and output size required
        self.feat_sizes = feat_sizes
        if batch_norm_input:
            self.batch_norm_input = nn.BatchNorm1d(self.feat_sizes[0], affine=False, momentum=0.1)
        else:
            self.batch_norm_input = None
        self.layers = []
        for l in range(len(self.feat_sizes) - 1):
            self.layers.append(nn.Linear(self.feat_sizes[l], self.feat_sizes[l+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        if self.batch_norm_input:
            x = self.batch_norm_input(x)
        for l in range(len(self.layers)):
            x = self.layers[l](x)
            if not l == len(self.layers) - 1:
                x = torch.relu(x)

        return x


def _compile_features_improved(Y, R, Omega, P, Q, A, nw_out):
    # EPS = 1e-4
    datafit = Y - P @ Q.mT - R @ A
    datafit_sq = datafit**2
    datafit_sq_mean, datafit_sq_var = _masked_mean_var(datafit_sq, mask=Omega, dim=(-2, -1))

    p = P.abs()
    p_mean = p.mean(dim=(-2, -1))

    q = Q.abs()
    q_mean = q.mean(dim=(-2, -1))

    a_abs = A.abs()

    full_err = Omega * datafit

    # Direction
    A_scale = (R * R).transpose(-2, -1) @ Omega.type(torch.float)
    A_scale_zero = A_scale == 0
    BA_temp = A_scale * A + R.mT @ full_err
    A_scale_safezero = A_scale + A_scale_zero * 1
    BA = BA_temp / A_scale_safezero
    BA[A_scale_zero] = 0

    BA = BA.abs()

    BA_mean = BA.mean(dim=(-2, -1))
    BA_var = BA.var(dim=(-2, -1))
    BA_max = BA.flatten(start_dim=-2).max(dim=-1)[0]

    prob_observation = Omega.sum(dim=(-2, -1)) / Omega.shape[-1] / Omega.shape[-2]

    feature_vec = [datafit_sq_mean.log(), datafit_sq_var.log(), p_mean.log(), q_mean.log(),
                    BA_mean.log(), BA_var.log(), BA_max.log(), prob_observation,
                   *[t.squeeze(-1) for t in torch.split(nw_out, 1, dim=-1)]]

    feature_vec = torch.stack(feature_vec, dim=-1)

    return feature_vec


def _masked_mean_var(x, mask=None, dim=(-2, -1)):
    if mask is None:
        mean, var = x.mean(dim=dim), x.var(dim=dim)
    else:
        assert(mask.shape == x.shape)
        mask_sum = mask.sum(dim=dim)
        mask_sum = torch.clamp(mask_sum, min=2)
        mean = (x * mask).sum(dim=dim) / mask_sum
        var_temp = (x - mean.view(*mean.shape, *(len(dim)*[1]))) * mask
        var = (var_temp ** 2).sum(dim=dim) / (mask_sum - 1)

    return mean, var