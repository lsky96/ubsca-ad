from copy import deepcopy
import numpy as np
import torch
import networkx as nx
from itertools import permutations


class ScenarioSet():
    stype = "ROmPQAN"
    data = None
    temp_data = None

    def __init__(self, **kwargs):
        self.data = {"batch_size": kwargs["batch_size"], "graph_param": None, "sampling_param": None}
        self.temp_data = {}

        if "graph_param" in kwargs.keys():
            self.data["graph_param"] = kwargs["graph_param"]

        if "sampling_param" in kwargs.keys():
            self.data["sampling_param"] = kwargs["sampling_param"]

        for t in self._valid_stypes():
            if all(elem in kwargs.keys() for elem in self._stype2storedvals(t)):
                self.stype = t
                for k in self._stype2storedvals(t):
                    self.data[k] = kwargs[k]
                break

    def __getitem__(self, item):
        if item in self._stype2storedvals(self.stype) or item in ["batch_size", "graph_param", "sampling_param"]:
            return self.data[item]

        d = self.data
        if self.stype == "ROmPQAN":
            if item == "Z":
                return d["P"] @ d["Q"].mT
            elif item == "Y":
                Y = d["Omega"] * (d["R"] @ (d["P"] @ d["Q"].mT + d["A"]) + d["N"])
                return Y
            else:
                raise ValueError
        elif self.stype == "ROmZAN":
            if item == "Y":
                Y = d["Omega"] * (d["R"] @ (d["Z"] + d["A"]) + d["N"])
                return Y
            else:
                raise ValueError

    def __setitem__(self, key, value):
        if key in self._stype2storedvals(self.stype):
            self.data[key] = value
        else:
            raise ValueError("Key cannot be set for stype.")

    def return_subset(self, idxs):
        subset_data = {}
        subset_data["batch_size"] = len(idxs)
        subset_data["graph_param"] = deepcopy(self.data["graph_param"])
        subset_data["sampling_param"] = deepcopy(self.data["sampling_param"])

        for k in self._stype2storedvals(self.stype):
            if k == "exception":
                pass  # no exception yet
            else:
                subset_data[k] = self.data[k][idxs]

        subset = ScenarioSet(**subset_data)
        return subset

    def _stype2storedvals(self, type):
        if type == "ROmPQAN":
            return ["R", "Omega", "P", "Q", "A", "N"]
        elif type == "ROmZAN":
            return ["R", "Omega", "Z", "A", "N"]
        else:
            raise ValueError

    def _valid_stypes(self):
        return ["ROmPQAN", "ROmZAN"]


def generate_synthetic_nw_scenarios(batch_size=1, num_timesteps=10, graph_param=None, sampling_param=None, nflow_version="exp+uni"):
    if sampling_param is None:
        sampling_param = {}
    if graph_param is None:
        graph_param = {}

    num_nodes = graph_param["num_nodes"]
    num_edges = graph_param["num_edges"]
    min_distance = graph_param["min_distance"]

    flow_distr = sampling_param["flow_distr"]  # {"rank", "scale"} rank can be number or interval
    anomaly_distr = {"prob": None, "amplitude": torch.tensor(1.0, dtype=torch.float)}
    anomaly_distr = anomaly_distr | sampling_param["anomaly_distr"]  # {"prob", "amplitude"}  # the amplitudes should be positive (number or interval)
    noise_distr = sampling_param["noise_distr"]  # {"variance"}
    observation_prob = sampling_param["observation_prob"]  # float or list of two floats

    trials = 0
    i_scenario = 0
    scenario_graphs = []

    while i_scenario < batch_size:

        """Graph Generation"""
        if trials > 5 * batch_size:
            raise RuntimeError("Passed graph parameters yield connected graph with too low probability.")

        pos = {i: torch.rand(2).numpy() for i in range(num_nodes)}
        G = nx.random_geometric_graph(num_nodes, min_distance, pos=pos)

        # ensure that the graph has the desired number of edges
        while G.number_of_edges() > num_edges:
            # remove excess edges randomly
            excess_edges = G.number_of_edges() - num_edges
            edge_indices = np.arange(len(G.edges()))  # create an array of edge indices
            edges_to_remove = np.random.choice(edge_indices, size=excess_edges, replace=False)
            G.remove_edges_from([list(G.edges())[i] for i in edges_to_remove])

        if nx.is_connected(G):
            scenario_graphs.append(G)
            i_scenario += 1

        trials += 1

    """Routing Matrix Generation"""
    num_flows = num_nodes * (num_nodes - 1)
    num_directed_edges = 2 * num_edges
    R = torch.zeros(batch_size, num_directed_edges, num_flows)
    od_pairs = list(permutations(list(range(num_nodes)), 2))
    # For each OD pair, calculate the minimum hop count route and add it to the routing matrix
    for i_scenario in range(batch_size):
        edges = list(scenario_graphs[i_scenario].edges())
        edges_digraph = [*edges, *[e[::-1] for e in edges]]
        for i_flow, od in enumerate(od_pairs):
            route = nx.shortest_path(scenario_graphs[i_scenario], od[0], od[1])
            for i_dir_edge, edge in enumerate(edges_digraph):
                # Check if the edge is present in the path
                # if set(edge).issubset(path):
                if _directed_edge_in_path(route, edge):
                    # If so, set the corresponding entry in the matrix to 1
                    R[i_scenario][i_dir_edge][i_flow] = 1

    """Flow Generation"""
    if isinstance(flow_distr["rank"], list):
        flow_rank_min = flow_distr["rank"][0]
        flow_rank_max = flow_distr["rank"][1]
        flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max + 1, size=(batch_size,))
    else:
        flow_rank_max = flow_distr["rank"]
        flow_ranks = torch.tensor(flow_distr["rank"])

    if nflow_version == "mardani":
        gauss_distr_var = 1 / num_flows
        U = torch.randn(batch_size, num_flows, flow_rank_max) * torch.sqrt(torch.tensor(gauss_distr_var))
        W = torch.randn(batch_size, num_timesteps, flow_rank_max)
    elif nflow_version == "exp+uni":
        uniform_distr_var = 1 / num_flows
        U = torch.rand(batch_size, num_flows, flow_rank_max) * torch.sqrt(torch.tensor(12 * uniform_distr_var))

        qsampler = torch.distributions.exponential.Exponential(torch.tensor(1.0))  # exponential with variance 1
        W = qsampler.sample(sample_shape=(batch_size, num_timesteps, flow_rank_max))

    elif nflow_version == "abs_gaussian":
        gauss_distr_var = 1 / flow_ranks.unsqueeze(-1).unsqueeze(-1)
        U = torch.randn(batch_size, num_flows, flow_rank_max).abs() * torch.sqrt(torch.tensor(gauss_distr_var))
        W = torch.randn(batch_size, num_timesteps, flow_rank_max).abs()

    elif nflow_version == "exp+exp":
        distr_scale = 1 / flow_ranks.unsqueeze(-1).unsqueeze(-1)
        qsampler = torch.distributions.exponential.Exponential(torch.tensor(1.0))
        U = qsampler.sample(sample_shape=(batch_size, num_flows, flow_rank_max)) * torch.tensor(distr_scale)
        W = qsampler.sample(sample_shape=(batch_size, num_timesteps, flow_rank_max))

    else:
        raise ValueError

    # Sampling for random rank
    if isinstance(flow_distr["rank"], list):
        # flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max+1, size=(batch_size,))
        rank_mask = torch.arange(flow_rank_max).expand(batch_size, -1) < flow_ranks.unsqueeze(-1)
        U = U * rank_mask.unsqueeze(-2)
        W = W * rank_mask.unsqueeze(-2)

    if "scale" in flow_distr:
        flow_scale = flow_distr["scale"]
        if isinstance(flow_scale, list):
            flow_scale = torch.rand(batch_size, 1, 1) * (flow_scale[1] - flow_scale[0]) + flow_scale[0]
            U = U * flow_scale
        else:
            U = U * flow_scale

    """Anomaly Generation"""
    if isinstance(anomaly_distr["prob"], list):
        assert(anomaly_distr["prob"][1] > anomaly_distr["prob"][0])
        A_seed = torch.rand(batch_size, 1, 1) * (anomaly_distr["prob"][1] - anomaly_distr["prob"][0]) + anomaly_distr["prob"][0]
    else:
        A_seed = anomaly_distr["prob"]

    A = torch.zeros(batch_size, num_flows, num_timesteps, dtype=torch.float)
    # {1,-1} anomalies
    temp_rand = torch.rand_like(A)
    anomaly_indicator_pos = temp_rand <= A_seed / 2  # anomalies with value +1
    anomaly_indicator_neg = temp_rand >= (1 - A_seed / 2)  # anomalies with value -1
    if isinstance(anomaly_distr["amplitude"], list):
        ano_amplitude = torch.rand(batch_size, 1, 1) \
                         * (anomaly_distr["amplitude"][1] - anomaly_distr["amplitude"][0]) + anomaly_distr["amplitude"][
                             0]
        A[anomaly_indicator_pos] = 1
        A[anomaly_indicator_neg] = -1
        A = A * ano_amplitude
    else:
        A[anomaly_indicator_pos] = anomaly_distr["amplitude"]
        A[anomaly_indicator_neg] = -anomaly_distr["amplitude"]

    """Noise Generation"""
    if isinstance(noise_distr["variance"], list):
        noise_var = torch.rand(batch_size, 1, 1) * (noise_distr["variance"][1] - noise_distr["variance"][0]) + noise_distr["variance"][0]
    else:
        noise_var = torch.tensor(noise_distr["variance"])
    N = torch.sqrt(noise_var) * torch.randn(batch_size, num_directed_edges, num_timesteps)

    """Observations"""
    if isinstance(observation_prob, list):
        obs_prob_temp = torch.rand(batch_size, 1, 1) * (observation_prob[1] - observation_prob[0]) + observation_prob[0]
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= obs_prob_temp
    else:
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= observation_prob

    scenario_set = ScenarioSet(batch_size=batch_size,
                               graph_param=graph_param,
                               sampling_param=sampling_param,
                               R=R, Omega=Omega, P=U, Q=W, A=A, N=N)

    return scenario_set


def _directed_edge_in_path(path, edge):
    for i in range(len(path)-1):
        subpath = path[i:(i+2)]
        if edge[0] == subpath[0] and edge[1] == subpath[1]:
            return True
    return False


def nw_scenario_observation(scenario_set):
    Y = scenario_set["Y"]
    R = scenario_set["R"]
    Omega = scenario_set["Omega"]
    return Y, R, Omega


def nw_scenario_subset(scenario_set, indices):
    if isinstance(scenario_set, ScenarioSet):
        return scenario_set.return_subset(indices)
    else:
        new_scenario_set = {}
        new_scenario_set["batch_size"] = len(indices)
        if "graph_param" in scenario_set.keys():
            new_scenario_set["graph_param"] = scenario_set["graph_param"]
        if "sampling_param" in scenario_set.keys():
            new_scenario_set["sampling_param"] = scenario_set["sampling_param"]
        new_scenario_set["Y"] = scenario_set["Y"][indices]
        new_scenario_set["R"] = scenario_set["R"][indices]
        new_scenario_set["Omega"] = scenario_set["Omega"][indices]
        new_scenario_set["Z"] = scenario_set["Z"][indices]
        new_scenario_set["A"] = scenario_set["A"][indices]
        new_scenario_set["N"] = scenario_set["N"][indices]

    return new_scenario_set
