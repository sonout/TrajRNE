import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import json
from itertools import chain, combinations
import torch
from _walker import random_walks as _random_walks

from road_network import RoadNetwork
from trajectory import Trajectory
from models import SpatialFlowConvolution

def generate_data(
    network,
    n_shortest_paths: int = 1280,
    window_size: int = 900,  
    number_negative: int = 3,
    save_batch_size=128,
    walks_type = 'trajectory', # shortest, random, trajectory
    wheighted_adj = None, # needed for trajectory walks
    traj_feature_df = None,
    traj_feats = "util", # which traj feat to use: util or avg_speed
    file_path = ".",
    file_name = "sre_traindata.json"
):
    """
    Generates the dataset like described in the corresponding paper. Since this needs alot of ram we use a batching approach.
    Args:
        n_shortest_paths (int, optional): how many shortest paths per node in graph. Defaults to 1280.
        window_size (int, optional): window size for distance neighborhood in meters. Defaults to 900.
        number_negative (int, optional): Negative samples to draw per positive sample.
        save_batch_size (int, optional): how many shortest paths to process between each save. Defaults to 128.
        file_path (str, optional): path where the dataset should be saved. Defaults to ".".
    """

    
    
    # Generate Walks
    if walks_type == 'random':
        print("Use random walks")
        paths = generate_random_paths(
            network.line_graph, n_shortest_paths=n_shortest_paths
        )
    elif walks_type == 'trajectory':
        print("Use trajectory wheighted walks")
        paths = generate_traj_paths(
            network.line_graph, wheighted_adj, n_shortest_paths=n_shortest_paths
        )
    else:
        print("Use shortest path")
        paths = generate_shortest_paths(
            network.line_graph, n_shortest_paths=n_shortest_paths
        )
    print(len(paths))
    
    
    ### Feature extraction: length, degree, traj_feats ###
    info = pd.DataFrame(network.gdf_edges)
    node_list = np.array(network.line_graph.nodes, dtype="l,l,l")
    
    # Length
    node_idx_to_length_map = info.loc[node_list.astype(list), "length"].to_numpy()
    # Road Type
    node_idx_to_highway_map = info.loc[node_list.astype(list), "highway"].to_numpy()

    # Add degree column to edge feature df
    for index, row in info.iterrows():
        second_node = index[1]
        degree = network.G.out_degree(second_node)
        info.at[index,'degree'] = degree

    node_idx_to_degree_map = info.loc[node_list.astype(list), "degree"].to_numpy()
    
    # Add trajectory feature
    if not ( traj_feature_df is None and traj_feats is None):
        info = info.join(traj_feature_df)
        # As those are continuous values we use binning to create categorical classes
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        info['traffic_bin'] =  pd.qcut(info[traj_feats], q=bins, labels=range(len(bins)-1))
        node_idx_to_traffic_map = info.loc[node_list.astype(list), 'traffic_bin'].to_numpy()
        

        
    ### Iterate over batches and save ### 
    save_path = os.path.join(file_path, file_name)
    # and generate train pairs
    for i in tqdm(range(0, len(paths), save_batch_size * len(network.line_graph.nodes))):
        trainset = extract_pairs( 
            node_idx_to_length_map,
            node_idx_to_highway_map,
            node_idx_to_degree_map,
            node_idx_to_traffic_map,
            node_list,
            paths[i : i + (save_batch_size * len(network.line_graph.nodes))],
            window_size,
            number_negative,
        )
        # save batch
        if not os.path.isfile(save_path):
            with open(save_path, "w") as fp:
                json.dump(trainset, fp)
        else:
            with open(save_path, "r") as fp:
                a = np.array(json.load(fp))
                a = np.unique(np.vstack([a, np.array(trainset)]), axis=0)
            with open(save_path, "w") as fp:
                json.dump(a.tolist(), fp)


def extract_pairs(
    info_length: np.array,
    info_highway: np.array,
    info_degree: np.array,
    info_traffic: np.array,
    node_list: np.array,
    node_paths: list,
    window_size: int,
    number_negative: int,
):
    """_summary_
    
     Generates the traning pairs consisting of (v_x, v_y, in window_size?, same degree?, same traffic?). This is highly optimized.

    Args:
        info_length (np.array): length for each node in graph (ordered by node ordering in graph)
        info_highway (np.array): type for each node in graph (ordered by node ordering in graph)
        node_list (np.array): nodes in graph (ordered by node ordering in graph)
        node_paths (list): shortest paths
        window_size (int): window_size in meters
        number_negative (int): number negative to draw for each node

    Returns:
        list: training pairs
    """
    res = []
    # lengths of orginal sequences in flatted with cumsum to get real position in flatted
    orig_lengths = np.array([0] + [len(x) for x in node_paths]).cumsum()
    flatted = list(chain.from_iterable(node_paths))
    # get all lengths of sequence roads
    flat_lengths = info_length[flatted]

    # generate window tuples
    node_combs = []
    for i in range(len(orig_lengths) - 1):
        lengths = flat_lengths[orig_lengths[i] : orig_lengths[i + 1]]
        # cumsum = lengths.cumsum()
        for j in range(len(lengths)):
            mask = (lengths[j:].cumsum() < window_size).sum()
            # idx = (np.abs(lengths[j:].cumsum() - window_size)).argmin()
            window = node_paths[i][j : j + mask]
            if len(window) > 1:
                combs = tuple(combinations(window, 2))
                node_combs.extend(combs)

    # save distinct tuples
    node_combs = list(dict.fromkeys(node_combs))
    node_combs = list(chain.from_iterable(node_combs))



    # generate same degree labels
    degrees = info_degree[node_combs].reshape(int(len(node_combs) / 2), 2)
    
    # generate same rad type labels
    highways = info_highway[node_combs].reshape(int(len(node_combs) / 2), 2)
    
    # generate same traffic labels
    traffic_labels = info_traffic[node_combs].reshape(int(len(node_combs) / 2), 2)
    
    

    # Generate pairs: node1, node2, true (on same walk), true if same degree
    pairs = np.c_[
        np.array(node_combs).reshape(int(len(node_combs) / 2), 2),
        np.ones(degrees.shape[0]),
        degrees[:, 0] == degrees[:, 1],
        traffic_labels[:, 0] == traffic_labels[:, 1],
        highways[:, 0] == highways[:, 1],
    ].astype(
        int
    )  # same type



    res.extend(tuple(pairs.tolist()))

    # generate negative sample with same procedure as for positive
    neg_nodes = np.random.choice(
        np.setdiff1d(np.arange(0, len(node_list)), node_combs),
        size=pairs.shape[0] * number_negative,
    )

    neg_pairs = pairs.copy()
    neg_pairs = neg_pairs.repeat(repeats=number_negative, axis=0)
    replace_mask = np.random.randint(0, 2, size=neg_pairs.shape[0]).astype(bool)
    neg_pairs[replace_mask, 0] = neg_nodes[replace_mask]
    neg_pairs[~replace_mask, 1] = neg_nodes[~replace_mask]
    neg_pairs[:, 2] -= 1

    neg_degree = info_degree[neg_pairs[:, :2].flatten()].reshape(
        neg_pairs.shape[0], 2
    )
    
    neg_traffic = info_traffic[neg_pairs[:, :2].flatten()].reshape(
        neg_pairs.shape[0], 2
    )
    
    neg_highways = info_highway[neg_pairs[:, :2].flatten()].reshape(
            neg_pairs.shape[0], 2
        )

    

    neg_pairs[:, 3] = neg_degree[:, 0] == neg_degree[:, 1]
    neg_pairs[:, 4] = neg_traffic[:, 0] == neg_traffic[:, 1]
    neg_pairs[:, 5] = neg_highways[:, 0] == neg_highways[:, 1]

    res.extend(tuple(neg_pairs.tolist()))

    return res



############### Implementations of walks variations #################
def generate_traj_paths(G: nx.Graph, weighted_adj, n_shortest_paths: int, walk_len=25):
    from scipy import sparse
    from _walker import random_walks as _random_walks

    # Get input for random walker
    A = sparse.csr_matrix(weighted_adj)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)

    # Iterate over nodes and run random walker
    nodes = np.arange(len(G.nodes))
    paths = []
    for source in tqdm(nodes):          
        node_paths = _random_walks(indptr, indices, data, [source], n_shortest_paths, walk_len + 1).astype(int)
        paths.extend(tuple(node_paths))
    return paths


def generate_random_paths(G: nx.Graph, n_shortest_paths: int, walk_len=25):

    nodes = np.arange(len(G.nodes))
    paths = []
    for source in tqdm(nodes):
        node_paths = walker.random_walks(G, n_walks=n_shortest_paths, walk_len=walk_len, start_nodes=[source]).tolist()
        paths.extend(tuple(node_paths))
    return paths


def generate_shortest_paths(G: nx.Graph, n_shortest_paths: int):
    """
    Generates shortest paths between node in graph G.
    Args:
        G (nx.Graph): graph
        n_shortest_paths (int): how many shortest path to generate per node in G.

    Returns:
        list: shortest_paths
    """
    adj = nx.adjacency_matrix(G)

    def get_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        if Pr[i, k] == -9999 and k != i:
            return []

        return path[::-1]

    _, P = shortest_path(
        adj, directed=True, method="D", return_predecessors=True, unweighted=True
    )

    nodes = np.arange(len(G.nodes))
    paths = []
    for source in tqdm(nodes):
        targets = np.random.choice(
            np.setdiff1d(np.arange(len(G.nodes)), [source]),
            size=n_shortest_paths,
            replace=False,
        )
        node_paths = []
        for target in targets:
            path = get_path(P, source, target)
            if path != []:
                node_paths.append(path)
        paths.extend(tuple(node_paths))
    return paths





def main(city):
    # Load Data

    # load trajectory data
    traj_features = pd.read_csv(f"data/trajectory/speed_features_unnormalized.csv")
    traj_features.set_index(["u", "v", "key"], inplace=True)
    traj_features["util"] = (traj_features["util"] - traj_features["util"].min()) / (traj_features["util"].max() - traj_features["util"].min())  # min max normalization
    traj_features["avg_speed"] = (traj_features["avg_speed"] - traj_features["avg_speed"].min()) / (traj_features["avg_speed"].max() - traj_features["avg_speed"].min())  # min max normalization
    traj_features.fillna(0, inplace=True)

    # Network
    # Load network
    network = RoadNetwork()
    network.load(f"data/network")
    trajectory = Trajectory(f"data/trajectory/road_segment_map_sample.csv", nrows=100000000).generate_TTE_datatset()
    data = network.generate_road_segment_pyg_dataset(include_coords=True, dataset=city)

    # Create the transition probability matrix 
    # For SFC with k=2
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    # precalc adj matrices
    SpatialFlowConvolution(data, device, network, trajectory, k=2, bidirectional=False, add_self_loops=True)
    # For trajectory weighted random wlaks of SRE 
    SpatialFlowConvolution(data, device, network, trajectory, k=1, bidirectional=False, add_self_loops=False)

    # Weighted Adjecency for trajectory walks
    adj = np.loadtxt(f"data/traj_adj_k_1_False.gz")


    # Generate Training Data
    generate_data(
        network,
        n_shortest_paths= 1000,
        window_size= 900, 
        number_negative= 3,
        save_batch_size=64,
        walks_type = 'trajectory', # shortest, random, trajectory
        wheighted_adj = adj, # needed for trajectory walks
        traj_feature_df = traj_features,
        traj_feats = "util", # which traj feat to use: util or avg_speed
        file_path = "data/",
        file_name = "sre_traindata.json",
    )

if __name__ == '__main__':
    city = "porto"
    main(city)
