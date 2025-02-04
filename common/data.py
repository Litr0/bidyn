import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
from scipy.sparse import csr_matrix
import networkx as nx
import random
from tqdm import tqdm

def load_dataset(name, **kwargs):
    return load_dataset_csv(name, **kwargs)

def load_dataset_csv(dataset_name, group="train", variant=None, get_edges=True,
    get_deepwalk_feats=False):
    # load dataset
    graph = nx.Graph()
    edges, edge_feats = [], []
    edge_labels = {}
    node_types = {}
    bad_users = set()
    bad_edges = set()
    full_bad_edges = {}
    labels_items = []
    removed_users = set()
    users = set()
    feats_len = None
    edges_by_user = defaultdict(list)
    with open("data/{}.csv".format(dataset_name), "r") as f:
        # USE ONLY 10% OF THE DATA
        """ lines = f.readlines()
        ten_percent_lines = int(len(lines) * 0.1)
        random.shuffle(lines)
        f = lines[:ten_percent_lines] """
        idx, line_num = 0, 0
        prev_time = None
        for line in f:
            if line.startswith("user_id"): continue
            toks = line.strip().split(",")
            time = toks[2]
            if time == prev_time: continue
            prev_time = time
            line_num += 1
            label = int(toks[3])
            labels_items.append((label, toks[1]))
            if label == 1:
                bad_users.add(user)
                bad_edges.add(idx)
            user, item = toks[:2]
            user, item = "A" + user, "B" + item
            node_types[user] = 0
            node_types[item] = 1
            feats = [float(x) for x in toks[4:]]
            if label == 1 or user in bad_users:
                if user not in full_bad_edges:
                    full_bad_edges[user] = []
                full_bad_edges[user].append(feats)
            feats_len = len(feats)
            edge_labels[user, item] = label
            edge_labels[item, user] = label
            edges.append((user, item, float(time)))
            edge_feats.append(feats)
            edges_by_user[user].append((float(time)))
            users.add(user)
            idx += 1
    edge_feats = [f for f, (u, v, t) in zip(edge_feats, edges) if u not in removed_users]
    edges = [(u, v, t) for u, v, t in edges if u not in removed_users]
    users -= set(removed_users)
    node_types = {k: v for k, v in node_types.items() if k not in
        removed_users}
    graph.add_edges_from([x[:2] for x in edges])
    nodes, node_types = zip(*sorted(node_types.items(), key=lambda x: x[1]))
    nodes, node_types = list(nodes), list(node_types)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v], t) for u, v, t in edges]

    mat_flat = nx.to_scipy_sparse_array(graph, nodelist=nodes)
    labels = np.array([1 if x in users and x in bad_users else 0 for x in
        nodes])
    n_asins = len(users)
    n_buyers = len(nodes) - n_asins
    asin_filter = np.array([(x in users) for x in nodes])
    buyer_filter = np.array([(x not in users) for x in nodes])

    bad_edges_feats = [f for i, f in enumerate(edge_feats) if i in bad_edges]

    mean_bad_edge_feats = np.mean(bad_edges_feats, axis=0)

    print("Mean bad edge feats: {}".format(mean_bad_edge_feats))

    mean_bad_edge_feats_by_user = {}
    for user, feats in full_bad_edges.items():
        mean_bad_edge_feats_by_user[user] = np.mean(feats, axis=0)
    
    mean_bad_edge_feats = np.mean([f for f in mean_bad_edge_feats_by_user.values()], axis=0)

    print("Mean bad edge feats: {}".format(mean_bad_edge_feats))

    d = {
        "mats": [],
        "mat_flat": mat_flat,
        "labels": labels,
        "n_asins": n_asins,
        "n_buyers": n_buyers,
        "asin_filter": asin_filter,
        "buyer_filter": buyer_filter,
        "asin_title_embs": None,
        "buyer_info": None,
        "asin_to_idx": None,
        "buyer_to_idx": None,
        "edge_feats": edge_feats,
        "edges": edges,  # edges are in time order. edge_feats follows same order
        "labels_items": labels_items,
        "name": dataset_name
    }
    return d

def get_edge_lists(dataset):
    us, vs = set(), set()
    for u, v, t in tqdm(dataset["edges"]):
        if not (dataset["asin_filter"][u] and dataset["buyer_filter"][v]):
            continue
        us.add(u)
        vs.add(v)
    u_to_idx = {u: i for i, u in enumerate(sorted(us))}
    v_to_idx = {v: i for i, v in enumerate(sorted(vs))}

    us_to_edges = [[] for _ in range(len(us))]
    vs_to_edges = [[] for _ in range(len(vs))]
    for (u, v, t), feats in tqdm(zip(dataset["edges"], dataset["edge_feats"])):
        if not (dataset["asin_filter"][u] and dataset["buyer_filter"][v]):
            continue
        us_to_edges[u_to_idx[u]].append((t, v_to_idx[v], feats))
        vs_to_edges[v_to_idx[v]].append((t, u_to_idx[u], feats))
    
    labels = dataset["labels"]
    us_to_edges_labels = [[] for _ in range(len(us))]
    vs_to_edges_labels = [[] for _ in range(len(vs))]
    for (u, v, t), feats, label in zip(dataset["edges"], dataset["edge_feats"], labels):
        if not (dataset["asin_filter"][u] and dataset["buyer_filter"][v]):
            continue
        us_to_edges_labels[u_to_idx[u]].append((t, v_to_idx[v], feats, label))
        vs_to_edges_labels[v_to_idx[v]].append((t, u_to_idx[u], feats, label))
    # including more neighbors gives diminishing returns, so still subsample
    if dataset["name"] in ["reddit", "steam_2017_new_swapped"]:
        for u in tqdm(range(len(us_to_edges))):
            us_to_edges[u] = random.sample(us_to_edges[u], min(200,
                len(us_to_edges[u])))
        for v in tqdm(range(len(vs_to_edges))):
            vs_to_edges[v] = random.sample(vs_to_edges[v], min(200,
                len(vs_to_edges[v])))

    return u_to_idx, v_to_idx, us_to_edges, vs_to_edges, us_to_edges_labels, vs_to_edges_labels
