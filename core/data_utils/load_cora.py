import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

import torch_geometric.transforms as T      # type: ignore
from torch_geometric.datasets import Planetoid      # type: ignore


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_cora_casestudy(seed: int = 0) -> Tuple[Any, np.ndarray]:
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # load data
    data_name = "cora"
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid("dataset", data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[: int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)]
    )
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8) :])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

    data.split_idx = {
        "train": torch.tensor(data.train_id),
        "valid": torch.tensor(data.val_id),
        "test": torch.tensor(data.test_id),
    }

    return data, data_citeid


# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = "dataset/cora_orig/cora"
    idx_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {
        x: i
        for i, x in enumerate(
            [
                "Case_Based",
                "Genetic_Algorithms",
                "Neural_Networks",
                "Probabilistic_Methods",
                "Reinforcement_Learning",
                "Rule_Learning",
                "Theory",
            ]
        )
    }
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape
    )
    data_edges = np.array(edges[~(edges == None).max(1)], dtype="int")
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text: bool = False, seed: int = 0) -> Tuple[Any, Optional[List[str]]]:
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open("dataset/cora_orig/mccallum/cora/papers") as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split("\t")[0]
        fn = line.split("\t")[1]
        pid_filename[pid] = fn

    path = "dataset/cora_orig/mccallum/cora/extractions/"
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path + fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if "Title:" in line:
                ti = line
            if "Abstract:" in line:
                ab = line
        text.append(ti + "\n" + ab)
    return data, text
