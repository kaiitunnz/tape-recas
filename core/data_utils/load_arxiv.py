from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
import torch_geometric.transforms as T  # type: ignore
from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore


def get_raw_text_arxiv(use_text: bool = False, seed: int = 0) -> Tuple[Any, Optional[List[str]]]:

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits["train"]] = True
    val_mask[idx_splits["valid"]] = True
    test_mask[idx_splits["test"]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.split_idx = idx_splits

    data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        "dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz", compression="gzip"
    )

    raw_text = pd.read_csv(
        "dataset/ogbn_arxiv_orig/titleabs.tsv",
        sep="\t",
        header=None,
        names=["paper id", "title", "abs"],
    )
    df = pd.merge(nodeidx2paperid, raw_text, on="paper id")
    text = []
    for ti, ab in zip(df["title"], df["abs"]):
        t = "Title: " + ti + "\n" + "Abstract: " + ab
        text.append(t)
    return data, text
