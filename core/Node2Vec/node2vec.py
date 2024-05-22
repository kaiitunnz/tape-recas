from time import time
from typing import Dict, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec  # type: ignore
from torch_sparse import SparseTensor  # type: ignore
from yacs.config import CfgNode as CN  # type: ignore

from core.data_utils.load import load_data
from core.utils import time_logger

LOG_FREQ = 10


def get_ckpt_dir(dataset_name: str) -> str:
    return f"output/{dataset_name}"

def get_emb_fname(
    dataset_name: str, seed: int
) -> str:
    return (
        f"{get_ckpt_dir(dataset_name)}/"
        "Node2Vec_"
        f"seed{seed}.emb"
    )


class Node2VecTrainer:
    def __init__(self, cfg: CN):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.epochs = cfg.node2vec.epochs
        self.batch_size = cfg.node2vec.batch_size
        self.num_workers = cfg.node2vec.num_workers
        self.max_iter = cfg.node2vec.max_iter

        self.data, self.num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed
        )
        if isinstance(self.data.edge_index, SparseTensor):
            r, c, _ = self.data.edge_index.coo()
            edge_index = torch.stack([r, c], dim=0)
        else:
            edge_index = self.data.edge_index
        self.model = Node2Vec(
            edge_index,
            embedding_dim=128,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
        ).to(self.device)
        self.ckpt_dir = get_ckpt_dir(self.dataset_name)
        self.ckpt = f"{self.ckpt_dir}/Node2Vec.pt"

    def _train(self, loader: DataLoader, optimizer: Optimizer) -> Tuple[float, float]:
        self.model.train()
        data = self.data
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        emb = self.model()
        train_acc = self.model.test(
            emb[data.train_mask],
            data.y[data.train_mask],
            emb[data.test_mask],
            data.y[data.test_mask],
            max_iter=self.max_iter,
        )
        return total_loss / len(loader), train_acc

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float, torch.Tensor]:
        self.model.eval()
        data = self.data
        emb = self.model()
        val_acc = self.model.test(
            emb[data.train_mask],
            data.y[data.train_mask],
            emb[data.val_mask],
            data.y[data.val_mask],
            max_iter=self.max_iter,
        )
        test_acc = self.model.test(
            emb[data.train_mask],
            data.y[data.train_mask],
            emb[data.test_mask],
            data.y[data.test_mask],
            max_iter=self.max_iter,
        )
        return val_acc, test_acc, emb

    @time_logger
    def train(self) -> Node2Vec:
        loader = self.model.loader(
            batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        for epoch in range(self.epochs):
            t0, es_str = time(), ""
            loss, train_acc = self._train(loader, optimizer)
            val_acc, _, _ = self._evaluate()
            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}"
                )
        return self.model

    @torch.no_grad()
    def eval_and_save(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, emb = self._evaluate()
        print(
            f"[Node2Vec] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n"
        )
        res = {"val_acc": val_acc, "test_acc": test_acc}
        return emb, res
