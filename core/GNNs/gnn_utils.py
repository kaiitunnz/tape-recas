from core.utils import init_path
import numpy as np

import torch


def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT', 'SAGE']:
        from core.GNNs.gnn_trainer import GNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return GNNTrainer


def get_ckpt_dir(dataset_name: str) -> str:
    return  f"output/{dataset_name}"


def get_pred_fname(
    dataset_name: str, lm_model_name: str, gnn_model_name: str, feature_type: str, seed: int
) -> str:
    return (
        f"{get_ckpt_dir(dataset_name)}/"
        f"{lm_model_name}_"
        f"{gnn_model_name}_"
        f"{feature_type}_"
        f"seed{seed}.pred"
    )


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []
        correct_arr = np.full((y_true.shape[0], y_true.shape[1]), False)

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            correct_arr[is_labeled, i] = correct
            acc_list.append(float(np.sum(correct))/len(correct))

        return {
            'acc': sum(acc_list)/len(acc_list),
            'correct': correct_arr,
        }


"""
Early stop modified from DGL implementation
"""


class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        if isinstance(path, list):
            self.path = [init_path(p) for p in path]
        else:
            self.path = init_path(path)

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        es_str = f'{self.counter:02d}/{self.patience:02d} | BestVal={self.best_score:.4f}@E{self.best_epoch}'
        return self.early_stop, es_str

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if isinstance(model, list):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), self.path[i])
        else:
            torch.save(model.state_dict(), self.path)
