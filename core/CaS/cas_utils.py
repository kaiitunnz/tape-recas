from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.functional import F
from torch_geometric.data.data import BaseData  # type: ignore
from torch_geometric.utils import to_undirected  # type: ignore
from torch_sparse import SparseTensor  # type: ignore
from tqdm import tqdm  # type: ignore

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_adj(data: BaseData) -> Tuple[SparseTensor, torch.Tensor]:
    N = data.num_nodes

    if isinstance(data.edge_index, SparseTensor):
        adj = data.edge_index
    else:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))

    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    return adj, deg_inv_sqrt


def gen_normalized_adjs(
    adj: SparseTensor, D_isqrt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def get_labels_from_name(
    labels: Union[List[str], str],
    split_idx: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(labels, list):
        label_tensor_list = []
        if len(labels) == 0:
            return torch.tensor([])
        for i in labels:
            label_tensor_list.append(split_idx[i])
        residual_idx = torch.cat(label_tensor_list)
    else:
        residual_idx = split_idx[labels]
    return residual_idx


def double_correlation_autoscale(
    data: BaseData,
    model_out: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    A1: torch.Tensor,
    alpha1: float,
    num_propagations1: int,
    A2: torch.Tensor,
    alpha2: float,
    num_propagations2: int,
    scale: float = 1.0,
    train_only: bool = False,
    device: str = _DEVICE,
    display: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_only:
        label_idx = torch.cat([split_idx["train"]])
        residual_idx = split_idx["train"]
    else:
        label_idx = torch.cat([split_idx["train"], split_idx["valid"]])
        residual_idx = label_idx

    y = pre_residual_correlation(
        labels=data.y.data, model_out=model_out, label_idx=residual_idx
    )
    resid = general_outcome_correlation(
        adj=A1,
        y=y,
        alpha=alpha1,
        num_propagations=num_propagations1,
        post_step=lambda x: torch.clamp(x, -1.0, 1.0),
        alpha_term=True,
        display=display,
        device=device,
    )

    orig_diff = y[residual_idx].abs().sum() / residual_idx.shape[0]
    resid_scale = orig_diff / resid.abs().sum(dim=1, keepdim=True)
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = resid_scale > 1000
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale * resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]
    y = pre_outcome_correlation(
        labels=data.y.data, model_out=res_result, label_idx=label_idx
    )
    result = general_outcome_correlation(
        adj=A2,
        y=y,
        alpha=alpha2,
        num_propagations=num_propagations2,
        post_step=lambda x: torch.clamp(x, 0, 1),
        alpha_term=True,
        display=display,
        device=device,
    )

    return res_result, result


def only_outcome_correlation(
    data: BaseData,
    model_out: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    A: torch.Tensor,
    alpha: float,
    num_propagations: int,
    labels: List[str],
    device: str = _DEVICE,
    display: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    res_result = model_out.clone()
    label_idxs = get_labels_from_name(labels, split_idx)
    y = pre_outcome_correlation(
        labels=data.y.data, model_out=model_out, label_idx=label_idxs
    )
    result = general_outcome_correlation(
        adj=A,
        y=y,
        alpha=alpha,
        num_propagations=num_propagations,
        post_step=lambda x: torch.clamp(x, 0, 1),
        alpha_term=True,
        display=display,
        device=device,
    )
    return res_result, result


def double_correlation_fixed(
    data: BaseData,
    model_out: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    A1: torch.Tensor,
    alpha1: float,
    num_propagations1: int,
    A2: torch.Tensor,
    alpha2: float,
    num_propagations2: int,
    scale: float = 1.0,
    train_only: bool = False,
    device: str = _DEVICE,
    display: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_only:
        label_idx = torch.cat([split_idx["train"]])
        residual_idx = split_idx["train"]

    else:
        label_idx = torch.cat([split_idx["train"], split_idx["valid"]])
        residual_idx = label_idx

    y = pre_residual_correlation(
        labels=data.y.data, model_out=model_out, label_idx=residual_idx
    )

    fix_y = y[residual_idx].to(device)

    def fix_inputs(x):
        x[residual_idx] = fix_y
        return x

    resid = general_outcome_correlation(
        adj=A1,
        y=y,
        alpha=alpha1,
        num_propagations=num_propagations1,
        post_step=lambda x: fix_inputs(x),
        alpha_term=True,
        display=display,
        device=device,
    )
    res_result = model_out + scale * resid

    y = pre_outcome_correlation(
        labels=data.y.data, model_out=res_result, label_idx=label_idx
    )

    result = general_outcome_correlation(
        adj=A2,
        y=y,
        alpha=alpha2,
        num_propagations=num_propagations2,
        post_step=lambda x: x.clamp(0, 1),
        alpha_term=True,
        display=display,
        device=device,
    )

    return res_result, result


def label_propagation(
    data: BaseData,
    split_idx: Dict[str, torch.Tensor],
    A: torch.Tensor,
    alpha: float,
    num_propagations: int,
    idxs: List,
) -> torch.Tensor:
    labels = data.y.data
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    label_idx = get_labels_from_name(idxs, split_idx)
    y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)

    return general_outcome_correlation(
        A,
        y,
        alpha,
        num_propagations,
        post_step=lambda x: torch.clamp(x, 0, 1),
        alpha_term=True,
    )


def pre_residual_correlation(
    labels: torch.Tensor, model_out: torch.Tensor, label_idx: torch.Tensor
) -> torch.Tensor:
    """Generates the initial labels used for residual correlation"""
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = int((labels.max() + 1).item())
    n = labels.shape[0]
    y = torch.zeros(n, c)
    y[label_idx] = (
        F.one_hot(labels[label_idx], c).float().squeeze(1) - model_out[label_idx]
    )
    return y


def general_outcome_correlation(
    adj: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    num_propagations: int,
    post_step: Callable[[torch.Tensor], torch.Tensor],
    alpha_term: bool,
    device: str = _DEVICE,
    display: bool = False,
):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    adj = adj.to(device)
    orig_device = y.device
    y = y.to(device)
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable=not display):
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1 - alpha) * y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)


def pre_outcome_correlation(
    labels: torch.Tensor, model_out: torch.Tensor, label_idx: torch.Tensor
) -> torch.Tensor:
    """Generates the initial labels used for outcome correlation"""
    labels = labels.cpu()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = int((labels.max() + 1).item())
    y = model_out.clone()
    if len(label_idx) > 0:
        y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)

    return y
