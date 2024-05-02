from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from yacs.config import CfgNode as CN  # type: ignore

from core.GNNs.gnn_utils import Evaluator, get_pred_fname
from core.data_utils.load import load_data, load_gpt_preds
from core.CaS.cas_utils import (
    double_correlation_autoscale,
    double_correlation_fixed,
    gen_normalized_adjs,
    only_outcome_correlation,
    process_adj,
)

_CaSFnType = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


class CaSRunner:
    def __init__(self, cfg: CN, feature_type: Optional[str]):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.lm_model_name = cfg.lm.model.name
        self.gnn_model_name = cfg.gnn.model.name  # Name of the predictor model
        self.feature_type = feature_type
        self.use_lm_pred = cfg.cas.use_lm_pred

        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=cfg.seed
        )

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        self.data = data
        self.split_idx = data.split_idx

        self.evaluator = Evaluator(name=self.dataset_name)

    def run(self) -> pd.DataFrame:
        adj, D_isqrt = process_adj(self.data)
        normalized_adjs = gen_normalized_adjs(adj, D_isqrt)

        cas_params, cas_fn = self._get_params(normalized_adjs)

        if self.use_lm_pred:
            model_preds = self._load_lm_pred()
        else:
            model_preds = self._load_gnn_pred()

        result_df = pd.DataFrame()
        eval_results = self.evaluate_original_preds(model_preds)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(True), eval_results
        )
        eval_results = self.evaluate_cas_preds(model_preds, cas_params, cas_fn)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(False), eval_results
        )

        return result_df

    def _get_method_name(self, is_original: bool) -> str:
        method_name = (
            f"{self.lm_model_name}+{self.feature_type}"
            if self.use_lm_pred
            else f"{self.lm_model_name}+{self.gnn_model_name}+{self.feature_type}"
        )
        return method_name if is_original else method_name + "+C&S"

    def _add_eval_results(
        self, result_df: pd.DataFrame, method_name: str, eval_results: Dict[str, float]
    ) -> pd.DataFrame:
        return pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "method": [method_name],
                        "train_acc": [eval_results["train"]],
                        "valid_acc": [eval_results["valid"]],
                        "test_acc": [eval_results["test"]],
                    },
                ),
            ]
        )

    def _eval(self, preds: torch.Tensor, split: str) -> float:
        idx = self.split_idx[split]
        return self.evaluator.eval(
            {
                "y_true": self.data.y[idx].view((-1, 1)),
                "y_pred": preds[idx].argmax(dim=-1, keepdim=True),
            }
        )["acc"]

    def evaluate_original_preds(self, preds: torch.Tensor) -> Dict[str, float]:
        self._validate_preds(preds)
        return {split: self._eval(preds, split) for split in ["train", "valid", "test"]}

    def evaluate_cas_preds(
        self, preds: torch.Tensor, cas_params: Dict[str, Any], cas_fn: _CaSFnType
    ) -> Dict[str, float]:
        self._validate_preds(preds)
        _, result = cas_fn(self.data, preds, self.split_idx, **cas_params)
        return {
            split: self._eval(result, split) for split in ["train", "valid", "test"]
        }

    def _validate_preds(self, preds: torch.Tensor):
        if (preds.sum(dim=-1) - 1).abs().max() > 1e-1:
            raise ValueError("Input predictions do not sum to 1.")

    def _get_params(
        self,
        normalized_adjs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[Dict[str, Any], _CaSFnType]:
        DAD, DA, AD = normalized_adjs
        # Now we use the default hyperparameters. Need fine-tuning. (C&S used Optuna.)
        # TODO
        params_dict = {
            "train_only": True,
            "alpha1": 1.0,
            "alpha2": 0.8,
            "scale": 10.0,
            "A1": DAD,
            "A2": DA,
            "num_propagations1": 50,
            "num_propagations2": 50,
        }
        cas_fn = double_correlation_fixed
        return params_dict, cas_fn

    def _topk_preds_to_logits(self, topk_preds: torch.Tensor) -> torch.Tensor:
        topk = topk_preds.size(-1)
        logits = torch.zeros(self.num_nodes, self.num_classes)
        weights = torch.tensor([[1 / (2**i) for i in range(topk)]]).expand(
            self.num_nodes, -1
        )
        logits.scatter_(-1, torch.clamp(topk_preds - 1, 0), weights)
        return logits

    def _load_lm_pred(self) -> torch.Tensor:
        def load_TA_E_logits(lm_pred_path: str) -> torch.Tensor:
            return torch.from_numpy(
                np.array(
                    np.memmap(
                        lm_pred_path,
                        mode="r",
                        dtype=np.float16,
                        shape=(self.num_nodes, self.num_classes),
                    )
                )
            ).to(torch.float32)

        def load_P_logits() -> torch.Tensor:
            topk = 3 if self.dataset_name == "pubmed" else 5
            topk_preds = load_gpt_preds(self.dataset_name, topk)
            return self._topk_preds_to_logits(topk_preds)

        if self.feature_type == "TA":
            print("Loading LM predictions (title and abstract) ...")
            LM_pred_path = (
                f"prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.pred"
            )
            print(f"LM_pred_path: {LM_pred_path}")
            logits = load_TA_E_logits(LM_pred_path)
        elif self.feature_type == "E":
            print("Loading LM predictions (explanations) ...")
            LM_pred_path = (
                f"prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.pred"
            )
            print(f"LM_pred_path: {LM_pred_path}")
            logits = load_TA_E_logits(LM_pred_path)
        elif self.feature_type == "P":
            print("Loading top-k predictions ...")
            logits = load_P_logits()
        elif self.feature_type == None:
            print("Loading an ensemble of LM predictions ...")
            LM_TA_pred_path = (
                f"prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.pred"
            )
            LM_E_pred_path = (
                f"prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.pred"
            )
            logits = torch.stack(
                [
                    load_TA_E_logits(LM_TA_pred_path),
                    load_TA_E_logits(LM_E_pred_path),
                    load_P_logits(),
                ],
                dim=0,
            ).mean(dim=0)
        else:
            raise ValueError("Invalid feature type")

        preds = F.softmax(logits, dim=-1)
        return preds

    def _load_gnn_pred(self) -> torch.Tensor:
        feature_type = self.feature_type or "ensemble"
        logits = torch.load(
            get_pred_fname(
                self.dataset_name,
                self.lm_model_name,
                self.gnn_model_name,
                feature_type,
                self.seed,
            ),
            map_location="cpu",
        )
        preds = F.softmax(logits, dim=-1)
        return preds
