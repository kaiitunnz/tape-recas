import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from yacs.config import CfgNode as CN  # type: ignore

import core.CaS.cas_utils as cas_utils
from core.GNNs.gnn_utils import Evaluator, get_pred_fname
from core.data_utils.load import load_data, load_gpt_preds
from core.CaS.cas_utils import gen_normalized_adjs, process_adj
from core.CaS.recas_params import ReCaSParams

_CaSFnType = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


class ReCaSRunner:
    def __init__(
        self,
        cfg: CN,
        feature_type: Optional[str],
        params_path: Optional[Union[str, Path]] = None,
    ):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.lm_model_name = cfg.lm.model.name
        self.gnn_model_name = cfg.gnn.model.name  # Name of the predictor model
        self.feature_type = feature_type
        self.use_lm_pred = cfg.cas.use_lm_pred
        self.params_path = Path(
            cfg.cas.recas_params_path if params_path is None else params_path
        )
        self.use_emb = cfg.gnn.train.use_emb if self.gnn_model_name == "MLP" else None
        self.train_only = cfg.cas.train_only

        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=cfg.seed
        )

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        self.data = data
        self.split_idx = data.split_idx

        self.evaluator = Evaluator(name=self.dataset_name)

    def run(self, params_list: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        adj, D_isqrt = process_adj(self.data)
        normalized_adjs = gen_normalized_adjs(adj, D_isqrt)

        if self.use_lm_pred:
            model_preds = self._load_lm_pred()
        else:
            model_preds = self._load_gnn_pred()

        cas_params_list, cas_fn_list, cas_fn_str_list = self._get_params(
            normalized_adjs, params_list
        )

        preds = model_preds
        for cas_params, cas_fn in zip(cas_params_list, cas_fn_list):
            _, preds = cas_fn(self.data, preds, self.split_idx, **cas_params)

        result_df = pd.DataFrame()
        # Evaluate original preds
        eval_results = self.evaluate_preds(model_preds)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(True), eval_results, cas_fn_str_list
        )
        # Evaluate ReCaS preds
        eval_results = self.evaluate_cas_preds(preds)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(False), eval_results, cas_fn_str_list
        )

        return result_df

    def _get_method_name(self, is_original: bool) -> str:
        feature_type = "Ensemble" if self.feature_type is None else self.feature_type
        method_name = (
            f"{self.lm_model_name}+{feature_type}"
            if self.use_lm_pred
            else f"{self.lm_model_name}+{self.gnn_model_name}+{feature_type}"
        )
        if self.use_emb is not None:
            method_name += f"+{self.use_emb}"
        return method_name if is_original else method_name + "+ReCaS"

    def _add_eval_results(
        self,
        result_df: pd.DataFrame,
        method_name: str,
        eval_results: Dict[str, float],
        cas_fn_str_list: List[str],
    ) -> pd.DataFrame:
        return pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "method": [method_name],
                        "cas_fn_list": [cas_fn_str_list],
                        "train_acc": [eval_results["train"]],
                        "valid_acc": [eval_results["valid"]],
                        "test_acc": [eval_results["test"]],
                    },
                ),
            ]
        )

    def _eval(self, preds: torch.Tensor, split: str) -> Dict[str, Any]:
        idx = self.split_idx[split]
        return self.evaluator.eval(
            {
                "y_true": self.data.y[idx].view((-1, 1)),
                "y_pred": preds[idx].argmax(dim=-1, keepdim=True),
            }
        )

    def evaluate_preds(self, preds: torch.Tensor) -> Dict[str, float]:
        self._validate_preds(preds)
        return {
            split: self._eval(preds, split)["acc"]
            for split in ["train", "valid", "test"]
        }
    
    def evaluate_cas_preds(self, preds: torch.Tensor) -> Dict[str, float]:
        return {
            split: self._eval(preds, split)["acc"]
            for split in ["train", "valid", "test"]
        }

    def _validate_preds(self, preds: torch.Tensor):
        if (preds.sum(dim=-1) - 1).abs().max() > 1e-1:
            raise ValueError("Input predictions do not sum to 1.")

    def _load_default_params(self) -> List[Dict[str, Any]]:
        params = ReCaSParams(self.params_path)
        feature_type = self.feature_type or "Ensemble"
        gnn_model_name = "None" if self.use_lm_pred else self.gnn_model_name
        return params.get(
            dataset=self.dataset_name,
            lm_name=self.lm_model_name,
            gnn_name=gnn_model_name,
            feature_type=feature_type,
            emb=self.use_emb,
        )

    def _get_params(
        self,
        normalized_adjs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        params_list: Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[_CaSFnType], List[str]]:
        def copy(params_list: List[Dict[str, Any]]):
            return [params_dict.copy() for params_dict in params_list]

        new_params_list = (
            self._load_default_params() if params_list is None else copy(params_list)
        )
        cas_fn_list = []
        cas_fn_str_list = []

        for params_dict in new_params_list:
            cas_fn = params_dict.pop("cas_fn")
            params_dict["device"] = self.device
            if cas_fn == "double_correlation_autoscale":
                params_dict.update(
                    {
                        "train_only": self.train_only,
                        "A1": normalized_adjs[params_dict["A1"]],
                        "A2": normalized_adjs[params_dict["A2"]],
                    }
                )
            elif cas_fn == "double_correlation_fixed":
                params_dict.update(
                    {
                        "train_only": self.train_only,
                        "A1": normalized_adjs[params_dict["A1"]],
                        "A2": normalized_adjs[params_dict["A2"]],
                    }
                )
            elif cas_fn == "only_outcome_correlation":
                params_dict.update(
                    {
                        "labels": ["train"] if self.train_only else ["train", "valid"],
                        "A": normalized_adjs[params_dict["A"]],
                    }
                )
            elif (
                cas_fn == "only_double_correlation_autoscale"
                or cas_fn == "only_double_correlation_fixed"
            ):
                params_dict.update(
                    {
                        "train_only": self.train_only,
                        "A": normalized_adjs[params_dict["A"]],
                    }
                )
            else:
                raise ValueError(f"Unknown CaS function: {cas_fn}")

            cas_fn_list.append(getattr(cas_utils, cas_fn))
            cas_fn_str_list.append(cas_fn)

        return new_params_list, cas_fn_list, cas_fn_str_list

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
                self.use_emb,
                self.seed,
            ),
            map_location="cpu",
        )
        preds = F.softmax(logits, dim=-1)
        return preds
