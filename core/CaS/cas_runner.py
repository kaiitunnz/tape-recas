import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from yacs.config import CfgNode as CN  # type: ignore

import core.CaS.cas_utils as cas_utils
from core.GNNs.gnn_utils import Evaluator, get_pred_fname
from core.data_utils.load import load_data, load_gpt_preds
from core.CaS.cas_utils import gen_normalized_adjs, process_adj

_CaSFnType = Callable[..., Tuple[torch.Tensor, torch.Tensor]]
_CAS_PARAMS_FPATH = Path(__file__).parent.resolve() / "cas_params.json"


class CaSRunner:
    def __init__(
        self,
        cfg: CN,
        feature_type: Optional[str],
        params_fpath: Union[str, Path] = _CAS_PARAMS_FPATH,
    ):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.lm_model_name = cfg.lm.model.name
        self.gnn_model_name = cfg.gnn.model.name  # Name of the predictor model
        self.feature_type = feature_type
        self.use_lm_pred = cfg.cas.use_lm_pred
        self.params_fpath = Path(params_fpath)

        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=cfg.seed
        )

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        self.data = data
        self.split_idx = data.split_idx

        self.evaluator = Evaluator(name=self.dataset_name)

    def run(self, params_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        adj, D_isqrt = process_adj(self.data)
        normalized_adjs = gen_normalized_adjs(adj, D_isqrt)

        cas_params, cas_fn, cas_fn_str = self._get_params(normalized_adjs, params_dict)

        if self.use_lm_pred:
            model_preds = self._load_lm_pred()
        else:
            model_preds = self._load_gnn_pred()

        result_df = pd.DataFrame()
        _, eval_results, eval_results_raw_none = self.evaluate_original_preds(model_preds)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(True), eval_results
        )
        _, eval_results, eval_results_raw_c = self.evaluate_c_preds(model_preds, cas_params, cas_fn)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(False, 'c'), eval_results, cas_fn_str
        )
        _, eval_results, eval_results_raw_s = self.evaluate_s_preds(model_preds, cas_params, cas_fn)
        result_df = self._add_eval_results(
            result_df, self._get_method_name(False, 's'), eval_results, cas_fn_str
        )
        raw_change_train_df = self.check_pred_change(eval_results_raw_none["train"], eval_results_raw_c["train"], eval_results_raw_s["train"])
        raw_change_valid_df = self.check_pred_change(eval_results_raw_none["valid"], eval_results_raw_c["valid"], eval_results_raw_s["valid"])
        raw_change_test_df = self.check_pred_change(eval_results_raw_none["test"], eval_results_raw_c["test"], eval_results_raw_s["test"])

        return result_df, raw_change_train_df, raw_change_valid_df, raw_change_test_df
    
    def check_pred_change(self, raw_none_np: np.ndarray, raw_c_np: np.ndarray, raw_r_np: np.ndarray) -> pd.DataFrame:
        raw_all_np = np.concatenate([raw_none_np, raw_c_np, raw_r_np], axis=1)
        raw_diff_np = np.diff(raw_all_np.astype(int), axis=1)
        raw_change = pd.DataFrame(raw_diff_np, columns=['None-C', 'C-S'])
        # for deep analysis of the change
        raw_change['FT'] = raw_change.apply(lambda x: True if x['None-C'] == -1 and x['C-S'] == 1 else False, axis=1)
        raw_change['TF'] = raw_change.apply(lambda x: True if x['None-C'] == 1 and x['C-S'] == -1 else False, axis=1)
        raw_change['NT'] = raw_change.apply(lambda x: True if x['None-C'] == 0 and x['C-S'] == 1 else False, axis=1)
        raw_change['TN'] = raw_change.apply(lambda x: True if x['None-C'] == 1 and x['C-S'] == 0 else False, axis=1)
        raw_change['NF'] = raw_change.apply(lambda x: True if x['None-C'] == 0 and x['C-S'] == -1 else False, axis=1)
        raw_change['FN'] = raw_change.apply(lambda x: True if x['None-C'] == -1 and x['C-S'] == 0 else False, axis=1)
        raw_change['NN'] = raw_change.apply(lambda x: True if x['None-C'] == 0 and x['C-S'] == 0 else False, axis=1)
        return raw_change

        

    def _get_method_name(self, is_original: bool, which_cas: str = '') -> str:
        feature_type = "Ensemble" if self.feature_type is None else self.feature_type
        method_name = (
            f"{self.lm_model_name}+{feature_type}"
            if self.use_lm_pred
            else f"{self.lm_model_name}+{self.gnn_model_name}+{feature_type}"
        )
        if is_original:       
            return method_name
        elif which_cas == 'c':
            return method_name + "+C"
        elif which_cas == 's':
            return method_name + "+S"
        else:
            raise ValueError("Invalid which_cas")

    def _add_eval_results(
        self, result_df: pd.DataFrame, method_name: str, eval_results: Dict[str, float], cas_fn_str: str
    ) -> pd.DataFrame:
        return pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "method": [method_name],
                        "cas_fn": [cas_fn_str],
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
        )

    def evaluate_original_preds(self, preds: torch.Tensor) -> Dict[str, float]:
        self._validate_preds(preds)
        split_acc = {split: self._eval(preds, split)["acc"] for split in ["train", "valid", "test"]}
        split_correct = {split: self._eval(preds, split)["correct"] for split in ["train", "valid", "test"]}
        return preds, split_acc, split_correct

    def evaluate_c_preds(
        self, preds: torch.Tensor, cas_params: Dict[str, Any], cas_fn: _CaSFnType
    ) -> Dict[str, float]:
        self._validate_preds(preds)
        res_result, _ = cas_fn(self.data, preds, self.split_idx, **cas_params)
        split_acc = {split: self._eval(res_result, split)["acc"] for split in ["train", "valid", "test"]}
        split_correct = {split: self._eval(res_result, split)["correct"] for split in ["train", "valid", "test"]}
        return res_result, split_acc, split_correct
    
    def evaluate_s_preds(
        self, preds: torch.Tensor, cas_params: Dict[str, Any], cas_fn: _CaSFnType
    ) -> Dict[str, float]:
        self._validate_preds(preds)
        _, result = cas_fn(self.data, preds, self.split_idx, **cas_params)
        split_acc = {split: self._eval(result, split)["acc"] for split in ["train", "valid", "test"]}
        split_correct = {split: self._eval(result, split)["correct"] for split in ["train", "valid", "test"]}
        return result, split_acc, split_correct
    
    # def evaluate_cas_preds(
    #     self, preds: torch.Tensor, cas_params: Dict[str, Any], cas_fn: _CaSFnType
    # ) -> Dict[str, float]:
    #     self._validate_preds(preds)
    #     _, result = cas_fn(self.data, preds, self.split_idx, **cas_params)
    #     return {
    #         split: self._eval(result, split) for split in ["train", "valid", "test"]
    #     }

    def _validate_preds(self, preds: torch.Tensor):
        if (preds.sum(dim=-1) - 1).abs().max() > 1e-1:
            raise ValueError("Input predictions do not sum to 1.")

    def _load_default_params(self) -> Dict[str, Any]:
        with open(self.params_fpath, "r") as f:
            all_params: Dict[str, Any] = json.load(f)
        feature_type = self.feature_type or "Ensemble"
        gnn_model_name = "None" if self.use_lm_pred else self.gnn_model_name
        return all_params[self.dataset_name][self.lm_model_name][gnn_model_name][
            feature_type
        ]

    def _get_params(
        self,
        normalized_adjs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        params_dict: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Any]:
        new_params_dict = (
            self._load_default_params() if params_dict is None else params_dict.copy()
        )
        saved_params_dict = new_params_dict.copy()  # TODO: remove this

        cas_fn = new_params_dict.pop("cas_fn")

        if cas_fn == "double_correlation_autoscale":
            new_params_dict.update(
                {
                    "train_only": True,
                    "A1": normalized_adjs[new_params_dict["A1"]],
                    "A2": normalized_adjs[new_params_dict["A2"]],
                }
            )
        elif cas_fn == "double_correlation_fixed":
            new_params_dict.update(
                {
                    "train_only": True,
                    "A1": normalized_adjs[new_params_dict["A1"]],
                    "A2": normalized_adjs[new_params_dict["A2"]],
                }
            )
        elif cas_fn == "only_outcome_correlation":
            new_params_dict.update(
                {
                    "labels": ["train"],
                    "A": normalized_adjs[new_params_dict["A"]],
                }
            )
        else:
            raise ValueError(f"Unknown CaS function: {cas_fn}")

        # Debug
        debug_params = new_params_dict.copy()
        if "A1" in debug_params:
            debug_params["A1"] = saved_params_dict["A1"]
        if "A2" in debug_params:
            debug_params["A2"] = saved_params_dict["A2"]
        if "A" in debug_params:
            debug_params["A"] = saved_params_dict["A"]
        print("params_dict:", debug_params)

        return new_params_dict, getattr(cas_utils, cas_fn), cas_fn

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
