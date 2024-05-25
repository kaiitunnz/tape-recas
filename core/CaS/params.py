import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseParams:
    def __init__(self, path: Optional[Path] = None):
        self._path = path
        if path is None:
            self.params = {}
        else:
            with open(path, "r") as f:
                self.params = json.load(f)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def _get(
        self,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str,
        feature_type: str,
        emb: Optional[str],
    ) -> Any:
        params = self.params[dataset][lm_name][gnn_name]
        if gnn_name == "MLP":
            params = params[str(emb)]
        params = params[feature_type]
        return params

    def _add(
        self,
        params: Any,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str,
        feature_type: str,
        emb: Optional[str],
    ):
        emb = str(emb)
        params_dict = self.params
        if dataset not in params_dict:
            params_dict[dataset] = {}
        params_dict = params_dict[dataset]
        if lm_name not in params_dict:
            params_dict[lm_name] = {}
        params_dict = params_dict[lm_name]
        if gnn_name not in params_dict:
            params_dict[gnn_name] = {}
        params_dict = params_dict[gnn_name]
        if gnn_name == "MLP":
            if emb not in params_dict:
                params_dict[emb] = {}
            params_dict = params_dict[emb]
        params_dict[feature_type] = params

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    @abstractmethod
    def get(
        self,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str,
        feature_type: str,
        emb: Optional[str],
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def add(
        self,
        params: Any,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str,
        feature_type: str,
        emb: Optional[str],
    ):
        raise NotImplementedError()
