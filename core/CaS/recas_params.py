from typing import Any, Dict, List, Optional

from core.CaS.params import BaseParams


class ReCaSParams(BaseParams):
    def get(
        self,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str = "microsoft/deberta-base",
        feature_type: str = "Ensemble",
        emb: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return super()._get(
            dataset=dataset,
            gnn_name=gnn_name,
            lm_name=lm_name,
            feature_type=feature_type,
            emb=emb,
        )

    def add(
        self,
        params: List[Dict[str, Any]],
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str = "microsoft/deberta-base",
        feature_type: str = "Ensemble",
        emb: Optional[str] = None,
    ):
        self._add(
            params,
            dataset=dataset,
            gnn_name=gnn_name,
            lm_name=lm_name,
            feature_type=feature_type,
            emb=emb,
        )
