from typing import Any, Dict, Optional

from core.CaS.params import BaseParams


class CaSParams(BaseParams):
    def get(
        self,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str = "microsoft/deberta-base",
        feature_type: str = "Ensemble",
        emb: Optional[str] = None,
    ) -> Dict[str, Any]:
        return super()._get(
            dataset=dataset,
            gnn_name=gnn_name,
            lm_name=lm_name,
            feature_type=feature_type,
            emb=emb,
        )

    def add(
        self,
        *,
        dataset: str,
        gnn_name: str,
        lm_name: str = "microsoft/deberta-base",
        feature_type: str = "Ensemble",
        emb: Optional[str] = None,
        params: Dict[str, Any],
    ):
        self._add(
            params,
            dataset=dataset,
            gnn_name=gnn_name,
            lm_name=lm_name,
            feature_type=feature_type,
            emb=emb,
        )
