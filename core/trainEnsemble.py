import time

import pandas as pd
import torch

from core.config import cfg, update_cfg
from core.GNNs.ensemble_trainer import EnsembleTrainer
from core.GNNs.gnn_utils import get_pred_fname


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        ensembler = EnsembleTrainer(cfg)
        pred, acc = ensembler.train()
        all_acc.append(acc)
        for feature_type, logits in pred.items():
            torch.save(
                logits.cpu(),
                get_pred_fname(
                    ensembler.dataset_name,
                    ensembler.lm_model_name,
                    ensembler.gnn_model_name,
                    feature_type,
                    seed,
                ),
            )

    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for f in df.keys():
            df_ = pd.DataFrame([r for r in df[f]])
            print(
                f"[{f}] ValACC: {df_['val_acc'].mean():.4f} ± {df_['val_acc'].std():.4f}, TestAcc: {df_['test_acc'].mean():.4f} ± {df_['test_acc'].std():.4f}"
            )
    print(f"Running time: {round((end-start)/len(seeds), 2)}s")


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    run(cfg)
