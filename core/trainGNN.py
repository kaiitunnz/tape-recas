import time

import pandas as pd
import torch

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.gnn_utils import get_pred_fname
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.config import cfg, update_cfg


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == "RevGAT":
        TRAINER = DGLGNNTrainer
    else:
        TRAINER = GNNTrainer

    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.train()
        logits, acc = trainer.eval_and_save()
        all_acc.append(acc)

        torch.save(
            logits.cpu(),
            get_pred_fname(
                trainer.dataset_name,
                trainer.lm_model_name,
                trainer.gnn_model_name,
                trainer.feature_type,
                trainer.use_emb,
                seed,
            ),
        )
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(
            f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
        )
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    run(cfg)
