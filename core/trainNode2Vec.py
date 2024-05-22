import time

from core.config import cfg, update_cfg
from core.Node2Vec.node2vec import Node2VecTrainer, get_emb_fname

import pandas as pd
import torch


def main(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        trainer = Node2VecTrainer(cfg)
        trainer.train()
        emb, acc = trainer.eval_and_save()
        all_acc.append(acc)

        torch.save(emb.cpu(), get_emb_fname(trainer.dataset_name, seed))
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(
            f"[Node2Vec] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
        )
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
