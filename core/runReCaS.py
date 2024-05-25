import time
import pandas as pd

from core.config import cfg, update_cfg
from core.CaS.recas_runner import ReCaSRunner


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = ReCaSRunner(cfg, feature_type)
            tmp_result_df = runner.run()
            all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)
    end = time.time()

    result_df = all_result_df[["method", "train_acc", "valid_acc", "test_acc"]].groupby(
        "method"
    )
    avg_df = result_df.mean()
    std_df = result_df.std()
    for (method, avg_row), (_, std_row) in zip(avg_df.iterrows(), std_df.iterrows()):
        print(
            f"[{method}] "
            f'TrainACC: {avg_row["train_acc"]:.4f} ± {std_row["train_acc"]:.4f}, '
            f'ValACC: {avg_row["valid_acc"]:.4f} ± {std_row["valid_acc"]}, '
            f'TestACC: {avg_row["test_acc"]:.4f} ± {std_row["test_acc"]:.4f}'
        )
    print(f"Running time: {round((end-start)/len(seeds), 2)}s")


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    run(cfg)
