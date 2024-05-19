import time
import pandas as pd
import os

from core.config import cfg, update_cfg
from core.CaS.cas_runner import CaSRunner


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    start = time.time()
    for seed in seeds:

        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = CaSRunner(cfg, feature_type)
            tmp_result_df, tmp_raw_change_train_df, tmp_raw_change_valid_df, tmp_raw_change_test_df = runner.run()
            all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)

            # save df for further analysis
            if cfg.cas.use_lm_pred:
                os.makedirs(f'sep_cas_rets/{cfg.dataset}/{seed}/LM', exist_ok=True)
                tmp_raw_change_train_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/LM/{feature_type}_raw_change_train.csv', index=False)
                tmp_raw_change_valid_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/LM/{feature_type}_raw_change_valid.csv', index=False)
                tmp_raw_change_test_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/LM/{feature_type}_raw_change_test.csv', index=False)
            else:
                os.makedirs(f'sep_cas_rets/{cfg.dataset}/{seed}/{cfg.gnn.model.name}', exist_ok=True)
                tmp_raw_change_train_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/{cfg.gnn.model.name}/{feature_type}_raw_change_train.csv', index=False)
                tmp_raw_change_valid_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/{cfg.gnn.model.name}/{feature_type}_raw_change_valid.csv', index=False)
                tmp_raw_change_test_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{seed}/{cfg.gnn.model.name}/{feature_type}_raw_change_test.csv', index=False)
    end = time.time()

    # save df for further analysis
    if cfg.cas.use_lm_pred:
        all_result_df.to_csv(f'sep_cas_rets/{cfg.dataset}/LM.csv', index=False)
    else:
        all_result_df.to_csv(f'sep_cas_rets/{cfg.dataset}/{cfg.gnn.model.name}.csv', index=False)

    result_df = all_result_df.drop('cas_fn', axis=1).groupby("method")
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
