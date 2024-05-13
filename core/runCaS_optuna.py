import time
import optuna
import pandas as pd

from core.config import cfg, update_cfg
from core.CaS.cas_runner import CaSRunner


def objective(trial, cfg):
    params_dict = {
        "alpha1": trial.suggest_float("alpha1", 0.1, 2.0),
        "alpha2": trial.suggest_float("alpha2", 0.1, 2.0),
        "scale": trial.suggest_float("scale", 5.0, 20.0),
        "num_propagations1": trial.suggest_int("num_propagations1", 10, 100),
        "num_propagations2": trial.suggest_int("num_propagations2", 10, 100),
        "A1": trial.suggest_int("A1", 0, 2),
        "A2": trial.suggest_int("A2", 0, 2),
    }
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    for seed in seeds:
        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = CaSRunner(cfg, feature_type)

            tmp_result_df = runner.run(params_dict)
            all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)

    result_df = all_result_df.groupby("method")
    avg_df = result_df.mean()
    return avg_df["valid_acc"].mean()


def run(cfg, best_params_dict):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = CaSRunner(cfg, feature_type)

            tmp_result_df = runner.run(best_params_dict)
            all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)
    end = time.time()

    result_df = all_result_df.groupby("method")
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
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=100)
    best_params_dict = study.best_params

    run(cfg, best_params_dict)
