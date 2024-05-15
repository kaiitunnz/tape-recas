import sys
import time
from typing import Any, Dict, Optional

import optuna  # type: ignore
import pandas as pd
from optuna.trial import Trial

from core.config import cfg, update_cfg
from core.CaS.cas_runner import CaSRunner


def suggest_params(trial: Trial) -> Dict[str, Any]:
    cas_fn = trial.suggest_categorical(
        "cas_fn",
        [
            "double_correlation_autoscale",
            "double_correlation_fixed",
            "only_outcome_correlation",
        ],
    )
    params_dict: Dict[str, Any]
    if cas_fn == "double_correlation_autoscale":
        params_dict = {
            "alpha1": trial.suggest_float("alpha1", 0.1, 1.0),
            "alpha2": trial.suggest_float("alpha2", 0.1, 1.0),
            "A1": trial.suggest_int("A1", 0, 2),
            "A2": trial.suggest_int("A2", 0, 2),
            "num_propagations1": trial.suggest_int("num_propagations1", 10, 100, step=10),
            "num_propagations2": trial.suggest_int("num_propagations2", 10, 100, step=10),
        }
    elif cas_fn == "double_correlation_fixed":
        params_dict = {
            "alpha1": trial.suggest_float("alpha1", 0.1, 1.0),
            "alpha2": trial.suggest_float("alpha2", 0.1, 1.0),
            "scale": trial.suggest_float("scale", 5.0, 20.0),
            "A1": trial.suggest_int("A1", 0, 2),
            "A2": trial.suggest_int("A2", 0, 2),
            "num_propagations1": trial.suggest_int("num_propagations1", 10, 100, step=10),
            "num_propagations2": trial.suggest_int("num_propagations2", 10, 100, step=10),
        }
    elif cas_fn == "only_outcome_correlation":
        params_dict = {
            "alpha": trial.suggest_float("alpha", 0.1, 1.0),
            "A": trial.suggest_int("A", 0, 2),
            "num_propagations": trial.suggest_int("num_propagations", 10, 100, step=10),
        }
    else:
        raise ValueError(f"Unknown CaS function: {cas_fn}")
    params_dict["cas_fn"] = cas_fn
    return params_dict


def objective(trial: Trial, feature_type: Optional[str], cfg) -> float:
    params_dict = suggest_params(trial)
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    for seed in seeds:
        cfg.seed = seed
        runner = CaSRunner(cfg, feature_type)
        tmp_result_df = runner.run(params_dict)
        all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)

    result_df = all_result_df.groupby("method")
    avg_df = result_df.mean()
    return avg_df.loc[runner._get_method_name(False)]["valid_acc"]


def run(cfg, best_params_dict: Dict[str, Dict[str, Any]]):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = CaSRunner(cfg, feature_type)
            best_params = best_params_dict[feature_type]
            print(f"Best parameters for '{runner._get_method_name(False)}': {best_params}")
            tmp_result_df = runner.run(best_params)
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
    best_params_dict = {}
    for feature_type in cfg.cas.feature_types:
        print(f"Feature type: {feature_type}", file=sys.stderr)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, feature_type, cfg), n_trials=100, n_jobs=cfg.cas.optuna.n_jobs
        )
        best_params_dict[feature_type] = study.best_params

    run(cfg, best_params_dict)
