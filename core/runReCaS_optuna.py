import sys
import time
from typing import Any, Dict, List, Optional

import optuna  # type: ignore
import pandas as pd
from optuna.trial import Trial

from core.config import cfg, update_cfg
from core.CaS.recas_runner import ReCaSRunner


def suggest_params_list(trial: Trial) -> List[Dict[str, Any]]:
    params_list = []

    prefix = "first_"
    first_cas_fn = trial.suggest_categorical(
        prefix + "cas_fn",
        [
            "double_correlation_autoscale",
            "only_outcome_correlation",
        ],
    )
    params_list.append(suggest_params(trial, first_cas_fn, prefix))

    prefix = "second_"
    second_cas_fn = trial.suggest_categorical(
        prefix + "cas_fn",
        # ["double_correlation_autoscale", "only_double_correlation_autoscale", None],
        ["double_correlation_autoscale", "only_double_correlation_autoscale"],
    )
    if second_cas_fn is not None:
        assert isinstance(second_cas_fn, str)  # for mypy
        params_list.append(suggest_params(trial, second_cas_fn, prefix))

    return params_list


def suggest_params(trial: Trial, cas_fn: str, prefix: str = "") -> Dict[str, Any]:
    params_dict: Dict[str, Any]
    if cas_fn == "double_correlation_autoscale":
        params_dict = {
            "alpha1": trial.suggest_float(prefix + "alpha1", 0.1, 1.0),
            "alpha2": trial.suggest_float(prefix + "alpha2", 0.1, 1.0),
            "A1": trial.suggest_int(prefix + "A1", 0, 2),
            "A2": trial.suggest_int(prefix + "A2", 0, 2),
            "num_propagations1": trial.suggest_int(
                prefix + "num_propagations1", 10, 100, step=10
            ),
            "num_propagations2": trial.suggest_int(
                prefix + "num_propagations2", 10, 100, step=10
            ),
        }
    elif cas_fn == "only_outcome_correlation":
        params_dict = {
            "alpha": trial.suggest_float(prefix + "alpha", 0.1, 1.0),
            "A": trial.suggest_int(prefix + "A", 0, 2),
            "num_propagations": trial.suggest_int(
                prefix + "num_propagations", 10, 100, step=10
            ),
        }
    elif cas_fn == "only_double_correlation_autoscale":
        params_dict = {
            "alpha": trial.suggest_float(prefix + "alpha", 0.1, 1.0),
            "A": trial.suggest_int(prefix + "A", 0, 2),
            "num_propagations": trial.suggest_int(
                prefix + "num_propagations", 10, 100, step=10
            ),
        }
    else:
        raise ValueError(f"Unknown CaS function: {cas_fn}")
    params_dict["cas_fn"] = cas_fn
    return params_dict


def objective(trial: Trial, feature_type: Optional[str], cfg) -> float:
    params_list = suggest_params_list(trial)
    old_seed = cfg.seed
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    for seed in seeds:
        cfg.seed = seed
        runner = ReCaSRunner(cfg, feature_type)
        tmp_result_df = runner.run(params_list)
        all_result_df = pd.concat([all_result_df, tmp_result_df], ignore_index=True)

    result_df = all_result_df[["method", "valid_acc"]].groupby("method")
    avg_df = result_df.mean()
    score = avg_df.loc[runner._get_method_name(False)].item()
    cfg.seed = old_seed  # Restore the seed.
    return score


def run(cfg, best_params_dict: Dict[str, List[Dict[str, Any]]]):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_result_df = pd.DataFrame()
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        for feature_type in cfg.cas.feature_types:
            runner = ReCaSRunner(cfg, feature_type)
            best_params = best_params_dict[feature_type]
            method_name = runner._get_method_name(False)
            print(f"Best parameters for '{method_name}': {best_params}")
            tmp_result_df = runner.run(best_params)
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
    best_params_dict = {}
    for feature_type in cfg.cas.feature_types:
        print(f"Feature type: {feature_type}", file=sys.stderr)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, feature_type, cfg),
            n_trials=cfg.cas.optuna.n_trials,
            n_jobs=cfg.cas.optuna.n_jobs,
        )
        first_params = {
            k[len("first_") :]: v
            for k, v in study.best_params.items()
            if k.startswith("first_")
        }
        second_params = {
            k[len("second_") :]: v
            for k, v in study.best_params.items()
            if k.startswith("second_") and v is not None
        }
        best_params_dict[feature_type] = (
            [first_params, second_params] if second_params else [first_params]
        )

    run(cfg, best_params_dict)
