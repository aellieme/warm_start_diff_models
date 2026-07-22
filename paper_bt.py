#!/usr/bin/env python3
"""
Literal Bayesian Bradley–Terry model from equation (1) of the paper
"Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies".

The only adaptation is the input format:
    dataset,model,score

Your files:
    combined_k10.csv
    combined_k20.csv
    combined_k100.csv

Each file is ranked separately, exactly as the paper ranks a fixed
metric/cutoff separately.

Model from the paper:
    W_ij ~ Binomial(N_ij, exp(beta_i)/(exp(beta_i)+exp(beta_j)))
    beta_i ~ Normal(0, sigma_bar)
    sigma_bar ~ LogNormal(0, 0.5)

Inference:
    Metropolis-Hastings
Final ranking:
    descending posterior mean of beta_i

Install in Colab if needed:
    !pip install pymc arviz pandas numpy
"""

from __future__ import annotations

import argparse
import shutil
from itertools import combinations
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pymc as pm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------- SETTINGS YOU MAY CHANGE --------

INPUT_FILES = [
    "combined_k10.csv",
    "combined_k20.csv",
    "combined_k100.csv",
]

# The paper states Metropolis-Hastings but does not specify these run-length
# settings in the article text. They control Monte Carlo accuracy only;
# they do not change the statistical model.
DRAWS = 20_000
TUNE = 5_000
CHAINS = 4
RANDOM_SEED = 42

# Your score is Recall, so larger is better.
HIGHER_IS_BETTER = True

# Your files do not contain standard deviations, so only exact equal scores
# are treated as ties. A tie contributes 0.5 in each direction, as in the paper.
EXACT_TIES = True


def locate_file(filename: str, input_dir: Path | None = None) -> Path:
    """Find an input file in the requested folder or common Colab locations."""
    candidates = [
        *((input_dir / filename,) if input_dir is not None else ()),
        Path(filename),
        Path("/content") / filename,
        Path("/mnt/data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Не найден файл {filename!r}. Загрузите его в Colab "
        "или положите рядом со скриптом."
    )


def scores_to_win_matrix(
    score_table: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """
    Convert dataset/model/score rows into W.

    W[i, j] = number of datasets where model i beats model j.
    Exact ties add 0.5 to both W[i, j] and W[j, i].
    """
    required = {"dataset", "model", "score"}
    missing = required - set(score_table.columns)
    if missing:
        raise ValueError(
            f"Нет обязательных столбцов: {sorted(missing)}. "
            "Нужны dataset, model, score."
        )

    table = score_table[["dataset", "model", "score"]].copy()
    table["model"] = table["model"].astype(str)

    duplicated = table.duplicated(["dataset", "model"], keep=False)
    if duplicated.any():
        bad = table.loc[duplicated, ["dataset", "model"]]
        raise ValueError(
            "Для одной пары dataset/model найдено несколько строк:\n"
            + bad.to_string(index=False)
        )

    models = sorted(table["model"].unique().tolist())
    model_index = {model: i for i, model in enumerate(models)}
    W = np.zeros((len(models), len(models)), dtype=float)

    for dataset, group in table.groupby("dataset", sort=False):
        values = group.set_index("model")["score"].dropna()

        for model_i, model_j in combinations(values.index, 2):
            score_i = float(values.loc[model_i])
            score_j = float(values.loc[model_j])
            i = model_index[model_i]
            j = model_index[model_j]

            if EXACT_TIES and score_i == score_j:
                W[i, j] += 0.5
                W[j, i] += 0.5
            else:
                i_wins = (
                    score_i > score_j
                    if HIGHER_IS_BETTER
                    else score_i < score_j
                )
                if i_wins:
                    W[i, j] += 1.0
                else:
                    W[j, i] += 1.0

    return models, W


def fit_paper_bayesian_bt(
    models: list[str],
    W: np.ndarray,
):
    """
    Fit equation (1) from the paper with Metropolis-Hastings.
    """
    pair_i: list[int] = []
    pair_j: list[int] = []
    observed_wins: list[int] = []
    comparisons: list[int] = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            n_ij = W[i, j] + W[j, i]
            if n_ij == 0:
                continue

            # With no std columns and no exact equal scores, these are integers.
            # Binomial requires integer counts.
            if not float(W[i, j]).is_integer() or not float(n_ij).is_integer():
                raise ValueError(
                    "Обнаружены дробные ничьи. Для буквальной Binomial-модели "
                    "из статьи нужны целые W_ij. Ваши текущие файлы, судя по "
                    "формату, не должны создавать эту проблему."
                )

            pair_i.append(i)
            pair_j.append(j)
            observed_wins.append(int(W[i, j]))
            comparisons.append(int(n_ij))

    pair_i_arr = np.asarray(pair_i, dtype=int)
    pair_j_arr = np.asarray(pair_j, dtype=int)
    observed_arr = np.asarray(observed_wins, dtype=int)
    comparisons_arr = np.asarray(comparisons, dtype=int)

    coords = {"model": models, "pair": np.arange(len(pair_i_arr))}

    with pm.Model(coords=coords) as model:
        # Equation (1) in the paper:
        sigma_bar = pm.LogNormal("sigma_bar", mu=0.0, sigma=0.5)

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=sigma_bar,
            dims="model",
        )

        probability = pm.math.sigmoid(
            beta[pair_i_arr] - beta[pair_j_arr]
        )

        pm.Binomial(
            "wins",
            n=comparisons_arr,
            p=probability,
            observed=observed_arr,
            dims="pair",
        )

        step = pm.Metropolis()

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=min(CHAINS, 4),
            step=step,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            progressbar=False,
        )

    return trace


def make_outputs(
    models: list[str],
    W: np.ndarray,
    trace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_samples = (
        trace.posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "model")
        .values
    )

    beta_mean = beta_samples.mean(axis=0)
    beta_low = np.quantile(beta_samples, 0.025, axis=0)
    beta_high = np.quantile(beta_samples, 0.975, axis=0)

    # Per MCMC sample, convert beta to positive normalized BT strengths.
    shifted = beta_samples - beta_samples.max(axis=1, keepdims=True)
    strengths = np.exp(shifted)
    strengths /= strengths.sum(axis=1, keepdims=True)

    weight_mean = strengths.mean(axis=0)
    weight_low = np.quantile(strengths, 0.025, axis=0)
    weight_high = np.quantile(strengths, 0.975, axis=0)

    ranking = pd.DataFrame(
        {
            "model": models,
            "beta_mean": beta_mean,
            "beta_2.5%": beta_low,
            "beta_97.5%": beta_high,
            "weight_mean": weight_mean,
            "weight_2.5%": weight_low,
            "weight_97.5%": weight_high,
            "total_wins": W.sum(axis=1),
        }
    )

    # The paper determines the final ranking by mean beta over MCMC samples.
    ranking = (
        ranking
        .sort_values("beta_mean", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    # Posterior mean pairwise win probabilities.
    probabilities = pd.DataFrame(
        np.eye(len(models)) * 0.5,
        index=models,
        columns=models,
        dtype=float,
    )

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                probabilities.iloc[i, j] = 0.5
            else:
                p_samples = 1.0 / (
                    1.0 + np.exp(-(beta_samples[:, i] - beta_samples[:, j]))
                )
                probabilities.iloc[i, j] = p_samples.mean()

    return ranking, probabilities


def run_one_file(filename: str) -> None:
    path = locate_file(filename)
    table = pd.read_csv(path)

    models, W = scores_to_win_matrix(table)
    trace = fit_paper_bayesian_bt(models, W)
    ranking, probabilities = make_outputs(models, W, trace)

    stem = path.stem
    output_dir = Path("/content") if Path("/content").exists() else Path(".")

    win_path = output_dir / f"{stem}_win_matrix.csv"
    ranking_path = output_dir / f"{stem}_bt_ranking.csv"
    probability_path = output_dir / f"{stem}_bt_probabilities.csv"
    trace_path = output_dir / f"{stem}_bt_trace.nc"

    pd.DataFrame(W, index=models, columns=models).to_csv(
        win_path,
        index_label="model",
    )
    ranking.to_csv(ranking_path, index=False)
    probabilities.to_csv(probability_path, index_label="model")
    az.to_netcdf(trace, trace_path)

    print("\n" + "=" * 80)
    print(f"Файл: {path.name}")
    print("Финальный рейтинг по posterior mean beta:")
    print(
        ranking.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nСохранено:")
    print(ranking_path)
    print(win_path)
    print(probability_path)
    print(trace_path)


def main() -> None:
    for filename in INPUT_FILES:
        run_one_file(filename)


if __name__ == "__main__":
    main()
