"""Economy-class airfare models on the Kaggle EaseMyTrip data (Feb-Mar 2022).

All models are fitted on log(price) and scored in rupees on flight codes that
never appear in the training set.
"""

import argparse
import datetime as dt
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

CATEGORICAL = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
]
NUMERIC = ["duration", "days_left"]
FEATURES = CATEGORICAL + NUMERIC
GROUP = "flight"
TARGET = "price"
SEED = 42
TEST_SIZE = 0.2
EXTRA_SEEDS = (0, 1, 2, 3, 7)
BASELINE = "Baseline: training median"
MODELS = [BASELINE, "OLS", "Ridge", "PCR", "RandomForest", "XGBoost"]
TREES = ["RandomForest", "XGBoost"]
# XGBoost row subsampling gives slightly different trees for a different thread count,
# so the default is pinned to keep reported scores reproducible across machines.
XGB_THREADS = 4
PACKAGES = ["numpy", "pandas", "scikit-learn", "statsmodels", "xgboost", "matplotlib"]
RUPEES = FuncFormatter(lambda v, _: f"{v:,.0f}")


class StatsmodelsOLS(RegressorMixin, BaseEstimator):
    """statsmodels OLS behind the sklearn fit/predict interface, so it can sit
    in a pipeline and still give a full coefficient summary."""

    def fit(self, X, y):
        self.result_ = sm.OLS(np.asarray(y), sm.add_constant(X, has_constant="add")).fit()
        return self

    def predict(self, X):
        return np.asarray(self.result_.predict(sm.add_constant(X, has_constant="add")))


def load_economy(path: Path) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    df = pd.read_csv(path)
    missing = set(FEATURES + [GROUP, TARGET, "class"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    steps = [(f"rows in {path.name}", len(df))]
    df = df[df["class"] == "Economy"]
    steps.append(('class == "Economy"', len(df)))
    df = df.dropna(subset=FEATURES + [GROUP, TARGET])
    steps.append(("no missing values", len(df)))
    df = df[df[TARGET] > 0]
    steps.append(("price > 0", len(df)))

    for label, rows in steps:
        print(f"{label:<32} {rows:>9,}")
    return df[FEATURES + [GROUP, TARGET]].reset_index(drop=True), steps


def shared_flight_count(train: pd.DataFrame, test: pd.DataFrame) -> int:
    return len(set(train[GROUP]) & set(test[GROUP]))


def split_by_flight(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP]))
    return df.iloc[train_idx], df.iloc[test_idx]


def encoder(drop_first: bool = False, scale: bool = True) -> ColumnTransformer:
    # OLS needs one dummy dropped per feature or the design matrix is singular.
    onehot = OneHotEncoder(
        drop="first" if drop_first else None,
        handle_unknown="ignore",
        sparse_output=False,
    )
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", onehot, CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def build_models(n_jobs: int | None = None) -> dict:
    rf_jobs = -1 if n_jobs is None else n_jobs
    xgb_jobs = XGB_THREADS if n_jobs is None else n_jobs
    return {
        BASELINE: DummyRegressor(strategy="median"),
        # Unscaled numerics so the OLS coefficients read as log-price per hour / per day.
        "OLS": make_pipeline(
            encoder(drop_first=True, scale=False).set_output(transform="pandas"),
            StatsmodelsOLS(),
        ),
        "Ridge": make_pipeline(encoder(), Ridge(alpha=1.0)),
        "PCR": make_pipeline(encoder(), PCA(n_components=0.95), LinearRegression()),
        "RandomForest": make_pipeline(
            encoder(),
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=rf_jobs, random_state=SEED
            ),
        ),
        "XGBoost": make_pipeline(
            encoder(),
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=xgb_jobs,
            ),
        ),
    }


def score(price: pd.Series, log_pred: np.ndarray) -> dict:
    pred = np.exp(log_pred)
    return {
        "MAE (INR)": mean_absolute_error(price, pred),
        "RMSE (INR)": root_mean_squared_error(price, pred),
        "MAPE (%)": 100 * mean_absolute_percentage_error(price, pred),
        "R2 (log price)": r2_score(np.log(price), log_pred),
    }


def fit_and_score(names, train, test, n_jobs=None, outputs_dir=None, verbose=True):
    models = build_models(n_jobs)
    rows, preds, pcr_components = [], {}, None
    for name in names:
        # pop, so a large fitted forest is freed before the next model is fitted
        model = models.pop(name)
        start = time.perf_counter()
        model.fit(train[FEATURES], np.log(train[TARGET]))
        log_pred = model.predict(test[FEATURES])
        preds[name] = np.exp(log_pred)
        rows.append({"Model": name, **score(test[TARGET], log_pred)})
        if verbose:
            print(f"  {name:<28} fitted in {time.perf_counter() - start:6.1f} s")
        if name == "OLS" and outputs_dir is not None:
            # Rows of one flight code are not independent, so the standard errors
            # in the saved summary are clustered by flight code.
            codes = pd.factorize(train[GROUP])[0]
            clustered = model[-1].result_.model.fit(cov_type="cluster", cov_kwds={"groups": codes})
            summary = clustered.summary().as_text()
            (outputs_dir / "ols_summary.txt").write_text(summary, encoding="utf-8")
        elif name == "PCR":
            pcr_components = model[1].n_components_
            if verbose:
                print(f"  PCR keeps {pcr_components} components")
    return pd.DataFrame(rows), preds, pcr_components


def extra_grouped_seeds(df: pd.DataFrame, n_jobs=None) -> pd.DataFrame:
    rows = []
    for seed in EXTRA_SEEDS:
        train, test = split_by_flight(df, seed)
        shared = shared_flight_count(train, test)
        if shared:
            raise RuntimeError(f"{shared} flight codes are in both train and test (seed {seed})")
        table, _, _ = fit_and_score(TREES, train, test, n_jobs)
        maes = dict(zip(table["Model"], table["MAE (INR)"], strict=True))
        rows.append(
            {
                "Seed": seed,
                **{f"{name} MAE (INR)": maes[name] for name in TREES},
            }
        )
        print(f"  seed {seed}: " + ", ".join(f"{name} {maes[name]:,.0f}" for name in TREES))
    per_seed = pd.DataFrame(rows)
    stats = per_seed.drop(columns="Seed")
    extra = pd.DataFrame(
        [
            {"Seed": "min", **stats.min().to_dict()},
            {"Seed": "max", **stats.max().to_dict()},
            {"Seed": "mean", **stats.mean().to_dict()},
        ]
    )
    return pd.concat([per_seed, extra], ignore_index=True)


def fmt(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in out.columns:
        if col.endswith("(%)"):
            out[col] = out[col].map(lambda v: f"{v:.1f}")
        elif col.startswith("R2"):
            # + 0.0 turns -0.0 into 0.0, so a tiny negative R2 prints as 0.000
            out[col] = out[col].map(lambda v: f"{round(v, 3) + 0.0:.3f}")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:,.0f}")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:,}")
    return out


def markdown_table(table: pd.DataFrame) -> str:
    right = [pd.api.types.is_numeric_dtype(table[c]) for c in table.columns]
    table = fmt(table).astype(str)
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join("---:" if r else "---" for r in right) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def plot_pred_vs_actual(actual, pred, model_name: str, path: Path) -> None:
    fig = Figure(figsize=(6.4, 5.4), dpi=150)
    ax = fig.subplots()
    hb = ax.hexbin(
        actual,
        pred,
        gridsize=60,
        xscale="log",
        yscale="log",
        bins="log",
        mincnt=1,
        cmap="Blues",
    )
    lo = min(actual.min(), pred.min()) * 0.9
    hi = max(actual.max(), pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1000, 2000, 5000, 10000, 20000, 40000) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_major_formatter(RUPEES)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Actual price (INR, log scale)")
    ax.set_ylabel("Predicted price (INR, log scale)")
    ax.set_title(f"{model_name} on held-out flight codes")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="test rows per cell")
    fig.tight_layout()
    fig.savefig(path)


def plot_mae_by_airline(by_airline: pd.DataFrame, model_name: str, path: Path) -> None:
    data = by_airline.sort_values(f"{model_name} MAE (INR)")
    labels = [f"{a} (n={n:,})" for a, n in zip(data["Airline"], data["Test rows"], strict=True)]
    y = np.arange(len(data))
    h = 0.4
    fig = Figure(figsize=(7.0, 0.55 * len(data) + 1.6), dpi=150)
    ax = fig.subplots()
    ax.barh(y - h / 2, data[f"{model_name} MAE (INR)"], height=h, label=model_name)
    ax.barh(y + h / 2, data["Baseline MAE (INR)"], height=h, label=BASELINE)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(RUPEES)
    ax.set_xlabel("Mean absolute error (INR)")
    ax.set_title("Error by airline, held-out flight codes")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    fig.tight_layout()
    fig.savefig(path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/Clean_Dataset.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="scratch output (OLS summary); not committed",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="threads for RandomForest and XGBoost "
        f"(default: all cores for RandomForest, {XGB_THREADS} for XGBoost)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    started = time.perf_counter()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    df, row_steps = load_economy(args.data)

    train, test = split_by_flight(df)
    shared = shared_flight_count(train, test)
    if shared:
        raise RuntimeError(f"{shared} flight codes are in both train and test")
    split_info = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_flights": train[GROUP].nunique(),
        "test_flights": test[GROUP].nunique(),
    }
    print(
        f"Grouped split: train {split_info['train_rows']:,} rows / "
        f"{split_info['train_flights']:,} flights, test {split_info['test_rows']:,} rows / "
        f"{split_info['test_flights']:,} flights, shared flights: {shared}"
    )
    grouped, preds, pcr_components = fit_and_score(
        MODELS, train, test, args.n_jobs, args.outputs_dir
    )

    r_train, r_test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    seen = r_test[GROUP].isin(set(r_train[GROUP])).mean()
    print(
        f"Random split: train {len(r_train):,} rows, test {len(r_test):,} rows, "
        f"{100 * seen:.1f}% of test rows have a flight code seen in training"
    )
    random_split, _, _ = fit_and_score(TREES, r_train, r_test, args.n_jobs)
    leak = pd.concat(
        [
            grouped[grouped["Model"].isin(TREES)].assign(Split="grouped by flight"),
            random_split.assign(Split="random rows"),
        ]
    )
    leak = leak[["Model", "Split", "MAE (INR)", "RMSE (INR)", "MAPE (%)", "R2 (log price)"]]
    leak = leak.sort_values(["Model", "Split"]).reset_index(drop=True)

    extra_seeds = ", ".join(str(s) for s in EXTRA_SEEDS)
    print(f"Grouped split seeds {extra_seeds} (RandomForest and XGBoost only):")
    seed_table = extra_grouped_seeds(df, args.n_jobs)

    best = grouped[grouped["Model"] != BASELINE].sort_values("MAE (INR)").iloc[0]["Model"]
    abs_err = pd.DataFrame(
        {
            "Airline": test["airline"].to_numpy(),
            "best": np.abs(preds[best] - test[TARGET].to_numpy()),
            "base": np.abs(preds[BASELINE] - test[TARGET].to_numpy()),
        }
    )
    by_airline = (
        abs_err.groupby("Airline")
        .agg(rows=("best", "size"), best=("best", "mean"), base=("base", "mean"))
        .reset_index()
        .rename(
            columns={
                "rows": "Test rows",
                "best": f"{best} MAE (INR)",
                "base": "Baseline MAE (INR)",
            }
        )
        .sort_values(f"{best} MAE (INR)")
        .reset_index(drop=True)
    )

    plot_pred_vs_actual(
        test[TARGET].to_numpy(), preds[best], best, args.results_dir / "pred_vs_actual.png"
    )
    plot_mae_by_airline(by_airline, best, args.results_dir / "mae_by_airline.png")

    run_date = dt.datetime.now(dt.UTC).astimezone().date().isoformat()
    main_table = markdown_table(grouped)
    leak_table = markdown_table(leak)
    seeds_table = markdown_table(seed_table)
    airline_table = markdown_table(by_airline)
    rows_table = markdown_table(pd.DataFrame(row_steps, columns=["Step", "Rows"]))
    versions = ", ".join(f"{p} {version(p)}" for p in PACKAGES)
    xgb_threads = XGB_THREADS if args.n_jobs is None else args.n_jobs
    size_mb = args.data.stat().st_size / 1e6
    runtime_min = (time.perf_counter() - started) / 60

    report = f"""# Results

Run on {run_date}.

## Test set: flight codes not seen in training

{main_table}

Models are fitted on log(price); predictions are exp(predicted log price), so they
estimate the median fare for a set of features rather than the mean. MAE, RMSE and
MAPE are in rupees on the original price scale; R2 is on log price.
Lowest MAE on this seed-{SEED} split: {best}.

## Random split vs grouped split

Same hyperparameters, refitted on a plain random 80/20 row split
(train_test_split, seed {SEED}). In that split {100 * seen:.1f}% of test rows have a
flight code that also appears in training.

{leak_table}

## Extra grouped-split seeds (RandomForest and XGBoost)

Same hyperparameters and features, GroupShuffleSplit with seeds {extra_seeds}.
The table above is the seed-{SEED} split. Min, max and mean are over these seeds,
not a confidence interval.

{seeds_table}

## MAE by airline ({best} vs baseline, grouped test set)

{airline_table}

## Data

File: `{args.data.name}`, {size_mb:.1f} MB.
Source: Kaggle "Flight Price Prediction" (EaseMyTrip fares, 11 Feb - 31 Mar 2022).

{rows_table}

Features: {", ".join(CATEGORICAL)} (one-hot); {", ".join(NUMERIC)} (numeric).
The flight code is used only as the split group.

## Split

GroupShuffleSplit on flight code, {TEST_SIZE:.0%} of flight codes held out, seed {SEED}.

| Set | Rows | Flight codes |
|---|---:|---:|
| Train | {split_info["train_rows"]:,} | {split_info["train_flights"]:,} |
| Test | {split_info["test_rows"]:,} | {split_info["test_flights"]:,} |

Flight codes in both sets: {shared}.

## Models

Fixed hyperparameters, no tuning. Random seed {SEED} for the splits, RandomForest and XGBoost.

- Baseline: DummyRegressor(strategy="median") on log price
- OLS: statsmodels, one-hot with the first level dropped, unscaled numerics
- Ridge: alpha=1.0, full one-hot, scaled numerics
- PCR: PCA keeping 95% of variance ({pcr_components} components on the main split), then
  linear regression
- RandomForest: 200 trees, min_samples_leaf=2
- XGBoost: 400 rounds, learning_rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8,
  {xgb_threads} threads (scores shift slightly with the thread count)

## Environment

Python {platform.python_version()}; {versions}

Run time: {runtime_min:.1f} min on {os.cpu_count()} logical CPUs.
"""
    (args.results_dir / "metrics.md").write_text(report, encoding="utf-8")

    for table in (main_table, leak_table, seeds_table, airline_table):
        print()
        print(table)
    print(f"\nWrote {args.results_dir / 'metrics.md'} and 2 figures")
    print(f"Run time: {runtime_min:.1f} min")
    return grouped


if __name__ == "__main__":
    main()
