import numpy as np
import pandas as pd

import flight_price_models as fpm

AIRLINES = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
TIMES = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS = ["zero", "one", "two_or_more"]


def synthetic_flights(n_flights: int = 60, rows_per_flight: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_flights):
        src, dst = rng.choice(CITIES, size=2, replace=False)
        flight = {
            "airline": AIRLINES[i % len(AIRLINES)],
            "flight": f"XX-{1000 + i}",
            "source_city": src,
            "departure_time": rng.choice(TIMES),
            "stops": rng.choice(STOPS),
            "arrival_time": rng.choice(TIMES),
            "destination_city": dst,
            "duration": round(float(rng.uniform(1, 20)), 2),
        }
        for days_left in rng.integers(1, 50, size=rows_per_flight):
            cls = "Business" if rng.random() < 0.15 else "Economy"
            price = 3000 + 150 * flight["duration"] + 4000 / days_left
            price *= 8 if cls == "Business" else 1
            price *= rng.lognormal(0, 0.1)
            rows.append({**flight, "class": cls, "days_left": int(days_left), "price": int(price)})
    df = pd.DataFrame(rows)
    df.insert(0, "Unnamed: 0", range(len(df)))
    return df


def test_pipeline_runs_on_synthetic_data(tmp_path):
    data = tmp_path / "Clean_Dataset.csv"
    synthetic_flights().to_csv(data, index=False)
    results = tmp_path / "results"
    outputs = tmp_path / "outputs"

    # One thread: on a few hundred rows, thread start-up and contention cost more than they save.
    args = {"--data": data, "--results-dir": results, "--outputs-dir": outputs, "--n-jobs": 1}
    metrics = fpm.main([str(x) for pair in args.items() for x in pair])

    assert list(metrics["Model"]) == fpm.MODELS
    numeric = metrics.drop(columns="Model").to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    assert (metrics["MAE (INR)"] > 0).all()

    expected = {"metrics.md", "pred_vs_actual.png", "mae_by_airline.png"}
    assert {p.name for p in results.iterdir()} == expected
    assert all((results / name).stat().st_size > 0 for name in expected)
    assert "cluster" in (outputs / "ols_summary.txt").read_text(encoding="utf-8")
    report = (results / "metrics.md").read_text(encoding="utf-8")
    assert "## Extra grouped-split seeds (RandomForest and XGBoost)" in report
    for seed in fpm.EXTRA_SEEDS:
        assert f"| {seed} |" in report
    assert "| min |" in report
    assert "| max |" in report
    assert "| mean |" in report
    overlap_lines = [
        line for line in report.splitlines() if line.startswith("Flight codes in both sets:")
    ]
    assert len(overlap_lines) == 1
    shared = int(overlap_lines[0].split(":")[1].strip().rstrip("."))
    assert shared >= 0
    assert "nan" not in report.lower()
