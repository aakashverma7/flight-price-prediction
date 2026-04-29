import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input file must be a CSV or Excel workbook.")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "flight" in df.columns:
        df = df.drop(columns=["flight"])

    if "price" not in df.columns:
        raise ValueError("Dataset must contain a `price` column.")

    df = df[df["price"] > 0].copy()
    df["log_price"] = np.log(df["price"])
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


def regression_metrics(y_true, predictions, model_name: str) -> dict:
    return {
        "model": model_name,
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
        "mae": mean_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regression models for flight price prediction.")
    parser.add_argument("--input-path", required=True, help="Path to the flight dataset.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(load_dataset(Path(args.input_path)))
    X = df.drop(columns=["price", "log_price"])
    y = df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_frame = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_test_frame = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

    rows = []

    ols_train = sm.add_constant(X_train_frame, has_constant="add")
    ols_test = sm.add_constant(X_test_frame, has_constant="add")
    ols_model = sm.OLS(y_train, ols_train).fit()
    ols_predictions = ols_model.predict(ols_test)
    rows.append(regression_metrics(y_test, ols_predictions, "ols"))
    (output_dir / "ols_summary.txt").write_text(ols_model.summary().as_text(), encoding="utf-8")

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_encoded, y_train)
    rows.append(regression_metrics(y_test, ridge_model.predict(X_test_encoded), "ridge"))

    pcr_model = Pipeline(
        steps=[
            ("pca", PCA(n_components=0.95)),
            ("regressor", LinearRegression()),
        ]
    )
    pcr_model.fit(X_train_encoded, y_train)
    rows.append(regression_metrics(y_test, pcr_model.predict(X_test_encoded), "pcr"))

    rf_model = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    rf_model.fit(X_train_encoded, y_train)
    rows.append(regression_metrics(y_test, rf_model.predict(X_test_encoded), "random_forest"))

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )
    xgb_model.fit(X_train_encoded, y_train)
    rows.append(regression_metrics(y_test, xgb_model.predict(X_test_encoded), "xgboost"))

    metrics = pd.DataFrame(rows).sort_values("rmse")
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
