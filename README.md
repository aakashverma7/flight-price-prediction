# Flight Price Prediction

This project compares multiple regression models for airfare prediction using airline, route, and travel metadata.

Data not included in this repository.

## Models and Workflow

- OLS regression
- ridge regression
- principal component regression
- random forest regression
- XGBoost regression
- categorical encoding for airline and route features
- numerical scaling for duration and booking lead time
- log-transformed price modeling

## Project Structure

- `src/flight_price_models.py`: end-to-end preprocessing, training, and evaluation
- `data/README.md`: expected dataset file and schema
- `requirements.txt`: Python dependencies

## Quick Start

```bash
pip install -r requirements.txt
python src/flight_price_models.py --input-path data/Flight_dataset.xlsx
```

## Outputs

The script writes results under `outputs/`:

- `model_metrics.csv`
- `ols_summary.txt`
