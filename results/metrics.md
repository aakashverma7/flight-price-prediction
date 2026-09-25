# Results

Run on 2026-09-26.

## Test set: flight codes not seen in training

| Model | MAE (INR) | RMSE (INR) | MAPE (%) | R2 (log price) |
|---|---:|---:|---:|---:|
| Baseline: training median | 2,537 | 3,759 | 42.7 | 0.000 |
| OLS | 1,697 | 2,539 | 26.8 | 0.565 |
| Ridge | 1,697 | 2,539 | 26.8 | 0.565 |
| PCR | 1,845 | 2,703 | 29.4 | 0.480 |
| RandomForest | 1,256 | 2,068 | 18.7 | 0.739 |
| XGBoost | 1,202 | 1,952 | 17.8 | 0.775 |

Models are fitted on log(price); predictions are exp(predicted log price), so they
estimate the median fare for a set of features rather than the mean. MAE, RMSE and
MAPE are in rupees on the original price scale; R2 is on log price.
Lowest MAE on this seed-42 split: XGBoost.

## Random split vs grouped split

Same hyperparameters, refitted on a plain random 80/20 row split
(train_test_split, seed 42). In that split 100.0% of test rows have a
flight code that also appears in training.

| Model | Split | MAE (INR) | RMSE (INR) | MAPE (%) | R2 (log price) |
|---|---|---:|---:|---:|---:|
| RandomForest | grouped by flight | 1,256 | 2,068 | 18.7 | 0.739 |
| RandomForest | random rows | 581 | 1,318 | 7.8 | 0.920 |
| XGBoost | grouped by flight | 1,202 | 1,952 | 17.8 | 0.775 |
| XGBoost | random rows | 998 | 1,690 | 14.6 | 0.849 |

## Extra grouped-split seeds (RandomForest and XGBoost)

Same hyperparameters and features, GroupShuffleSplit with seeds 0, 1, 2, 3, 7.
The table above is the seed-42 split. Min, max and mean are over these seeds,
not a confidence interval.

| Seed | RandomForest MAE (INR) | XGBoost MAE (INR) |
|---|---:|---:|
| 0 | 1,256 | 1,208 |
| 1 | 1,261 | 1,236 |
| 2 | 1,087 | 1,101 |
| 3 | 1,312 | 1,269 |
| 7 | 1,282 | 1,243 |
| min | 1,087 | 1,101 |
| max | 1,312 | 1,269 |
| mean | 1,239 | 1,212 |

## MAE by airline (XGBoost vs baseline, grouped test set)

| Airline | Test rows | XGBoost MAE (INR) | Baseline MAE (INR) |
|---|---:|---:|---:|
| GO_FIRST | 5,419 | 1,050 | 1,585 |
| Indigo | 7,843 | 1,062 | 2,379 |
| Vistara | 13,604 | 1,200 | 2,777 |
| Air_India | 10,645 | 1,279 | 2,731 |
| AirAsia | 2,258 | 1,360 | 3,061 |
| SpiceJet | 1,484 | 1,736 | 2,460 |

## Data

File: `Clean_Dataset.csv`, 24.7 MB.
Source: Kaggle "Flight Price Prediction" (EaseMyTrip fares, 11 Feb - 31 Mar 2022).

| Step | Rows |
|---|---:|
| rows in Clean_Dataset.csv | 300,153 |
| class == "Economy" | 206,666 |
| no missing values | 206,666 |
| price > 0 | 206,666 |

Features: airline, source_city, departure_time, stops, arrival_time, destination_city (one-hot); duration, days_left (numeric).
The flight code is used only as the split group.

## Split

GroupShuffleSplit on flight code, 20% of flight codes held out, seed 42.

| Set | Rows | Flight codes |
|---|---:|---:|
| Train | 165,413 | 1,248 |
| Test | 41,253 | 312 |

Flight codes in both sets: 0.

## Models

Fixed hyperparameters, no tuning. Random seed 42 for the splits, RandomForest and XGBoost.

- Baseline: DummyRegressor(strategy="median") on log price
- OLS: statsmodels, one-hot with the first level dropped, unscaled numerics
- Ridge: alpha=1.0, full one-hot, scaled numerics
- PCR: PCA keeping 95% of variance (23 components on the main split), then
  linear regression
- RandomForest: 200 trees, min_samples_leaf=2
- XGBoost: 400 rounds, learning_rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8,
  4 threads (scores shift slightly with the thread count)

## Environment

Python 3.13.13; numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, statsmodels 0.15.0, xgboost 3.4.1, matplotlib 3.11.2

Run time: 5.4 min on 20 logical CPUs.
