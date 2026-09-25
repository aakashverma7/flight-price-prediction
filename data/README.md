# Data

The data file is not committed. Put it at `data/Clean_Dataset.csv`.

Source: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
by Shubham Bathwal on Kaggle, license CC0: Public Domain. Economy and business fares
scraped from EaseMyTrip for flights between Delhi, Mumbai, Bangalore, Kolkata,
Hyderabad and Chennai, 11 Feb - 31 Mar 2022.

To get it: sign in to Kaggle, open the dataset page, click Download, unzip, and copy
`Clean_Dataset.csv` into this folder. The zip also contains `economy.csv` and
`business.csv`; they are not used.

## Clean_Dataset.csv

300,153 rows, one fare quote per row.

| Column | Used as | Notes |
|---|---|---|
| (unnamed first column) | dropped | row index |
| `airline` | categorical feature | |
| `flight` | split group only | flight code such as `SG-8709`; not a feature |
| `source_city` | categorical feature | |
| `departure_time` | categorical feature | time-of-day bucket, `Early_Morning` to `Late_Night` |
| `stops` | categorical feature | `zero`, `one`, `two_or_more` |
| `arrival_time` | categorical feature | time-of-day bucket |
| `destination_city` | categorical feature | |
| `class` | filter | only `Economy` rows are kept (206,666) |
| `duration` | numeric feature | hours |
| `days_left` | numeric feature | days between the fare quote and departure |
| `price` | target | fare in INR, modelled as log(price) |
