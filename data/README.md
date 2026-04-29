# Data Notes

Data not included in this repository.

Place the source file in this folder as `Flight_dataset.xlsx` or `Flight_dataset.csv`.

Expected columns:

- `airline`
- `flight` (optional; dropped by the script)
- `source_city`
- `departure_time`
- `stops`
- `arrival_time`
- `destination_city`
- `duration`
- `days_left`
- `price`

If an additional column such as `class` is present, the script will automatically treat it as a feature.
