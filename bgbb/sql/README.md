bgbb_cli
==============================
Library to make it easier to apply BGBB model to databricks data.


# Structure
```
../bgbb_cli
├── README.md
├── __init__.py
├── bgbb_udfs.py
├── sql_utils.py
└── sql_utils_tests.py
```

- `sql_utils.py`: Defines functions to create and modify SQL queries
    - `run_rec_freq_spk`: given model and holdout date ranges, create a spark dataframe representing recency/frequency data from clients_daily that can be used by the BGBB model. Also return the query string used.
        - This uses the function `mk_rfn_q` to fill the dates and sample_id info into the query string
    - `reduce_rfn_spk`: Given the spark dataframe result of `run_rfn_spk`, reduce it to the number of recency/frequency patterns that can be used to find the maximum likelihood parameters during fitting.
- `bgbb_udfs.py`: Given number of days to predict returns in the future (`return_in_next_n_days`) and number of days in the future to predict if users are still active (`alive_n_days_later`), create vectorized spark UDFs that use BGBB functionality to compute retention quantities.
