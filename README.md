[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


bgbb_lib
==============================

Reimplementation of the BGBB model (Beta-geometric beta-binomial) based on the [lifetimes](http://lifetimes.readthedocs.io/) library using numba. The papers used for the derivations are [Fader/Hardie 2010](http://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf) and
[Fader & Hardie 2009](http://web-docs.stern.nyu.edu/old_web/emplibrary/Peter%20Fader.pdf). This runs in python 3.

# Terminology

Taken from [here](https://lifetimes.readthedocs.io/en/master/Quickstart.html#the-shape-of-your-data)

- _frequency_ represents the number of repeat purchases the customer has made. This means that it’s one less than the total number of purchases. This is actually slightly wrong. It’s the count of time periods the customer had a purchase in. So if using days as units, then it’s the count of days the customer had a purchase on.
- _T_ represents the age of the customer in whatever time units chosen (weekly, in the above dataset). This is equal to the duration between a customer’s first purchase and the end of the period under study.
- _recency_ represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer’s first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

Since the BGBB model is the discrete time version, _n_ is used instead of _T_, and represents the number of days a user could have potentially returned.


# Library Structure

```
bgbb_lib
├── README.md
├── bgbb
│   ├── __init__.py
│   ├── bgbb_likelihood.py
│   ├── core.py
│   ├── numba_special.py
│   ├── tests.py
│   └── wrappers.py
└── setup.py
```

- `bgbb_likelihood.py` defines the BGBB likelihood function in numba
    - relies on special functions redefined in `numba_special.py` in numba
- `core.py` defines the BGBB class with the likelihood function from `bgbb_likelihood.py` and some numba functions for computing probability alive and expected number of returns from `numba_special.py`
    - each of methods in this class take model parameters and series/arrays representing `frequency`, `recency`, `n`(umber of opportunities to return)
    - a version of the BGBB model that takes a pandas DataFrame (with `frequency`, `recency`, `n` columns) is in `wrappers.py`. This is just a terser version that's easier for quick interactive notebooks.


# Time definitions
The following diagram shows example event time labels for users `A, B, C, D` over days labeled 6-12.
```
         ┌───────────────────────────── Model start
         │           ┌───────────────── Holdout start
         │           │     ┌─────────── Holdout last day
         │           │     │
         │           │     │
         ▼           ▼     ▼                       Freq      Rec      T
User     6  7  8  9  10 11 12    Freq  Rec   T     Calc      Calc     Calc
────────────────────┬────────────────────────────
A        x        x │   x        1     3     3     2-1=1     9-6=3    9-6
B        x     x    │            1     2     3               8-6=2    9-6
C        x  x       │            1     1     3               7-6=1    9-6
D           x  x    │            1     1     2               8-7=1    9-7
  Rec (1)   ────
  T (2)     ───────
  Freq         ─

```
At the bottom, F/R/T labeled for final example (User D).


# Calculated quantities
So far, the calculated quantities of interest that the BGBB model can produce are

- conditional probability alive (A10 from Fader & Hardie 2009)
    - implemented in numba as `BGBB.cond_prob_alive_nb`
- expected number of future visits across the next n transaction opportunities ((12) from Fader & Hardie 2009)
    - implemented as `BGBB.cond_exp_rets_till_nb`

Other quantities that haven't been implemented but potentially could be are (using customer/transaction terminology from the paper)
- The probability that a customer with given purchase history has a given # of transactions (11)
    - i.e., probability that they'll return 3 times in the next 14 days
- mean of the marginal posterior distribution of _p_ (return rate) for a user (13)
- mean of the marginal posterior distribution of _θ_ (churn rate) for a user (14)

## Bundling
This library can be bundled into an egg for upload into databricks, using code from `a-4.0.1-bgbb-packaging.ipynb`.

CLI utils can also be copied into the databricks filesystem.

- Example stuff set up in new environment & dir `test_egg`. `bgbb` folder copied into there as well.

```
source activate test_egg
python setup.py bdist_egg
easy_install dist/bgbb-0.1-py3.6.egg
python -c "import bgbb; print(bgbb)"
python -c "from bgbb import bgbb"
```

# Databricks runtime restrictions

## 5.2 ML (Beta)
- python: 3.6.5
- spark_version: "5.2.x-cpu-ml-scala2.11"
    - this string breaks mozetl-databricks.py
- llvmlite: 0.23.1
    - pip can't uninstall this lib on Databricks
- numba: 0.38.0+0.g2a2b772fc.dirty
- https://docs.databricks.com/release-notes/runtime/5.2ml.html

## 5.2
- this library is currently configured to use this runtime version
- python: 3.5.2 (typing bug)
- spark_version: "4.3.x-scala2.11"
- llvmlite: 0.13.0
- numba: 0.28.1
    - `prange` not included until [0.34](https://numba.pydata.org/numba-doc/dev/release-notes.html#version-0-34-0)
- https://docs.databricks.com/release-notes/runtime/5.2.html

# License
Licensed under either
 - Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or
 - MIT license ([LICENSE-MIT](LICENSE-MIT))
