from collections import OrderedDict
from functools import wraps
import inspect
from typing import Dict, Union, List
import numpy as np

from lifetimes import BetaGeoBetaBinomFitter

abgd_names = "alpha beta gamma delta".split()


def unload(dct: Dict, ks: Union[List[str], str]) -> List[float]:
    if isinstance(ks, str):
        ks = ks.split()
    return [dct[k] for k in ks]


def frt(f):
    """Wraps lifetimes model methods that potentially
    take 'frequency', 'recency', 'T' as params,
    replace the func w/ one that takes a dataframe
    with these arguments as column names.
    """
    frt_params = {"frequency", "recency", "T", "n", "n_custs"}
    sig = inspect.signature(f)
    params = list(sig.parameters)

    @wraps(f)
    def wrapper(self, data, **k):
        k.update({p: data[p] for p in params if p in frt_params})
        return f(self, **k)

    return wrapper


def to_abgd_od(params) -> Dict[str, float]:
    if isinstance(params, OrderedDict):
        return params
    return OrderedDict(zip(abgd_names, params))


class Rfn:
    """TODO: document
    """

    def __init__(self, mod):
        self.mod = mod

    def cond_prob_alive(self, df, params: List[float]=None, n_days_later=0,
                        nb=True):
        params = mod_par_list(params, self.mod)
        frequency, recency, n = unload(df, "frequency recency n")
        kw = dict(
            frequency=frequency,
            recency=recency,
            n=n,
            params=params,
            n_days_later=n_days_later,
        )
        if nb:
            return self.mod.cond_prob_alive_nb(**kw)
        return self.mod.cond_prob_alive(**kw)
        # return self.mod.cond_prob_alive(
        #     frequency, recency, n, params, n_days_later=n_days_later
        # )

    @wraps(BetaGeoBetaBinomFitter.fit)
    def fit(self, df, **kw):
        frequency, recency, n = unload(df, "frequency recency n")
        if "n_custs" in df:
            n_custs = df["n_custs"]
        else:
            print("Warning: no n_custs column")
            n_custs = np.ones_like(frequency)
        return self.mod.fit(frequency, recency, n, n_custs, **kw)

    @wraps(BetaGeoBetaBinomFitter._loglikelihood)
    def _loglikelihood(self, df, params=None, para=True):
        x, tx, T = unload(df, "frequency recency n")
        params = mod_par_list(params, self.mod)
        return self.mod._loglikelihood(params, x, tx, T, para=para)

    @wraps(BetaGeoBetaBinomFitter
           .conditional_expected_number_of_purchases_up_to_time)
    def cond_exp_rets_till(self, df, n_days_later, params=None, nb=False):
        x, tx, n = unload(df, "frequency recency n")
        params = mod_par_list(params, self.mod)
        kw = dict(
            t=n_days_later, frequency=x, recency=tx, n=n, params=params,
        )
        if nb:
            return self.mod.cond_exp_rets_till_nb(**kw)
        return self.mod.cond_exp_rets_till(**kw)


def mod_par_list(params, mod):
    if params is not None:
        return params
    return list(mod.params_.values())
