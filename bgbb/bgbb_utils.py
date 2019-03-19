from collections import OrderedDict
from itertools import count
from typing import List, Optional, Sequence, Tuple

import numpy as np

# sim_dat = gen_buy_die(n_opps, n_users, abgd=SimPars.lst, p_th=[pp, th])
import numpy.random as nr

from numba import njit
from pandas import DataFrame

Prob = float


###############
# Simulations #
###############
def _gen_probs(abgd: List[float], n=10):
    pa, pb, ta, tb = abgd
    p = nr.beta(pa, pb, size=n)
    th = nr.beta(ta, tb, size=n)
    return p, th


def gen_buy_die(
    n_opps,
    n_users,
    abgd: List[float],
    p_th: Optional[Tuple[Sequence[Prob], Sequence[Prob]]] = None,
    n_opps_name="n_opps",
    seed=0,
):
    """
    """
    nr.seed(seed)
    if p_th is None:
        raise NotImplementedError
        # to dict
        p, th = _gen_probs(abgd, n=n_users)
    else:
        p, th = p_th
    txs, xs = np.empty_like(p, dtype=np.int), np.empty_like(p, dtype=np.int)
    bern = lambda p: nr.binomial(1, p, size=n_opps)

    for i, pi, thi in zip(count(), p, th):
        buys, dies = bern(pi), bern(thi)
        xs[i], txs[i] = get_x_tx(buys, dies)

    ret = DataFrame(
        OrderedDict([("p", p), ("th", th), ("frequency", xs), ("recency", txs)])
    ).assign(**{n_opps_name: n_opps})
    return ret


@njit
def get_x_tx(buys, dies) -> Tuple[int, int]:
    """
    Converts simulated 'buy'/'die' events into frequency/recency
    counts.
    """
    x = 0
    tx = 0
    for opp, buy, die in zip(range(1, len(buys) + 1), buys, dies):
        if die:
            break
        if buy:
            x += 1
            tx = opp
    return x, tx


class AbgdParams:
    def __init__(self, a, b, g, d):
        self.dct = OrderedDict(zip("abgd", [a, b, g, d]))

    @classmethod
    def from_lst(cls, abgd):
        return cls(*abgd)

    @classmethod
    def from_dct(cls, dct):
        return cls(*[dct[k] for k in "abgd"])

    @property
    def lst(self):
        return [self.dct[k] for k in "abgd"]

    @property
    def named_dct(self):
        dct = self.dct.copy()
        for name, letter in zip(["alpha", "beta", "gamma", "delta"], "abgd"):
            dct[name] = dct.pop(letter)
        return dct

    @property
    def greek_dct(self):
        dct = self.dct.copy()
        for name, letter in zip(["α", "β", "γ", "δ"], "abgd"):
            dct[name] = dct.pop(letter)
        return dct

    l = lst
    n = named_dct

    @property
    def to_dict(self):
        return self.dct

    def mod_param(self, **par_fns):
        dct = self.dct.copy()
        for par_letter, f in par_fns.items():
            dct[par_letter] = f(dct[par_letter])
        return self.from_dct(dct)

    def __repr__(self):
        st_repr = ", ".join(
            "{}: {:.2f}".format(k, v) for k, v in self.greek_dct.items()
        )
        return "BGBB Hyperparams <{}>".format(st_repr)
