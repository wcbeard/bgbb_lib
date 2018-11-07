from typing import Union

from scipy.special import betaln, gammaln
import numpy as np
from numpy import exp
import pandas as pd
from lifetimes import BetaGeoBetaBinomFitter

from bgbb.wrappers import frt, Rfn, to_abgd_od
from bgbb.bgbb_likelihood import nb_loglikelihood
from bgbb.numba_special import p_alive_exp_p1_p2, cond_exp_rets_till_p2345


def as_array(s: Union[np.array, pd.Series]):
    try:
        return s.values
    except AttributeError:
        return s


class BGBB(BetaGeoBetaBinomFitter):
    def __init__(self, penalizer_coef=0, params=None, data=None):
        if params is not None:
            self.params_ = to_abgd_od(params)
        if data is not None:
            self.data = data
        self.rfn = Rfn(self)
        super().__init__(penalizer_coef=penalizer_coef)

    # conditional expected number of events up to time
    c_exp_num_ev_to = frt(
        BetaGeoBetaBinomFitter.conditional_expected_number_of_purchases_up_to_time
    )

    _loglikelihood = staticmethod(nb_loglikelihood)
    old_loglikelihood = BetaGeoBetaBinomFitter._loglikelihood

    @classmethod
    def _negative_log_likelihood(
        kls, params, frequency, recency, n, n_custs, penalizer_coef=0
    ):
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return (
            -np.mean(kls._loglikelihood(params, frequency, recency, n) * n_custs)
            + penalizer_term
        )

    @classmethod
    def cond_prob_alive(self, frequency, recency, n, params, n_days_later=0):
        """
        Version to be able to explicitly pass RFN columns.
        Conditional probability alive.

        From https://github.com/CamDavidsonPilon/lifetimes:
        Conditional probability customer is alive at transaction opportunity
        n + n_days_later.

        .. math:: P(alive at n + n_days_later
                    |alpha, beta, gamma, delta, frequency, recency, n)

        See (A10) in Fader and Hardie 2010.

        Parameters
        ----------
        n_days_later: array_like
            transaction opportunities

        Returns
        -------
        array_like
            alive probabilities

        """
        alpha, beta, gamma, delta = params
        x, tx = frequency, recency

        p1 = betaln(alpha + x, beta + n - x) - betaln(alpha, beta)
        p2 = betaln(gamma, delta + n + n_days_later) - betaln(gamma, delta)
        p3 = self._loglikelihood(params, x, tx, n)

        return exp(p1 + p2) / exp(p3)

    @classmethod
    def cond_prob_alive_nb(kls, frequency, recency, n, params, n_days_later=0):
        "Numba Version"
        x, tx = frequency, recency
        xa, txa, na = map(as_array, [x, tx, n])
        p3 = kls._loglikelihood(params, x, tx, n)

        return p_alive_exp_p1_p2(params, xa, na, n_days_later, p3)

    def cond_exp_rets_till(self, t, frequency, recency, n, params):
        x, tx = frequency, recency
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + t) - gammaln(gamma + delta + n + t))

        return p1 * p2 * p3 * (p4 - p5)

    def cond_exp_rets_till_nb(self, t, frequency, recency, n, params):
        xa, txa, na = map(as_array, [frequency, recency, n])

        p1 = 1 / exp(self._loglikelihood(params, xa, txa, na))
        ret = cond_exp_rets_till_p2345(t, xa, txa, na, params, p1)
        return ret
