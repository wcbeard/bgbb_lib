from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from lifetimes import BetaGeoBetaBinomFitter
from numpy import exp
from scipy.special import betaln, gammaln

from bgbb.bgbb_likelihood import nb_loglikelihood
from bgbb.bgbb_utils import AbgdParams
from bgbb.numba_special import cond_exp_rets_till_p2345, p_alive_exp_p1_p2
from bgbb.wrappers import Rfn, frt, to_abgd_od

Prob = float  # float: [0, 1]


def as_array(s: Union[np.array, pd.Series]):
    try:
        return s.values
    except AttributeError:
        return s


class BGBB(BetaGeoBetaBinomFitter):
    def __init__(
        self, penalizer_coef=0, params: Optional[List[float]] = None, data=None
    ):
        if params is not None:
            self.params_ = to_abgd_od(params)
            self.Params = AbgdParams(*self.params_.values())
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
        cls, params, frequency, recency, n, n_custs, penalizer_coef=0
    ):
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return (
            -np.mean(
                cls._loglikelihood(params, frequency, recency, n) * n_custs
            )
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
    def cond_prob_alive_nb(cls, frequency, recency, n, params, n_days_later=0):
        "Numba Version"
        x, tx = frequency, recency
        xa, _, na = map(as_array, [x, tx, n])
        p3 = cls._loglikelihood(params, x, tx, n)

        return p_alive_exp_p1_p2(params, xa, na, n_days_later, p3)

    def cond_exp_rets_till(self, t, frequency, recency, n, params):
        x, tx = frequency, recency
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = (
            delta
            / (gamma - 1)
            * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        )
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + t) - gammaln(gamma + delta + n + t))

        return p1 * p2 * p3 * (p4 - p5)

    def cond_exp_rets_till_nb(self, t, frequency, recency, n, params):
        xa, txa, na = map(as_array, [frequency, recency, n])

        p1 = 1 / exp(self._loglikelihood(params, xa, txa, na))
        ret = cond_exp_rets_till_p2345(t, xa, txa, na, params, p1)
        return ret

    @classmethod
    def _log_expec_p_th(cls, a: float, b: float, g: float, d: float, f, r, n):
        l_ap1 = cls._loglikelihood([a + 1, b, g, d], f, r, n)
        l_gp1 = cls._loglikelihood([a, b, g + 1, d], f, r, n)
        ll = cls._loglikelihood([a, b, g, d], f, r, n)

        l_a_frac = np.log(a) - np.log(a + b)
        l_g_frac = np.log(g) - np.log(g + d)

        return (l_a_frac + l_ap1 - ll), (l_g_frac + l_gp1 - ll)

    @classmethod
    def _expec_p_th(cls, a, b, g, d, f, r, n):
        lp_exp, lth_exp = cls._log_expec_p_th(a, b, g, d, f, r, n)
        return np.exp(lp_exp), np.exp(lth_exp)

    def latent_variable_mean(
        self, f, r, n
    ) -> Tuple[Sequence[Prob], Sequence[Prob]]:
        """
        Expected values of users' latent transaction and dropout probabilities
        eqns (13), (14) from "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
        Fader 2009.
        """
        return self._expec_p_th(*self.Params, f, r, n)
