# import math
from numba import njit, prange
import numpy as np
from numpy import log, exp, logaddexp
from scipy.special import betaln

from bgbb.numba_special import nb_lbeta_vec12, nb_lbeta_vec2


def get_vals(x):
    return x.values if hasattr(x, "values") else x


@njit
def nbsum_scalar(x, tx, recency_T, alpha, beta, gamma, delta, J, beta_ab, beta_gd):
    if recency_T <= -1:
        return -np.inf

    j = J[: recency_T + 1]
    return log(
        np.sum(
            exp(
                nb_lbeta_vec2(alpha + x, beta + tx - x + j)
                - beta_ab
                + nb_lbeta_vec2(gamma + 1, delta + tx + j)
                - beta_gd
            )
        )
    )


@njit
def nbsum_(x, tx, recency_T, a, b, g, d, J, ab, gd):
    N = len(x)
    res = np.empty(N)
    for i in range(N):
        res[i] = nbsum_scalar(x[i], tx[i], recency_T[i], a, b, g, d, J, ab, gd)
    return res


@njit(parallel=True)
def nbsum_p(x, tx, recency_T, a, b, g, d, J, ab, gd):
    N = len(x)
    res = np.empty(N)
    for i in prange(N):
        res[i] = nbsum_scalar(x[i], tx[i], recency_T[i], a, b, g, d, J, ab, gd)
    return res


def nbsum(x, tx, recency_T, a, b, g, d, J, ab, gd, para=True):
    f = nbsum_p if para else nbsum_
    return f(x, tx, recency_T, a, b, g, d, J, ab, gd)


def nb_loglikelihood_(alpha, beta, gamma, delta, x, tx, T, para=True):
    """Loglikelihood for optimizer."""
    # print("Using faster Loglikelihood")

    # Constrain the params to be > 0
    if min([alpha, beta, gamma, delta]) <= 0:
        return np.full_like(x, -np.inf)

    beta_ab = betaln(alpha, beta)
    beta_gd = betaln(gamma, delta)

    # indiv_loglike = (
    #     betaln(alpha + x, beta + T - x) - beta_ab + betaln(gamma, delta + T) - beta_gd
    # )

    recency_T = T - tx - 1
    xa, txa, Ta, recency_Ta = map(get_vals, [x, tx, T, recency_T])

    indiv_loglike = (
        nb_lbeta_vec12(alpha + xa, beta + Ta - xa)
        - beta_ab
        + nb_lbeta_vec2(gamma, delta + Ta)
        - beta_gd
    )

    # indiv_loglike = indiv_ll(alpha, beta, gamma, delta, xa, Ta, beta_ab, beta_gd)
    # assert np.allclose(indiv_loglike, indiv_loglike1)
    # print('.')

    J = np.arange(recency_T.max() + 1)

    # s = _sum(x, tx, recency_T)
    s = nbsum(
        xa, txa, recency_Ta, alpha, beta, gamma, delta, J, beta_ab, beta_gd, para=para
    )

    indiv_loglike = logaddexp(indiv_loglike, s)

    return indiv_loglike


############
# Wrappers #
############
def nb_loglikelihood(params, x, tx, T, para=True):
    alpha, beta, gamma, delta = params
    return nb_loglikelihood_(alpha, beta, gamma, delta, x, tx, T, para=para)


def nb_loglikelihood_df(params, df, rfn_names=False, para=True, nll=False, ncusts=None):
    if rfn_names:
        cnames = ["frequency", "recency", "n"]
    else:
        cnames = ["x", "tx", "T"]
    x, tx, T = [df[c] for c in cnames]
    alpha, beta, gamma, delta = params
    ll = nb_loglikelihood_(alpha, beta, gamma, delta, x, tx, T, para=para)
    if nll:
        if ncusts is None:
            print("WARNING: ncusts not passed")
            ncusts = 1
        return -np.mean(ll * ncusts)
    return ll


#########
# Tests #
#########
def test_nb_loglikelihood():
    res = nb_loglikelihood([.1, .1, .2, .3], np.r_[3], np.r_[4], np.r_[5])
    assert np.allclose(res, [-6.31032992])


# test_nb_lbeta()
# test_nb_lbeta_vec2()
