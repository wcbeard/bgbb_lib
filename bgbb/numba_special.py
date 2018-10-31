import math
from numba import njit
from numba import vectorize

from scipy.special import betaln, logsumexp
import numpy as np
from numpy import random as nr, exp


def get_vals(x):
    return x.values if hasattr(x, "values") else x


@njit
def nb_logsumexp(xs):
    mx = np.max(xs)
    s = 0
    for x in xs:
        s += np.exp(x - mx)
    return mx + np.log(s)


# TODO: delete?
# @njit
@vectorize
def lgamma(x):
    return math.lgamma(x)


@njit
def nb_lbeta(z, w):
    "https://www.mathworks.com/help/matlab/ref/betaln.html"
    return lgamma(z) + lgamma(w) - lgamma(z + w)


@njit
def nb_lbeta_vec2(z, w: np.array):
    N = len(w)
    res = np.empty(N)
    for i in range(N):
        res[i] = nb_lbeta(z, w[i])
    return res


@njit
def nb_lbeta_vec12(z: np.array, w: np.array):
    N = len(w)
    res = np.empty(N)
    for i in range(N):
        res[i] = nb_lbeta(z[i], w[i])
    return res


##############
# Components #
##############
@njit
def p_alive_exp_p1_p2(params, xa, na, n_days_later, p3):
    "For cond_prob_alive_nb"
    alpha, beta, gamma, delta = params
    p1nb = nb_lbeta(alpha + xa, beta + na - xa) - nb_lbeta(alpha, beta)
    p2nb = nb_lbeta(gamma, delta + na + n_days_later) - nb_lbeta(gamma, delta)
    return exp(p1nb + p2nb) / exp(p3)


@njit
def cond_exp_rets_till_p2345(t, frequency, recency, n, params, p1):
    alpha, beta, gamma, delta = params
    x = frequency

    p2 = exp(nb_lbeta(alpha + x + 1, beta + n - x) - nb_lbeta(alpha, beta))
    p3 = delta / (gamma - 1) * exp(lgamma(gamma + delta) -
                                   lgamma(1 + delta))
    p4 = exp(lgamma(1 + delta + n) - lgamma(gamma + delta + n))
    p5 = exp(lgamma(1 + delta + n + t) - lgamma(gamma + delta + n + t))

    return p1 * p2 * p3 * (p4 - p5)

#########
# Tests #
#########


def test_nb_lbeta(niter=1000):
    for _ in range(niter):
        d, g = nr.rand(2) * 1000
        a = nb_lbeta(d, g)
        b = betaln(d, g)
        assert np.allclose(a, b), "{} != {} ({}, {})".format(a, b, d, g)


def test_nb_lbeta_vec2():
    assert np.allclose(nb_lbeta_vec2(1, np.arange(7)), betaln(1, np.arange(7)))


def test_nb_logsumexp(n_iter=100):
    nr.seed(0)
    for _ in range(n_iter):
        a = 10 ** (nr.rand(100) * 10)
        assert np.allclose(nb_logsumexp(a), logsumexp(a))


# test_nb_lbeta()
# test_nb_lbeta_vec2()
