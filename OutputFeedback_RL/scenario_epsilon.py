# originally from The Risk of Making Decisions from data through the lens of the scenario approach,
# https://marco-campi.unibs.it/pdf-pszip/2021-SYSID.pdf
# inspired by the matlab code found at the end of
# Wait and Judge Scenario Optimisation,
# https://marco-campi.unibs.it/pdf-pszip/Wait-and-judge.PDF, or https://doi.org/10.1007/s10107-016-1056-9
import timeit
import numpy as np
import scipy.special
from decimal import Decimal as dc


#################################################################################
# important note: this function is numerically unstable for values N > 1000
# use the Decimal one in that case, slower but accurate
#################################################################################
def eps_general_smallN(k, N, beta):

    """
    adapted from equation (3)
    (n k) * t**(N-k) - beta/N * sum_{i=k}^{N-1} (i k) * t**(i-k) = 0 ,
    which is

    1 - beta/N * (N k)**(-1) * sum_{i=k}^{N-1} (i k) * t**(i-N) = 0

    """
    if N > 1000:
        raise ValueError('N bigger than 1000 causes numerical issues. '
                         'Please use the numerically stable version of this function.')

    n_over_k_minus_one = dc(scipy.special.comb(N, k, exact=True)) ** -1
    n_over_k_minus_one = float(n_over_k_minus_one)

    i_over_k = np.zeros((N-k, ))
    for idx in range(k, N):
        i_over_k[idx-k] = float(dc.ln( dc(scipy.special.comb(idx, k, exact=True)) ))
    i_minus_n = np.arange(start=k, stop=N) - N

    # when to stop the bisection method
    bisection_precision = 1e-6

    t1 = 0.
    t2 = 1.

    while t2 - t1 > bisection_precision:
        t = (t1 + t2) / 2
        # print(f'Bisection precision 1 : {t2 - t1}')
        polyt = 1. - n_over_k_minus_one * (beta / N) * np.sum(np.exp(i_over_k + i_minus_n * np.log(t)))
        if polyt > 0:
            t2 = t
        else:
            t1 = t

        eps = 1. - t1

    return eps


def eps_general(k, N, beta, fast=False, verbose=False):

    """
    adapted from equation (3)
    (n k) * t**(N-k) - beta/N * sum_{i=k}^{N-1} (i k) * t**(i-k) = 0 ,
    which is

    1 - beta/N * (N k)**(-1) * sum_{i=k}^{N-1} (i k) * t**(i-N) = 0

    """
    start = timeit.default_timer()
    n_over_k_minus_one = dc(scipy.special.comb(N, k, exact=True)) ** -1
    if verbose:
        print(f'Compute n_over_k: {timeit.default_timer()-start}')

    # start = timeit.default_timer()
    # i_over_k = np.empty((N-k, ), dtype=np.object)
    # for idx in range(k, N):
    #     i_over_k[idx-k] = dc.ln(dc(scipy.special.comb(idx, k, exact=True)))
    # if verbose:
    #     print(f'Compute i_over_k: {timeit.default_timer() - start}')

    # smarter way
    # we need to compute an array with entries ( i k ), where i \in [k, N]
    # try to get the current entry from the previous
    # previous entry: (i-1 k) = (i-1)! / (k! (i-1-k)! )
    # current entry: (i k) = i! / (k! (i-k)! ) = i/(i-k)  *  (i-1)! / (k! (i-1-k)! )
    # i.e. array[i] = i/(i-k)  *  array[i-1]
    # finally, we take the log of the elements
    # i.e. array[i] = log( (i/(i-k))  *  ( (i-1)! / (k! (i-1-k)! ) ) = log(i/(i-k)) + array[i-1]
    # as in array[i-1] there is already the log
    start = timeit.default_timer()
    i_over_k_dp = np.empty((N - k,), dtype=np.object)
    prec_comb = dc(1.)
    i_over_k_dp[0] = dc.ln(prec_comb)
    for idx in range(k+1, N):
        i_over_k_dp[idx - k] = i_over_k_dp[idx-k-1] + dc.ln( dc(idx / (idx-k)) )
    if verbose:
        print(f'Compute i_over_k dynamic programming: {timeit.default_timer() - start}')

    # debug test: find the absolute difference between these two methods
    # need to un-comment the other computation of i_over_k
    # print(f'Max difference: {max([abs(i_over_k[i] - i_over_k_dp[i]) for i in range(len(i_over_k))])}')

    i_minus_n = np.arange(start=k, stop=N) - N

    # when to stop the bisection method
    if fast:
        bisection_precision = 1e-4
        # epsilon cannot be smaller than k/N
        # and will probably stay within k/N + 0.1
        # since t = 1-eps, then the bounds for the search are 1-k/N, 1-(k/N+0.1)
        t1 = dc(max(0., 0.9 - k / N))
        t2 = dc(min(1., 1. - k / N))
    else:
        bisection_precision = 1e-6
        t1 = dc(0.)
        t2 = dc(1.)

    while t2 - t1 > bisection_precision:
        t = (t1 + t2) / 2
        if verbose:
            print(f'Bisection precision 1 : {t2 - t1}')
        tmp = i_over_k_dp + i_minus_n * dc.ln(t)
        sumexptmp = sum([dc.exp(i) for i in tmp])
        polyt = dc(1.) - n_over_k_minus_one * dc(beta / N) * sumexptmp
        # polyt = 1. - n_over_k_minus_one * dc((beta / N) * np.sum(dc.exp(i_over_k + i_minus_n * dc.ln(t))))

        if polyt > 0:
            t2 = t
        else:
            t1 = t

        eps = dc(1.) - t1

    return float(eps)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tqdm

    N = 100000
    beta = 1e-3
    epsis = []
    numerical_stable_epsis = []
    # for i in tqdm.tqdm(range(0, N, N//20)):
        # epsis.append(eps_general_smallN(k=i, N=N, beta=beta))
    numerical_stable_epsis.append(eps_general(k=50000, N=N, beta=beta))
    print(numerical_stable_epsis)
    # plt.plot(epsis)
    plt.plot(numerical_stable_epsis)
    plt.grid()
    plt.show()
