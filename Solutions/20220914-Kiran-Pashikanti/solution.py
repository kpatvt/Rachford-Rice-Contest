from typing import Tuple
import mpmath
import numpy as np
from check_convergence import EPS_T
import decimal

MAX_ITR = 100


decimal.getcontext().prec = 25


def solve_rr_2phase_decimal(k, z, nc, eps, max_iterations, initial_beta=-1.0):
    zero = decimal.Decimal(0)
    one = decimal.Decimal(1)
    eps = decimal.Decimal(eps)
    k = [decimal.Decimal(i) for i in k]
    z = [decimal.Decimal(i) for i in z]
    k_minus_1 = [k[i] - decimal.Decimal(1) for i in range(nc)]

    k_min = min(k)
    k_max = max(k)

    beta_min = one / (one - k_max)
    beta_max = one / (one - k_min)

    # Guess halfway in the interval
    beta_guess = (beta_min + beta_max)/2

    def rr(beta_trial):
        result = decimal.Decimal(0)
        for i in range(nc):
            result += z[i]*(k_minus_1[i] / (one + beta_trial * k_minus_1[i]))
        return result

    def drr(beta_trial):
        result = decimal.Decimal(0)
        for i in range(nc):
            result += -z[i]*((k_minus_1[i] / (one + beta_trial * k_minus_1[i])) ** 2)
        return result

    def rr_alt(beta_trial):
        result = decimal.Decimal(0)
        for i in range(nc):
            result += z[i]*(k_minus_1[i] / (beta_trial + (one - beta_trial) * k[i]))
        return result

    def drr_alt(beta_trial):
        result = decimal.Decimal(0)
        for i in range(nc):
            result += z[i]*((k_minus_1[i] / (beta_trial + (one - beta_trial) * k[i])) ** 2)

        return result

    converged = False
    iter_count = 0
    use_alt_form = False
    f_at_point_5 = rr(decimal.Decimal(0.5))

    # Use original form when f(0.5) <= 0.0
    if f_at_point_5 > zero:
        use_alt_form = True
        beta_guess = one - beta_guess
        beta_min = one - beta_max
        beta_max = one - beta_min

    while iter_count < max_iterations:
        if not use_alt_form:
            f = rr(beta_guess)
            f_prime = drr(beta_guess)
            if f > zero:
                if beta_guess > beta_min:
                    beta_min = beta_guess
            elif f < zero:
                if beta_guess < beta_max:
                    beta_max = beta_guess
        else:
            f = rr_alt(beta_guess)
            f_prime = drr_alt(beta_guess)
            if f > zero:
                if beta_guess < beta_max:
                    beta_max = beta_guess
            elif f < zero:
                if beta_guess > beta_min:
                    beta_min = beta_guess

        delta = f/f_prime
        beta_guess_new = beta_guess - delta

        if beta_guess_new > beta_max or beta_guess_new < beta_min:
            beta_guess_new = (beta_min + beta_max)/decimal.Decimal(2)

        beta_guess = beta_guess_new

        if not use_alt_form:
            V = beta_guess
        else:
            V = one - beta_guess

        if abs(f) < eps:
            converged = True
            break

        iter_count += 1

    x = [z[i] / (one + V * k_minus_1[i]) for i in range(nc)]
    y = [k[i] * x[i] for i in range(nc)]

    return iter_count, converged, V, np.array(x), np.array(y)


# Set the precision - 25 dps is minimum required to converge all cases
mpmath.mp.dps = 25

def solve_rr_2phase_mpmath(k, z, nc, eps, max_iterations, initial_beta=-1.0):
    eps = mpmath.mpf(eps)
    k = mpmath.matrix(k)
    z = mpmath.matrix(z)
    k_minus_1 = k - 1

    k_min = min(k)
    k_max = max(k)

    beta_min = 1 / (1 - k_max)
    beta_max = 1 / (1 - k_min)

    # Guess halfway in the interval
    beta_guess = (beta_min + beta_max)/2
    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)

    def rr(beta_trial):
        result = mpmath.mpf(0)
        for i in range(nc):
            result += z[i]*(k_minus_1[i] / (one + beta_trial * k_minus_1[i]))
        return result

    def drr(beta_trial):
        result = mpmath.mpf(0)
        for i in range(nc):
            result += -z[i]*((k_minus_1[i] / (one + beta_trial * k_minus_1[i])) ** 2)
        return result

    def rr_alt(beta_trial):
        result = mpmath.mpf(0)
        for i in range(nc):
            result += z[i]*(k_minus_1[i] / (beta_trial + (one - beta_trial) * k[i]))
        return result

    def drr_alt(beta_trial):
        result = mpmath.mpf(0)
        for i in range(nc):
            result += z[i]*((k_minus_1[i] / (beta_trial + (one - beta_trial) * k[i])) ** 2)

        return result

    converged = False
    iter_count = 0
    use_alt_form = False
    f_at_point_5 = rr(mpmath.mpf(0.5))

    # Use original form when f(0.5) <= 0.0
    if f_at_point_5 > zero:
        use_alt_form = True
        beta_guess = one - beta_guess
        beta_min = one - beta_max
        beta_max = one - beta_min

    while iter_count < max_iterations:
        if not use_alt_form:
            f = rr(beta_guess)
            f_prime = drr(beta_guess)
            if f > zero:
                if beta_guess > beta_min:
                    beta_min = beta_guess
            elif f < zero:
                if beta_guess < beta_max:
                    beta_max = beta_guess
        else:
            f = rr_alt(beta_guess)
            f_prime = drr_alt(beta_guess)
            if f > zero:
                if beta_guess < beta_max:
                    beta_max = beta_guess
            elif f < zero:
                if beta_guess > beta_min:
                    beta_min = beta_guess

        delta = f/f_prime
        beta_guess_new = beta_guess - delta

        if beta_guess_new > beta_max or beta_guess_new < beta_min:
            beta_guess_new = (beta_min + beta_max)/2

        beta_guess = beta_guess_new

        if not use_alt_form:
            V = beta_guess
        else:
            V = one - beta_guess

        if abs(f) < eps:
            converged = True
            break

        iter_count += 1

    x = [z[i] / (one + V * k_minus_1[i]) for i in range(nc)]
    y = [k[i] * x[i] for i in range(nc)]

    return iter_count, converged, V, np.array(x), np.array(y)


def rachford_rice_solver(Nc: int, zi: np.ndarray, Ki: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """
    This function solves for the root of the Rachford-Rice equation. The solution
    is defined by the contestent and this is the only part of the code that the
    contestent should change.

    The input is the number of components (Nc), the total composition (zi), and the K-values (KI).

    The output is the number of iterations used (N), the vapor molar composition (yi), the liquid
    molar composition (xi), the vapor molar fraction (V), and the liquid molar fraction (L).
    """
    # ===== Add your code below (remember to remove the dummy variables). =====

    # ===== REMOVE DUMMY VALUES BELOW =====
    # Niter = 1
    # yi = np.array([0])
    # xi = np.array([0])
    # V = 0
    # L = 0

    Niter, converged, V, xi, yi = solve_rr_2phase_decimal(Ki, zi, Nc, EPS_T, MAX_ITR)
    L = 1 - V

    # =====================================
    if Niter >= MAX_ITR:
        print("******************************************************")
        print("*** The maximum number of iterations was exceeded! ***")
        print("******************************************************")

    return Niter, yi, xi, V, L