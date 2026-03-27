import math

# Task 1 — Bernoulli MLE

def bernoulli_log_likelihood(data, theta):
    if not data:
        raise ValueError("Data must not be empty.")

    if not (0 < theta < 1):
        raise ValueError("Theta must be between 0 and 1 (exclusive).")

    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")

    log_likelihood = 0.0
    for x in data:
        log_likelihood += x * math.log(theta) + (1 - x) * math.log(1 - theta)

    return log_likelihood
def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    if not data:
        raise ValueError("Data must not be empty.")

    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")

    n = len(data)
    num_successes = sum(data)
    num_failures = n - num_successes

    mle = num_successes / n

    if candidate_thetas is None:
        candidate_thetas = [0.1, 0.25, 0.5, 0.75, 0.9]

    log_likelihoods = {}

    for theta in candidate_thetas:
        if 0 < theta < 1:
            log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)

    best_candidate = max(log_likelihoods, key=log_likelihoods.get)

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }

# Task 2 — Poisson MLE

def poisson_log_likelihood(data, lam):
    if not data:
        raise ValueError("Data must not be empty.")

    if lam <= 0:
        raise ValueError("Lambda must be positive.")

    for x in data:
        if not (isinstance(x, int) and x >= 0):
            raise ValueError("Data must contain non-negative integers only.")

    log_likelihood = 0.0

    for x in data:
        log_likelihood += x * math.log(lam) - lam - math.lgamma(x + 1)

    return log_likelihood

def poisson_mle_analysis(data, candidate_lambdas=None):
    if not data:
        raise ValueError("Data must not be empty.")
    for x in data:
        if not (isinstance(x, int) and x >= 0):
            raise ValueError("Data must contain non-negative integers only.")

    n = len(data)
    mle = sum(data) / n

    if candidate_lambdas is None:
        candidate_lambdas = [0.5, 1, 2, 3, 5]

    log_likelihoods = {}
    for lam in candidate_lambdas:
        if lam > 0:
            log_likelihoods[lam] = poisson_log_likelihood(data, lam)

    best_candidate = max(log_likelihoods, key=log_likelihoods.get)

    return {
        "mle": mle,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }
