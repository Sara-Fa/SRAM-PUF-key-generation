import numpy as np
from scipy.stats import norm, binom
from scipy.integrate import quad


def f_Q(x: np.ndarray | float, lambda_val: float) -> np.ndarray | float:
    """
    Density of per-cell reliability x in the quasi-global stability model.

    f_Q(x; λ) = λ * φ(λ * Φ^{-1}(x)) / φ(Φ^{-1}(x))
    """
    x_arr = np.clip(np.asarray(x), 1e-12, 1 - 1e-12)
    z = norm.ppf(x_arr)
    phi_z = norm.pdf(z)
    phi_lambda_z = norm.pdf(lambda_val * z)
    result = lambda_val * phi_lambda_z / phi_z
    if np.isscalar(x):
        return float(result)
    return result


def delta_from_K_eta(K: int, eta: float) -> float:
    """
    Δ ≤ 1 / (2 * sqrt(1 + K/(Φ⁻¹(η))²))
    We use equality to pick the largest Δ satisfying the bound.
    """
    phi_inv_eta = norm.ppf(np.float64(eta))
    if phi_inv_eta == 0:
        raise ValueError("Phi^{-1}(eta) cannot be 0; avoid eta=0.5.")
    return float(np.float64(1.0) / (np.float64(2.0) * np.sqrt(np.float64(1.0) + K / (phi_inv_eta ** 2))))


def p_unreliable_exact(delta: float, lambda_val: float) -> float:
    """
    P[UNREL] = ∫_{1/2-Δ}^{1/2+Δ} f_Q(x; λ) dx
    """
    lower = max(0.5 - delta, 1e-12)
    upper = min(0.5 + delta, 1 - 1e-12)
    val, _ = quad(lambda t: f_Q(t, lambda_val), lower, upper, limit=200)
    return float(val)


def p_unreliable_approx(delta: float, lambda_val: float) -> float:
    """
    Small-Δ approximation: P[UNREL] ≈ 2Δ * f_Q(1/2; λ) = 2Δ * λ
    """
    return 2.0 * delta * lambda_val


def eta_from_K_delta(K: int, delta: float) -> float:
    """
    Invert Δ(K, η) = 1 / (2 * sqrt(1 + K / (Φ⁻¹(η))²)) to get η as a function of K and Δ.
    Use np.float64 throughout to preserve precision near 1.
    Φ⁻¹(η) = sqrt( K / (1/(4Δ²) − 1) ) → η = Φ( sqrt( K / (1/(4Δ²) − 1) ) )
    """
    K64 = np.float64(K)
    delta64 = np.float64(delta)
    denom64 = (np.float64(1.0) / (np.float64(4.0) * (delta64 ** np.float64(2.0)))) - np.float64(1.0)
    if denom64 <= np.float64(0.0):
        return float(norm.cdf(0.0))
    phi_inv64 = np.sqrt(K64 / denom64, dtype=np.float64)
    # Use survival function for better tail precision: eta = 1 - sf(z)
    sf_val = np.float64(norm.sf(float(phi_inv64)))
    eta64 = np.float64(1.0) - sf_val
    return float(eta64)


def eta_expr_from_K_delta(K: int, delta: float) -> str:
    """
    Return a precise textual expression for eta as '1-10^-k' even when 1-eta underflows.
    Uses scipy.stats.norm.logsf(z) for numerically stable tail computation.
    """
    K64 = np.float64(K)
    delta64 = np.float64(delta)
    denom64 = (np.float64(1.0) / (np.float64(4.0) * (delta64 ** np.float64(2.0)))) - np.float64(1.0)
    if denom64 <= np.float64(0.0):
        return "1-10^-0"
    z = np.sqrt(K64 / denom64, dtype=np.float64)
    # k = -log10(1-eta) = -log10(Q(z)) = -log10e * log(Q(z))
    log10e = np.float64(1.0) / np.log(np.float64(10.0))
    log10Q = np.float64(norm.logsf(float(z))) * log10e
    k = -log10Q
    k_int = int(np.round(float(k))) if np.isfinite(k) and k > 0 else 0
    return f"1-10^-{k_int}"


def delta_from_K_maxvar_eta(K: int, max_variance: float, eta: float) -> float:
    """
    Compute delta from K, max_variance, and eta based on equation 23.
    
    Equation 23: K ≥ max_var² / (delta² (1-eta))
    Solving for delta: delta ≥ sqrt(max_var² / (K * (1-eta)))
    
    This is an ADDITIONAL formula that provides an alternative way to compute delta
    by incorporating the maximum variance among reliabilities. It does NOT replace
    the existing delta_from_K_eta formula.
    
    Args:
        K: Number of enrollment reads
        max_variance: Maximum variance among all reliabilities
        eta: Confidence parameter (0 < eta < 1)
    
    Returns:
        Delta value computed from the variance-based formula (equation 23)
    """
    # Ensure inputs are valid
    if K <= 0:
        raise ValueError("K must be positive")
    if max_variance < 0:
        raise ValueError("max_variance must be non-negative")
    if not (0 < eta < 1):
        raise ValueError("eta must be between 0 and 1")
    
    # Convert to float64 for precision
    K64 = np.float64(K)
    max_var64 = np.float64(max_variance)
    eta64 = np.float64(eta)
    
    # Equation 23: K ≥ max_var² / (delta² (1-eta))
    # Solving for delta: delta ≥ sqrt(max_var² / (K * (1-eta)))
    # We use equality to get the minimum delta satisfying the bound
    denominator = K64 * (np.float64(1.0) - eta64)
    
    if denominator <= 0:
        raise ValueError("K * (1-eta) must be positive")
    
    delta = np.sqrt((max_var64 ** 2) / denominator)
    return float(delta)


def probability_discard(L: int, G: int, p_unrel: float) -> float:
    """
    P[Binom(L+G, p_unrel) > G] = 1 - BinomCDF(G; n=L+G, p=p_unrel)
    """
    n = int(L) + int(G)
    # Ensure valid bounds
    if n <= 0:
        return float(1.0)
    # 1 - P[X <= G]
    return float(1.0 - binom.cdf(int(G), n, float(p_unrel)))


def find_minimum_surplus(L: int, p_unrel: float, P_disc: float, G_max: int = 1000) -> tuple[int | None, float | None]:
    """
    Find the smallest G such that P[Binom(L+G, p_unrel) > G] <= P_disc.
    Returns (G, actual_discard_probability). If not found up to G_max, returns (None, None).
    """
    for G in range(0, int(G_max) + 1):
        prob = probability_discard(int(L), int(G), float(p_unrel))
        if prob <= float(P_disc):
            return int(G), float(prob)
    return None, None

