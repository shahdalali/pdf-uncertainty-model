import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm
import corner

# constants
TRUE_PARAMS = [0.8, 3.0, 2.0, 4.0, 0.8, 4.0, 2.0, 4.0]
INIT_PARAMS = [0.6, 3.2, 2.2, 4.2, 0.6, 4.1, 2.2, 4.2]

BOUNDS = [
    (0.1, 2.0), (1.0, 5.0), (0.1, 10.0), (0.1, 20.0),
    (0.1, 2.0), (2.0, 6.0), (0.1, 10.0), (0.1, 30.0)
]

NOISE_LEVEL = 0.02
HESSIAN_STEP = 1e-2


# physics model

def normalisation(a_u, b_u, c1, c2, targ=2):
    """Compute normalization constant A enforcing ∫q_v dx = target."""
    def integrand(x): 
        return x**a_u * (1 - x)**b_u * (1 + c1*np.sqrt(x) + c2*x)

    integral, _ = quad(integrand, 1e-4, 1)

    if not np.isfinite(integral) or integral == 0:
        raise ValueError(f"Invalid integral for params: {a_u}, {b_u}, {c1}, {c2}")

    return targ / integral


def u_valence(x, a_u, b_u, c1, c2):
    """Valence u-quark PDF (∫u_v dx = 2)"""
    A_u = normalisation(a_u, b_u, c1, c2, targ=2)
    return A_u * x**a_u * (1 - x)**b_u * (1 + c1*np.sqrt(x) + c2*x)

def d_valence(x, a_d, b_d, d1, d2):
    """Valence d-quark PDF (∫d_v dx = 1)"""
    A_d = normalisation(a_d, b_d, d1, d2, targ=1)
    return A_d * x**a_d * (1 - x)**b_d * (1 + d1*np.sqrt(x) + d2*x)

def F2p(x, *p):
    """Proton structure function."""
    u = u_valence(x, *p[:4])
    d = d_valence(x, *p[4:])
    return x * ((4/9) * u + (1/9) * d)

def F2n(x, *p):
    """Neutron structure function."""
    u = u_valence(x, *p[:4])
    d = d_valence(x, *p[4:])
    return x * ((1/9) * u + (4/9) * d)


# data generation


def generate_data(xvals, params, noise=NOISE_LEVEL):
    """Generate synthetic F2p and F2n data with multiplicative noise."""
    F2p_true = F2p(xvals, *params)
    F2n_true = F2n(xvals, *params)

    F2p_data = F2p_true * np.random.normal(1, noise, size=xvals.shape)
    F2n_data = F2n_true * np.random.normal(1, noise, size=xvals.shape)

    F2p_err = noise * F2p_data
    F2n_err = noise * F2n_data

    return F2p_data, F2n_data, F2p_err, F2n_err



# fit

def chi_squared(p, x, F2p_data, F2n_data, err_p, err_n):
    """Total χ² for proton and neutron."""
    F2p_fit = F2p(x, *p)
    F2n_fit = F2n(x, *p)

    chi_p = np.sum(((F2p_fit - F2p_data) / err_p) ** 2)
    chi_n = np.sum(((F2n_fit - F2n_data) / err_n) ** 2)

    return chi_p + chi_n


def fit_model(xvals, F2p_data, F2n_data, F2p_err, F2n_err):
    """Perform χ² minimization."""
    chi2 = partial(
        chi_squared,
        x=xvals,
        F2p_data=F2p_data,
        F2n_data=F2n_data,
        err_p=F2p_err,
        err_n=F2n_err
    )

    result = minimize(chi2, INIT_PARAMS, method='L-BFGS-B', bounds=BOUNDS)

    if not result.success:
        raise RuntimeError("Fit failed")

    return result.x, chi2



# hessian + covariance

def hessian(chi2, p, step=HESSIAN_STEP):
    """Numerical Hessian via finite differences"""
    n = len(p)
    H = np.zeros((n, n))

    for i in tqdm(range(n), desc="Hessian fitting progress"):
        for j in range(n):
            
            dp_i = np.zeros(n)
            dp_j = np.zeros(n)
            dp_i[i] = step
            dp_j[j] = step

            pp = chi2(p + dp_i + dp_j)
            pm = chi2(p + dp_i - dp_j)
            mp = chi2(p - dp_i + dp_j)
            mm = chi2(p - dp_i - dp_j)

            H[i, j] = (pp - pm - mp + mm) / (4 * step**2)

    return 0.5 * (H + H.T)


# PDF uncertainty (Hessian)

def pdf_uncertainty(xgrid, p, cov):
    """Compute Hessian PDF uncertainties."""
    eigvals, eigvecs = np.linalg.eigh(cov)

    uv_err = np.zeros_like(xgrid)
    dv_err = np.zeros_like(xgrid)

    for i in range(len(p)):
        shift = eigvecs[:, i] * np.sqrt(max(eigvals[i], 0))

        plus = p + shift
        minus = p - shift

        uv_err += 0.25 * (u_valence(xgrid, *plus[:4]) -
                          u_valence(xgrid, *minus[:4]))**2

        dv_err += 0.25 * (d_valence(xgrid, *plus[4:]) -
                          d_valence(xgrid, *minus[4:]))**2

    return np.sqrt(uv_err), np.sqrt(dv_err)


# plotting


def plot_fit(x, data, err, fit):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.errorbar(x, data, yerr=err, fmt='o', markersize=3, label="Data")
    plt.plot(x, fit, 'r-', label="Fit", linewidth=0.8)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("F2", fontsize=14)
    plt.legend(fontsize=12)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pdf(xgrid, u_mean, uv_err, d_mean, dv_err):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(xgrid, u_mean, label=r"$u_v(x)$", color='blue', linewidth=0.8)
    plt.fill_between(xgrid, u_mean - uv_err, u_mean + uv_err, color='blue', alpha=0.3)
    plt.plot(xgrid, d_mean, label=r"$d_v(x)$", color='green', linewidth=0.8)
    plt.fill_between(xgrid, d_mean - dv_err, d_mean + dv_err, color='green', alpha=0.3)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel("PDF", fontsize=14)
    plt.title("Valence PDFs with Hessian Uncertainty Bands")
    plt.legend(fontsize=12)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_rel_fractional(xgrid, u_mean, uv_err, d_mean, dv_err):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.xscale('log')
    plt.plot(xgrid, u_mean/u_mean, label=r"$u_v(x)$", color='blue', linewidth=0.8)
    plt.fill_between(xgrid, (u_mean - uv_err)/u_mean, (u_mean + uv_err)/u_mean, color='blue', alpha=0.3)
    plt.plot(xgrid, d_mean/d_mean, label=r"$d_v(x)$", color='green', linewidth=0.8)
    plt.fill_between(xgrid, (d_mean - dv_err)/d_mean, (d_mean + dv_err)/d_mean, color='green', alpha=0.3)
    plt.xlabel(r"$log(x)$", fontsize=14)
    plt.ylabel("Fractional uncertainty", fontsize=14)
    plt.title("Hessian Relative Fractional Uncertainty in Valence PDFs")
    plt.legend(fontsize=12)
    #plt.grid(True)

    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    

# main

np.random.seed(42)

def main():
    xvals = np.linspace(0.01, 0.99, 50)

    # Generate data
    F2p_data, F2n_data, F2p_err, F2n_err = generate_data(xvals, TRUE_PARAMS)

    # Fit
    p_best, chi2 = fit_model(xvals, F2p_data, F2n_data, F2p_err, F2n_err)

    # Hessian + covariance
    H = hessian(chi2, p_best)
    cov = np.linalg.pinv(H)

    # Fit curve
    F2_fit = F2p(xvals, *p_best) + F2n(xvals, *p_best)
    F2_data = F2p_data + F2n_data
    F2_err = np.sqrt(F2p_err**2 + F2n_err**2)

    plot_fit(xvals, F2_data, F2_err, F2_fit)

    # PDF uncertainties
    xgrid = np.linspace(0.01, 0.99, 100)

    u = u_valence(xgrid, *p_best[:4])
    d = d_valence(xgrid, *p_best[4:])

    uv_err, dv_err = pdf_uncertainty(xgrid, p_best, cov)

    plot_pdf(xgrid, u, uv_err, d, dv_err)

    plot_rel_fractional(xgrid, u, uv_err, d, dv_err)
    
    plt.show()


if __name__ == "__main__":
    main()