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
CROSS_SECTION_STEP = 1e4       # GeV^2, root(s) = 100 GeV



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

def dsigma_photon(x1, M=10 , *pnew, s=CROSS_SECTION_STEP):
    alpha = 1/137
    x2 = M**2 / (x1 * s)
    if x2 > 1 or x2 < 1e-4:
        return 0.0
    
    # PDFs
    u = u_valence(x1, *pnew[:4])
    ubar = u_valence(x2, *pnew[:4])
    d = d_valence(x1, *pnew[4:])
    dbar = d_valence(x2, *pnew[4:])

    pdf_sum = (4/9) * (u * ubar) + (1/9) * (d * dbar)
    
    prefactor = (4 * np.pi * alpha**2) / (9 * M**2 * s)
    return prefactor * pdf_sum


def prop_factor(M, Mv, Gv):
    return M**4 / ((M**2 - Mv**2)**2 + (Mv * Gv)**2)


def dsigma_Z(x1, M=10, *pnew, s=CROSS_SECTION_STEP):
    
    alpha = 1 / 137
    x2 = M**2 / (x1 * s)
    if np.any((x2 > 1) | (x2 < 1e-4)):
        return 0.0

    # PDFs
    u = u_valence(x1, *pnew[:4])
    ubar = u_valence(x2, *pnew[:4])
    d = d_valence(x1, *pnew[4:])
    dbar = d_valence(x2, *pnew[4:])

    # Z couplings
    gZu = 0.35
    gZd = -0.44

    pdf_Z = (gZu**2) * (u * ubar) + (gZd**2) * (d * dbar)
    prefactor = (4 * np.pi * alpha**2) / (9 * M**2 * s)
    Zprop = prop_factor(M, 91.189, 2.50)

    return prefactor * pdf_Z * Zprop
   

# data generation

def generate_data(x1vals, M_fixed, true_params=TRUE_PARAMS, s=CROSS_SECTION_STEP):

    dyz_true = np.array([dsigma_Z(x1, M_fixed, *true_params, s=s) for x1 in x1vals])
    dyphoton_true = np.array([dsigma_photon(x1, M_fixed, *true_params, s=s) for x1 in x1vals])
    
    min_val_ph = np.max(dyz_true) * 1e-3  # say 0.1% of max value
    min_val_z = np.max(dyphoton_true) * 1e-3  # say 0.1% of max value
    
    dyz_error = 0.02 * np.maximum(dyz_true, min_val_z)
    dyphoton_error = 0.02 * np.maximum(dyphoton_true, min_val_ph)
    
    znoise = np.random.normal(1, 0.02, size=dyz_true.shape)
    photonnoise = np.random.normal(1, 0.02, size=dyphoton_true.shape)
    
    dyz_data = dyz_true * znoise # absolute noise
    dyphoton_data = dyphoton_true * photonnoise
    
    return dyz_data, dyphoton_data, dyz_error, dyphoton_error
    


def chi_sqrd_dy(p, x1vals, M_fixed, dyz_data, dyphoton_data, dyz_error, dyphoton_error ,true_params=TRUE_PARAMS, s=CROSS_SECTION_STEP):

    dyzfit = np.array([dsigma_Z(x1, M_fixed, *p, s=s) for x1 in x1vals])
    dyphotonfit = np.array([dsigma_photon(x1, M_fixed, *p, s=s) for x1 in x1vals])
    chi2z = np.sum(((dyz_data - dyzfit) / dyz_error)**2)
    chi2photon = np.sum(((dyphoton_data - dyphotonfit) / dyphoton_error)**2)
    
    return chi2z + chi2photon



def fit_model(x1vals, M_fixed, dyz_data, dyphoton_data, dyz_error, dyphoton_error):
    """Perform χ² minimization."""
    chi2 = partial(
        chi_sqrd_dy,
        x1vals=x1vals,
        M_fixed=M_fixed,
        dyz_data=dyz_data,
        dyphoton_data=dyphoton_data,
        dyz_error=dyz_error,
        dyphoton_error=dyphoton_error
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
    plt.ylabel("Cross-Section", fontsize=14)
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
    xvals = np.linspace(0.08, 0.95, 50)

    # Generate data

    M_fixed = 10

    dyz_data, dyphoton_data, dyz_error, dyphoton_error = generate_data(xvals, M_fixed, TRUE_PARAMS)
    # Fit
    p_best, chi2 = fit_model(xvals, M_fixed, dyz_data, dyphoton_data, dyz_error, dyphoton_error)

    # Hessian + covariance
    H = hessian(chi2, p_best)
    cov = np.linalg.pinv(H)

    # Fit curve
    dy_fit = np.array([dsigma_photon(x1, M_fixed, *p_best) + dsigma_Z(x1, M_fixed, *p_best) for x1 in xvals])
    dy_data = dyphoton_data + dyz_data
    dy_err = np.sqrt(dyphoton_error**2 + dyz_error**2)

    plot_fit(xvals, dy_data, dy_err, dy_fit)

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