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

def F2total(x, params):
    return F2p(x, *params) + F2n(x, *params)

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


def bootstrap_fit(xvals, F2p_rep, F2n_rep, F2p_err, F2n_err, init, bounds):

    def chi(p):
        F2p_fit = F2p(xvals, *p)
        F2n_fit = F2n(xvals, *p)

        chi_p = np.sum(((F2p_fit - F2p_rep) / F2p_err) ** 2)
        chi_n = np.sum(((F2n_fit - F2n_rep) / F2n_err) ** 2)

        return chi_p + chi_n

    result = minimize(chi, init, method='L-BFGS-B', bounds=bounds)
    return result.x if result.success else None



def pdf_uncertainty(xgrid, param_samples):
    """Compute MC PDF uncertainties."""
    
    uvs = []
    dvs = []

    for params in param_samples:
        uvs.append(u_valence(xgrid, *params[:4]))
        dvs.append(d_valence(xgrid, *params[4:]))

    uvs = np.array(uvs)
    dvs = np.array(dvs)

    u_mean = np.mean(uvs, axis=0)
    u_std = np.std(uvs, axis=0)
    

    d_mean = np.mean(dvs, axis=0)
    d_std = np.std(dvs, axis=0)
    
    return u_mean, u_std, d_mean, d_std

def plot_pdf(xvals, uv_mean, uv_std, dv_mean, dv_std):
    plt.plot(xvals, uv_mean, label=r"$u_v(x)$", color='blue', linewidth=0.8)
    plt.fill_between(xvals, uv_mean - uv_std, uv_mean + uv_std, color='blue', alpha=0.3)

    plt.plot(xvals, dv_mean, label=r"$d_v(x)$", color='green', linewidth=0.8)
    plt.fill_between(xvals, dv_mean - dv_std, dv_mean + dv_std, color='green', alpha=0.3)
    plt.show()
    
    
def plot_rel_fractional(xgrid, u_mean, uv_std, d_mean, dv_std):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.xscale('log')
    plt.plot(xgrid, u_mean/u_mean, label=r"$u_v(x)$", color='blue', linewidth=0.8)
    plt.fill_between(xgrid, (u_mean - uv_std)/u_mean, (u_mean + uv_std)/u_mean, color='blue', alpha=0.3)
    plt.plot(xgrid, d_mean/d_mean, label=r"$d_v(x)$", color='green', linewidth=0.8)
    plt.fill_between(xgrid, (d_mean - dv_std)/d_mean, (d_mean + dv_std)/d_mean, color='green', alpha=0.3)
    plt.xlabel(r"$log(x)$", fontsize=14)
    plt.ylabel("Fractional uncertainty", fontsize=14)
    plt.title("Monte Carlo Relative Fractional Uncertainty in Valence PDFs")
    plt.legend(fontsize=12)
    #plt.grid(True)

    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


# main execution

np.random.seed(42)

def main():
    xvals = np.linspace(0.01, 0.99, 100)
    
    true_vals = [0.8, 3.0, 2.0, 4.0, 0.8, 4.0, 2.0, 4.0] 
    true_params = [0.8, 3, 2, 4, 0.8, 4, 2, 4]
    initial = true_params.copy()

    bounds = [(0.1, 2.0), (1.0, 5.0), (0.1, 5.0), (0.1, 10.0),
              (0.1, 2.0), (2.0, 6.0), (0.1, 5.0), (0.1, 10.0)]

    F2p_data, F2n_data, F2p_err, F2n_err = generate_data(xvals, true_params)

    F2total_data = F2p_data + F2n_data
    F2total_err = np.sqrt(F2p_err**2 + F2n_err**2)

    # Bootstrap
    n_samples = 200
    pbootstrap = []

    for _ in tqdm(range(n_samples), desc="Bootstrapping"):
        F2p_rep = F2p_data + np.random.normal(0, F2p_err)
        F2n_rep = F2n_data + np.random.normal(0, F2n_err)
        fit = bootstrap_fit(xvals, F2p_rep, F2n_rep, F2p_err, F2n_err, initial, bounds)
        if fit is not None:
            pbootstrap.append(fit)

    pbootstrap = np.array(pbootstrap)
    mean_params = np.mean(pbootstrap, axis=0)


    #pguess = minimize(chi_fixed, init, method='L-BFGS-B', bounds=bounds)
    
    chi2_values = np.array([chi_squared(p, xvals, F2p_data, F2n_data, F2p_err, F2n_err) for p in pbootstrap])

    cutoff = np.percentile(chi2_values, 80)
    filtered = pbootstrap[chi2_values < cutoff]

    avgs = np.mean(filtered, axis=0)
    stdvs = np.std(filtered, axis=0)
 
    filtered = np.array(filtered)

    labels = ['a_u', 'b_u', 'c1', 'c2', 'a_d', 'b_d', 'd1', 'd2']



    for label, avg, stdv in zip(labels, avgs, stdvs):
        print(f"{label} = {avg:.4f} \u00B1 {stdv:.3f}")
    
    bestfit = F2total(xvals, avgs)

    u = u_valence(xvals, *avgs[:4])
    d = d_valence(xvals, *avgs[4:])
    

    uv_mean, uv_std, dv_mean, dv_std = pdf_uncertainty(xvals, filtered)

    plot_pdf(xvals, uv_mean, uv_std, dv_mean, dv_std)

    plot_rel_fractional(xvals, u, uv_std, d, dv_std)
    

if __name__ == "__main__":
    main()