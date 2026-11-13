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


    try:
       integral, _ = quad(integrand, 1e-4, 1)
    except Exception as e:
       print(f"Integration failed for: a_u={a_u}, b_u={b_u}, c1={c1}, c2={c2} => {e}")
       return np.nan
   
    if not np.isfinite(integral) or integral <= 0:
       print(f"Bad integral: {integral} for a_u={a_u}, b_u={b_u}, c1={c1}, c2={c2}")
       return np.nan

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
    if x2 > 1 or x2 < 1e-4:
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
    


def chi_sqrd_dy(p, x1vals, M_fixed, dyphoton_data, dyz_data, dyphoton_error, dyz_error ,true_params=TRUE_PARAMS, s=CROSS_SECTION_STEP):

    dyzfit = np.array([dsigma_Z(x1, M_fixed, *p, s=s) for x1 in x1vals])
    dyphotonfit = np.array([dsigma_photon(x1, M_fixed, *p, s=s) for x1 in x1vals])
    chi2z = np.sum(((dyz_data - dyzfit) / dyz_error)**2)
    chi2photon = np.sum(((dyphoton_data - dyphotonfit) / dyphoton_error)**2)
    
    return chi2z + chi2photon



def fit_model(x1vals, M_fixed, dyphoton_data, dyz_data, dyphoton_error, dyz_error):
    """Perform χ² minimization."""
    chi2 = partial(
        chi_sqrd_dy,
        x1vals=x1vals,
        M_fixed=M_fixed,
        dyphoton_data=dyphoton_data,
        dyz_data=dyz_data,
        dyphoton_error=dyphoton_error,
        dyz_error=dyz_error
    )

    result = minimize(chi2, INIT_PARAMS, method='L-BFGS-B', bounds=BOUNDS)

    if not result.success:
        raise RuntimeError("Fit failed")

    return result.x, chi2


# monte carlo bootstrap


def bootstrap_fit(xvals, M_fixed, dyph_rep, dyz_rep, dyphoton_error, dyz_error, init, bounds): 
    chi2 = partial(
        chi_sqrd_dy,
        x1vals=xvals,
        M_fixed=M_fixed,
        dyphoton_data=dyph_rep,
        dyz_data=dyz_rep,
        dyphoton_error=dyphoton_error,
        dyz_error=dyz_error
        )

    result = minimize(chi2, init, method='L-BFGS-B', bounds=bounds)
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

    
    xvals = np.linspace(0.08, 0.95, 50)

    # Generate data

    M_fixed = 10

    dyphoton_data, dyz_data, dyphoton_error,  dyz_error = generate_data(xvals, M_fixed, TRUE_PARAMS)
    # Fit
    p_best, chi2 = fit_model(xvals, M_fixed, dyphoton_data, dyz_data, dyphoton_error, dyz_error)

    # Bootstrap
    n_samples = 250
    pbootstrap = []

    for _ in tqdm(range(n_samples), desc="Bootstrapping"):
        
        dyphoton_fit = dyphoton_data + np.random.normal(0, dyphoton_error)
        dyz_fit = dyz_data + np.random.normal(0, dyz_error)
        
        fit = bootstrap_fit(xvals, M_fixed, dyphoton_fit, dyz_fit, dyphoton_error, dyz_error, INIT_PARAMS ,BOUNDS)
        if fit is not None:
            pbootstrap.append(fit)

    pbootstrap = np.array(pbootstrap)
    mean_params = np.mean(pbootstrap, axis=0)


    #pguess = minimize(chi_fixed, init, method='L-BFGS-B', bounds=bounds)
    
    chi2_values = np.array([chi_sqrd_dy(p, xvals, M_fixed, dyphoton_data, dyz_data , dyphoton_error, dyz_error) for p in pbootstrap])

    cutoff = np.percentile(chi2_values, 35)
    filtered = pbootstrap[chi2_values < cutoff]

    avgs = np.mean(filtered, axis=0)
    stdvs = np.std(filtered, axis=0)
 
    filtered = np.array(filtered)

    labels = ['a_u', 'b_u', 'c1', 'c2', 'a_d', 'b_d', 'd1', 'd2']



    for label, avg, stdv in zip(labels, avgs, stdvs):
        print(f"{label} = {avg:.4f} \u00B1 {stdv:.3f}")
    
    # Fit curve
    dy_fit = np.array([dsigma_photon(x1, M_fixed, *p_best) + dsigma_Z(x1, M_fixed, *p_best) for x1 in xvals])
    dy_data = dyphoton_data + dyz_data
    dy_err = np.sqrt(dyphoton_error**2 + dyz_error**2)

    plot_fit(xvals, dy_data, dy_err, dy_fit)

    # plot PDF uncertainties
    u = u_valence(xvals, *avgs[:4])
    d = d_valence(xvals, *avgs[4:])
    

    uv_mean, uv_std, dv_mean, dv_std = pdf_uncertainty(xvals, filtered)

    plot_pdf(xvals, uv_mean, uv_std, dv_mean, dv_std)

    plot_rel_fractional(xvals, u, uv_std, d, dv_std)
    

if __name__ == "__main__":
    main()