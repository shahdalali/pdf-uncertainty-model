# pdf-uncertainty-model
Master's thesis code comparing the Hessian matrix method and Monte Carlo method as used in estimating uncertainties in Parton Distribution Functions.
## Overview
This repository contains the simulation and statistical inference pipeline for my Master's thesis in theoretical particle physics. The project extracts **up ($`u_v`$) and down ($`d_v`$) valence quark Parton Distribution Functions (PDFs)** from synthetic collider data and compares two standard uncertainty quantification methods, the **Hessian eigenvector method** and **Monte Carlo (MC) bootstrapping method**.

By testing parameter recovery across two types of physical processes (**Deep Inelastic Scattering (DIS)** and the **Drell-Yan (DY) process**) this codebase evaluates how experimental observables and statistical error-propagation methods impact PDF constraints across momentum fractions x.

---

## Physics and Statistical Methodology

### 1. Valence PDF Parametrisation
Both valence distributions follow a standard phenomenological parametrisation normalised via numerical integration (`scipy.integrate.quad`) to satisfy quark number sum rules $`(\int_0^1 u_v(x)\,dx = 2`$ and $`\int_0^1 d_v(x)\,dx = 1)`$:

$$\[q_v(x) = A_q x^{a_q} (1 - x)^{b_q} \left(1 + c_1 \sqrt{x} + c_2 x\right)\]$$

### 2. Simulated Observables
Synthetic datasets are generated using known parameters with 2% Gaussian multiplicative noise:
* **Deep Inelastic Scattering (DIS):** Simulates proton $`(F_2^p)`$ and neutron $`(F_2^n)`$ structure functions across $`x \in [0.01, 0.99]`$.
* **Drell-Yan (DY) Process:** Simulates differential cross-sections $`(d\sigma/dM)`$ mediated by virtual photon $`(\gamma^*)`$ and Z-boson exchange at $`\sqrt{s} = 100\text{ GeV}`$ and $`M = 10`$ GeV across $`x_1 \in [0.08, 0.95]`$.

### 3. Parameter Fitting and Uncertainty Propagation
Best-fit parameters are obtained via bounded χ² minimisation using `L-BFGS-B`. Uncertainties are propagated to the PDFs using two different approaches:
* **Hessian Method:** Computes the numerical second-derivative matrix of the χ² surface via finite differences, inverts it using the Moore-Penrose pseudoinverse (`np.linalg.pinv`), and propagates diagonalised eigenvector shifts into PDF error bands.
* **Monte Carlo Bootstrap Method:** Generates 200–250 pseudo-data replicas, refits the model to each replica, filters replicas using a χ² percentile cutoff, and computes the empirical mean and standard deviation across the ensemble.

---

## Repository Structure

```text
---pdf-uncertainty-model/
  |--- README.md
	|--- drell-yan/
	|	|--- bayesian/
	|	|	|--- main.py  # Drell-Yan (γ* + Z) χ² fit and Hessian eigenvector uncertainty analysis
	|	|
	|	|--- mc/
	|		|--- main.py  # Drell-Yan (γ* + Z) χ² fit and Monte Carlo replica uncertainty analysis
	|
	|--- structure_functions/
		|--- bayesian/
		|	|--- main.py # DIS (F2p, F2n) χ² fit and Hessian eigenvector uncertainty analysis
		|
		|--- mc/
			|--- main.py # DIS (F2p, F2n) χ² fit and Monte Carlo replica uncertainty analysis
```		
---



## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shahdalali/pdf-uncertainty-model.git
   cd pdf-uncertainty-model
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install numpy scipy matplotlib tqdm corner
   ```

---

## Usage

Each script is fitted with a fixed random seed (`np.random.seed(42)`) for reproducibility. Run any of the four pipelines from the root `pdf-uncertainty-model` directory by specifying its folder path:

```bash
# Run Deep Inelastic Scattering (Structure Functions) analyses
python structure_functions/bayesian/main.py
python structure_functions/mc/main.py

# Run Drell-Yan analyses
python drell-yan/bayesian/main.py
python drell-yan/mc/main.py
```

### Outputs
Running a script outputs parameter recovery estimates (for Monte Carlo runs) and generates multiple diagnostic plots:
1. **Observable Fit:** Synthetic data with error bars overlaid against the best-fit structure function or differential cross-section.
2. **Valence PDFs:** Reconstructed $u_v(x)$ and $d_v(x)$ distributions with $1\sigma$ uncertainty bands.
3. **Relative Fractional Uncertainty:** Log-scale comparison of $\delta q_v(x) / q_v(x)$ to highlight kinematic regions where data constraints weaken (such as low-$`x`$ and high-$`x`$ extrapolations).

---

## Key Observations and Limitations
-**Linear vs. Non-Linear Error Propagation:** The Hessian approximation assumes a quadratic $\chi^2$ minimum, whereas Monte Carlo bootstrapping captures non-linear parameter correlations and boundary effects without relying on finite-difference step sizes.
-**Kinematic Coverage:** In Drell-Yan simulations at fixed invariant mass $M$ and $\sqrt{s}$, kinematic cuts on $x_2 = M^2 / (x_1 s)$ restrict phase space, increasing fractional uncertainties at extreme $x$ values compared to full-range DIS data.
-**Future Extensions:** Incorporating real experimental datasets (e.g., HERA, LHC), DGLAP $Q^2$-evolution, and sea-quark/gluon distributions.

---

## Acknowledgements
* Supervised by **Prof. R. S. Thorne** at **University College London**.
* Built using standard scientific Python libraries (`NumPy`, `SciPy`, `Matplotlib`).
