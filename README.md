# Prony Stabilization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

A Python package implementing stabilized Prony's method for exponential analysis. This code accompanies the Master's thesis of **Hari Rajagopal**, University of Passau (2026).

---

## Features

- Prony's method with oversampling for noise robustness  
- SVD-based ESPRIT for pole estimation  
- Condition number analysis of Hankel matrices  
- Residual (data misfit) metrics  
- Sensitivity analysis tools  
- Monte Carlo experiment framework  
- Publication-ready plotting functions  

---

## Requirements

- Python ≥ 3.9  
- NumPy ≥ 1.21  
- SciPy ≥ 1.7  
- Matplotlib ≥ 3.4  
- Pandas ≥ 1.3  
- Plotly ≥ 5.3  
- OpenPyXL ≥ 3.0  
- Seaborn ≥ 0.12  

---

## Installation

Clone the repository, create a virtual environment, and install in editable mode:

```bash
git clone https://github.com/hrganayath/prony-stabilization.git
cd prony-stabilization
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

---

## Quick Start

```python
import numpy as np
from prony import prony_method, generate_clean_data, match_estimates

# Create a synthetic signal with 2 exponentials
dt = 1.0 #sampling interval (seconds); adjust as needed
t = np.arange(30)*dt
a_true = np.array([1.0, 0.5])
# omega_true: continuous-time complex exponents (sigma + j*omega)
# signal model: y[k] = sum_i a_i * exp(omega_i * t[k])
omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
y = generate_clean_data(t, a_true, omega_true)

# Apply Prony's method with oversampling factor 2
a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)

# Match estimates to true parameters
a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)

print("True amplitudes:", a_true)
print("Estimated amplitudes:", a_est_m)
print("True exponents:", omega_true)
print("Estimated exponents:", omega_est_m)
print("Condition number of Hankel matrix:", cond)
```

---

## Project Structure

```
prony-stabilization/
├── src/
│   └── prony/
│       ├── __init__.py
│       ├── core.py
│       ├── residual.py
│       ├── utils.py
│       └── data.py
├── experiments/
│   ├── run_experiments.py
│   ├── plotting.py
│   ├── comparison_tables.py
│   └── sensitivity_analysis.py
├── tests/
│   ├── test_core.py
│   ├── test_utils.py
│   └── test_residual.py
├── README.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

---

## Reproducing Thesis Experiments

To reproduce all numerical experiments:

```bash
python -m experiments.run_experiments
```

This will:

- Run Monte Carlo simulations across noise levels and oversampling factors  
- Generate aggregated CSV result files  
- Create Excel comparison tables  
- Produce thesis figures (PNG)  
- Generate interactive Plotly visualizations  

Outputs are saved in the `results/` directory.

---

## Documentation

### Core Functions

- `prony_method(y, oversampling_factor, n)` — main algorithm  
- `compute_residual_error(...)` — error analysis  
- `match_estimates(...)` — parameter matching  
- `frequency_error_generic(...)` — frequency error metric  
- `generate_clean_data(...)` — synthetic signal generation  
- `generate_noisy_data(...)` — noise injection  

Refer to docstrings for detailed documentation.

---

## Results

Key findings from the thesis:

- Oversampling improves Hankel conditioning  
- Fixed-window evaluation enables fair comparison across oversampling factors  
- SVD-based ESPRIT is robust for pole estimation under noise  
- Column scaling stabilizes amplitude recovery  

Example outputs generated:

- RMSE heatmaps  
- Conditioning analysis plots  
- Sensitivity curves  
- Interactive error visualizations  

---

## Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a branch (`git checkout -b feature/name`)  
3. Commit changes (`git commit -m "description"`)  
4. Push (`git push origin feature/name`)  
5. Open a Pull Request  

---

## License

Licensed under the MIT License. See [`LICENSE`](LICENSE).

---

## Citation

If you use this code in research:

```bibtex
@mastersthesis{Rajagopal2026,
  author = {Hari Rajagopal},
  title = {Stabilization of Prony's Method for Exponential Analysis},
  school = {University of Passau},
  year = {2026}
}
```

---

## Contact

**Hari Rajagopal**  
Email: hrg.anayath@gmail.com  
GitHub: https://github.com/hrganayath/prony-stabilization  

---

## Acknowledgments

- University of Passau  
- Thesis advisor and committee  
- Scientific Python open-source ecosystem  

---

**Last updated:** February 2026

