# Notebooks

Run in numbered order. Each notebook imports functions from `src/co2gaps` rather than
defining them inline, so the tested code and the analysis code are the same.

| Notebook | Purpose | Plan step |
| --- | --- | --- |
| `00_setup_colab.ipynb` | Clone repo, install, link Drive data, run tests (run first in every Colab session) | 0 |
| `01_data_audit.ipynb` | Load, parse, de-duplicate and audit the raw extract | 1 |
| `02_gap_reconstruction.ipynb` | Masked-gap comparison of the interpolation methods on the development period | 3 |
| `03_baselines.ipynb` | Regression baselines and the typical-day anomaly method, time-ordered cross-validation | 4 |
| `04_ode_alternative.ipynb` | Physics-based alternative: CO₂ mass-balance ODE (verification, calibration, time-ordered CV) | 5 |
| `05_final_test_and_application.ipynb` | Final test (used once), check on sensor …326, fill its real gaps, daily means | 6 |
| `06_performance.ipynb` | Timing, profiling, FLOP / arithmetic-intensity estimates | 8 |
