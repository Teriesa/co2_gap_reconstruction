# Notebooks

Run in numbered order. Each notebook imports functions from `src/co2gaps` rather than
defining them inline, so the tested code and the analysis code are the same.

| Notebook | Purpose | Plan step |
| --- | --- | --- |
| `00_setup_colab.ipynb` | Clone repo, install, link Drive data, run tests (run first in every Colab session) | 0 |
| `01_data_audit.ipynb` | Load, parse, de-duplicate and audit the raw extract | 1 |
| `02_gap_reconstruction.ipynb` | Masked-gap comparison of the interpolation methods on the development period | 3 |
| `03_validation_and_selection.ipynb` | Regression baseline, final test, check on sensor …326, fill its real gaps | 4–6 |
| `04_performance.ipynb` | Timing, profiling, FLOP / arithmetic-intensity estimates | 8 |
