# Notebooks

Run in numbered order. Each notebook imports functions from `src/co2gaps` rather than
defining them inline, so the tested code and the analysis code are the same.

| Notebook | Purpose | Plan step |
| --- | --- | --- |
| `00_setup_colab.ipynb` | Clone repo, install, link Drive data, run tests (run first in every Colab session) | 0 |
| `01_data_audit.ipynb` | Load, parse, de-duplicate and audit the raw extract | 1 |
| `02_gap_reconstruction.ipynb` | Masked-gap experiments for every method | 3–5 |
| `03_validation_and_selection.ipynb` | Metrics by gap length, final test, apply to sensor …326 | 6 |
| `04_performance.ipynb` | Timing, profiling, FLOP / arithmetic-intensity estimates | 8 |
