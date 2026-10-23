"""Reconstruction of missing indoor CO₂ measurements from smart-building sensors.

MMA3001 Numerical Methods and Machine Learning — individual project, Dataset 2.

Modules
-------
io
    Load the raw sensor CSV, parse its JSON payloads, remove duplicate uploads and
    produce an audit summary.
gaps
    Create artificial (masked) gaps with known true values and reconstruct missing
    samples using interpolation or regression methods.
evaluate
    Error metrics (RMSE, MAE, overshoot) and method-comparison tables.
perf
    Simple, repeatable timing helpers used in the performance analysis.
"""

__version__ = "0.1.0"
