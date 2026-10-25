"""Timing helpers and FLOP / arithmetic-intensity estimates (Week 2.1).

Used by ``notebooks/06_performance.ipynb``. FLOP counts are per floating-point
operation (add, subtract, multiply, divide each count as one), estimated from the
formulas the methods evaluate; bytes count 8-byte float64 values read and written.
"""

from __future__ import annotations

import statistics
import time

BYTES_PER_FLOAT = 8
"""Size of one float64 value in bytes."""

LINEAR_FLOPS_PER_SLOT = 5
"""C = C_a + (C_b - C_a) (t - t_a) / (t_b - t_a): 2 subtractions, 1 multiply,
1 divide, 1 addition per filled slot."""

LINEAR_BYTES_PER_SLOT = 2 * BYTES_PER_FLOAT
"""Read the slot time t and write the estimate C (the edge values stay in registers)."""

RK4_FLOPS_PER_STEP = 30
"""Four slope evaluations f(C) = lam (C_out - C) + S (3 FLOPs each = 12), the three
intermediate states (2 FLOPs each = 6), and the weighted update
C + h/6 (k1 + 2 k2 + 2 k3 + k4) (about 12): roughly 30 FLOPs per step."""

RK4_BYTES_PER_STEP = 3 * BYTES_PER_FLOAT
"""Read the step size and S, write C."""


def time_call(function, repeat=5):
    """Median wall-clock time of ``function()`` over ``repeat`` runs (seconds).

    The median is used, rather than the mean, because occasional runs are slowed by
    other activity on the machine.

    Parameters
    ----------
    function : callable
        Called with no arguments.
    repeat : int
        Number of timed runs (at least 1).

    Returns
    -------
    float
        Median elapsed time in seconds.
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def linear_flops(n_slots):
    """Estimated FLOPs to fill ``n_slots`` missing readings by linear interpolation."""
    return LINEAR_FLOPS_PER_SLOT * n_slots


def rk4_flops(n_steps):
    """Estimated FLOPs for ``n_steps`` RK4 steps of the CO₂ ODE."""
    return RK4_FLOPS_PER_STEP * n_steps


def arithmetic_intensity(flops, bytes_moved):
    """Arithmetic intensity, FLOPs per byte moved (Week 2.1 roofline model).

    Raises
    ------
    ValueError
        If ``bytes_moved`` is not positive.
    """
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved
