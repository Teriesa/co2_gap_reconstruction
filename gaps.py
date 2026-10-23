"""Masked-gap creation and missing-value reconstruction methods.

Validation strategy: sensor 6012002000869 has no gaps, so stretches of its real data
are hidden ("masked"), reconstructed by each method, and compared against the true
hidden values.

Implemented in Steps 3–5 of the project plan.
"""

from __future__ import annotations

#: Nominal sampling interval of the environmental sensors [minutes].
SAMPLE_INTERVAL_MIN = 10
