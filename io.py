"""Loading, parsing, de-duplication and auditing of the raw sensor extract.

The raw CSV stores one upload per row. Each ``jsondata`` cell holds a JSON list of
measured variables (CO₂, temperature, humidity, ...), each with a Unix timestamp,
value and unit. Functions in this module turn that into tidy tables.

Implemented in Step 1 of the project plan.
"""

from __future__ import annotations

#: Sensor timestamps are reported in Unix seconds (UTC); analysis uses local time.
LOCAL_TIMEZONE = "Australia/Melbourne"

#: Physically plausible CO₂ range for an indoor NDIR sensor [ppm].
#: Values outside this range are flagged, not silently deleted.
CO2_VALID_RANGE_PPM = (350.0, 5000.0)
