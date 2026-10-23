# Data

* `raw/`: place `5EnvSensor_MayToDec2024_180kRows.csv` here (not committed; about 353 MB).
* `sample/co2_clean.csv`: cleaned CO₂ readings for the two sensors used in the project,
  written by `notebooks/01_data_audit.ipynb`. Columns:

  | Column | Meaning |
  | --- | --- |
  | `sensor` | Sensor ID (`6012002000869` = validation, `6012002000326` = application) |
  | `time` | Reading time, local (Australia/Melbourne), ISO 8601 with UTC offset |
  | `co2_ppm` | CO₂ concentration in ppm |

Source: MMA3001 Dataset 2, Monash Smart Infrastructure Occupancy and Environmental
Data, supplied by the teaching team via Moodle. Data ownership remains with Monash
University; it is not covered by this repository's MIT licence.
