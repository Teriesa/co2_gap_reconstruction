#Reconstructing Missing Indoor CO₂ Measurements from Smart-Building Sensors

MMA3001 Numerical Methods and Machine Learning, Individual Project 
Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data
Author: Thi Thao Trang (Teriesa) Phan

##1. Problem

Ventilation control and indoor-air-quality reporting need continuous CO₂ records, but the wireless sensors drop readings. This project reconstructs the missing readings and decides which method to use for gaps of 1 to 24 hours.

Input: time-stamped indoor CO₂ readings (ppm, 10-minute sampling) and the positions of missing readings.
Output: reconstructed values for each missing reading, a flag for gaps too long to fill (over 24 h), and validation errors (RMSE, MAE, overshoot) per method and gap length.
Intended use: filling gaps before computing daily air-quality summaries; not a substitute for real measurements in compliance or safety decisions.

##2. Data

The raw extract has 179,447 rows from five sensors; 76% of the readings are repeated uploads, leaving 42,843 unique readings [01_data_audit, copies_by_sensor.csv]. Two sensors are usable:

Sensor	Role	Period	Readings	Gaps
…869 (Building 60, G.38)	Validation	29 Feb – 14 Jun 2024	13,800	longest 30 min
…326	Application	7 Sep 2025 – 22 Apr 2026	29,038	61 over 30 min, 2 over 24 h

[audit_chosen_sensors.csv]. Readings are checked against a plausible range of 350–5000 ppm and a rate-of-change limit of 1000 ppm/h; flagged readings are kept and reported, not deleted.

3. Computational solution

Missing readings are filled by linear interpolation between the last reading before the gap and the first reading after it. Gaps with more than 24 h missing are left empty. The workflow is a tested Python package, co2gaps:

io.py: load the raw CSV, drop repeated uploads, parse the JSON, find gaps, flag readings.
gaps.py: create masked gaps and fill gaps with the interpolation methods.
evaluate.py: compute RMSE, MAE and overshoot and build comparison tables.
baselines.py, ode.py: the alternative methods (Section 4); perf.py: timing and FLOP counts.
4. Alternative solutions

Eight alternatives were compared with linear interpolation on identical masked gaps:

Family	Methods
Interpolation	PCHIP, Akima, cubic spline
Regression baselines	mean (DummyRegressor), typical day (median by hour, weekday/weekend)
Hybrid	typical day + linearly interpolated anomaly
Physics	single-zone CO₂ mass-balance ODE, dC/dt = λ(C_out − C) + S(t), solved with RK4, forward only and with end correction
5. Validation

Real gaps have no known answer, so complete stretches of sensor …869 were hidden (masked gaps of 1, 3, 6, 12 and 24 h, 40 of each) and each method's reconstruction was scored against the hidden readings. Methods were developed and tuned on 29 Feb – 19 May 2024 with time-ordered 3-fold cross-validation, and the final choice was made once on the held-out test period 20 May – 14 Jun 2024.

Median RMSE on the held-out test period, sensor …869 [05_final_test_and_application, 04_test_869_results.csv], in ppm:

Method	1 h	3 h	6 h	12 h	24 h
Linear	9.0	10.3	10.9	13.0	16.1
PCHIP	9.2	10.6	11.6	14.3	16.3
ODE + end correction (RK4)	9.1	10.6	12.9	15.8	25.5
ODE, forward only (RK4)	10.6	11.9	14.9	20.2	32.7
Typical day + anomaly	9.5	11.7	14.1	18.1	34.9
Akima	10.2	11.5	15.1	31.3	60.1
Typical day	23.6	17.5	27.3	36.0	40.5
Mean	34.1	26.8	32.9	33.7	33.2
Cubic spline	16.2	31.3	60.5	136.7	266.4

Linear interpolation had the lowest error at every gap length. On sensor …326 (a different room) the ODE and hybrid methods were slightly more accurate for 12–24 h gaps (24 h: 20.6 and 20.8 vs 24.6 ppm) [04_check_326_results.csv]. Applied to …326, the method filled 2,141 gaps (2,433 readings) and left 2 gaps over 24 h empty; filling changed daily mean CO₂ by −9.3 to +1.8 ppm, median about 0 [04_326_gap_report.csv, 04_326_daily_mean_effect.csv].

6. Optimisation and performance

Profiling (cProfile, line_profiler) showed that parsing the JSON text takes almost all of the pipeline time [06_cprofile_top10.csv]. Dropping repeated uploads before parsing (io.drop_repeated_uploads) gives identical readings, makes the pipeline about 3× faster and cuts the raw table from 652 to 157 MB [06_optimisation.csv]. Linear interpolation needs about 5 FLOPs per filled reading (arithmetic intensity about 0.3 FLOP/byte), so gap filling is limited by Python overhead, not arithmetic; its cost grows linearly with the number of gaps [06_flops.csv, 06_scaling.csv]. Timings depend on the machine.

7. Limitations

Interpolation cannot recover events that start and end inside a gap; error grows with gap length; the best method depends on the room; gaps over 24 h are not filled. See the report for details.

8. AI use

AI tools were used during this project. How they were used and how their output was checked is described in the AI-reflection section of the report.

How to reproduce

Data. The raw file 5EnvSensor_MayToDec2024_180kRows.csv (about 353 MB) is supplied on the unit's Moodle page and is not committed. The cleaned series are committed as data/sample/co2_clean.csv, so notebooks 02–05 and the tests run without it; notebooks 01 and 06 need it.

Google Colab (as used for this project).

Put the raw file in Google Drive at MyDrive/MMA3001_project/ (or change DRIVE_CSV in the first code cell of each notebook).
Open notebooks/00_setup_colab.ipynb in Colab and run it: it clones the repository, installs the package, links the raw file and runs the tests.
Run notebooks 01 to 06 in order (see notebooks/README.md).

Locally (Python 3.10+).

bash
git clone https://github.com/Teriesa/co2_gap_reconstruction.git
cd co2_gap_reconstruction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# put the raw CSV at data/raw/5EnvSensor_MayToDec2024_180kRows.csv, then run the notebooks

Tests and documentation.

bash
pytest -v                                    # unit tests
pytest --junitxml=reports/test-report.xml    # saved test report
pdoc src/co2gaps -o docs                     # HTML documentation (open docs/index.html)
Repository structure
co2_gap_reconstruction/
├── src/co2gaps/     source package: io, gaps, evaluate, baselines, ode, perf
├── notebooks/       analysis notebooks 00–06, run in order
├── tests/           pytest unit tests
├── data/raw/        raw CSV (not committed)
├── data/sample/     cleaned CO₂ series used by notebooks 02–05 and the tests
├── results/         figures and tables produced by the notebooks
├── docs/            HTML documentation generated by pdoc
├── reports/         pytest test report
├── requirements.txt, pyproject.toml, LICENSE
Licence

MIT Licence; see LICENSE. The dataset remains the property of its owners and is not covered by this licence.
