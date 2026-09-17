# Faradaic Current Separation for Pulsed Electrolysis

`faradaic_separation_final.py` processes time-resolved voltage and current data from pulsed electrolysis experiments using an external RC circuit. It detects the ON portions of voltage/current pulses, separates the initial transient response from the current classified as faradaic, and calculates current and voltage values associated with user-provided gas chromatograph (GC) injection times.

The code performs the following steps:

1. Detects pulse ON segments using voltage and current thresholds.
2. Determines the complete ON/OFF pulse-cycle duration.
3. Selects an averaging window that is at least 30 seconds by default and is a whole-number multiple of the pulse-cycle duration.
4. Uses two fitting methods to identify the faradaic portion of each ON segment (two different options are available, read below for more information).
5. Calculates the average faradaic current, voltage, and fraction of time classified as faradaic around each GC injection.

## Requirements

The code requires Python 3 and the following packages:

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl scipy
```

`matplotlib` is required by the module and is used for diagnostic plotting.
`openpyxl` is required when reading `.xlsx` files with pandas.

Save `faradaic_separation_final.py` in the same directory as the analysis script,
or install it as part of a Python package. It can then be imported using:

```python
import faradaic_separation_final as fase
```

## Input Data

The main function expects four one-dimensional `pandas.Series` objects:

- `TimeData`: time in seconds
- `VoltageData`: voltage in volts
- `CurrentData`: current in milliamperes
- `GCInjectionTimes`: GC injection times in seconds

`TimeData`, `VoltageData`, and `CurrentData` must:

- Have the same length.
- Be positionally aligned.
- Use the units listed above.

The GC injection times and electrochemical measurement times must use the same time reference.

### Sampling and Time-Axis Recommendations

`TimeData` should be monotonically increasing and should contain positive time
values because the inverse-square-root-time fit evaluates `t⁻¹ᐟ²`. Invalid,
`NaN`, or infinite transformed values are removed before fitting.

The sampling rate must be high enough to resolve the initial current transient.
Each pulse needs at least three valid data points for fitting, but substantially
more points are recommended. If the transient occurs between recorded samples,
the code cannot reliably distinguish it from the later current.

A reasonably uniform sampling rate is also recommended. Timestamp jitter,
missing samples, or a sampling rate that changes during a pulse can affect the
fitted masks and the calculated faradaic-time fraction.

The following recommendations apply to the input data:

- `TimeData`, `VoltageData`, and `CurrentData` must have equal lengths.
- The three series must describe the same measurement points and be positionally
  aligned.
- Time must be provided in seconds.
- Voltage must be provided in volts.
- Current must be provided in milliamperes.
- GC injection times and electrochemical times must use the same time reference.
- The time axis should be positive and monotonically increasing.

## Pulse Detection

An ON segment is detected when both of the following conditions are satisfied:

```text
Voltage > voltage_threshold
Current > current_threshold
```

The default thresholds are:

- Voltage threshold: `1.229 V`
- Current threshold: `0.0 mA`

These values can be changed when calling `get_faradaic_current()`.

The default voltage threshold corresponds to the thermodynamic water-splitting voltage under standard conditions. Depending on the experiment, cell configuration, voltage convention, and overpotentials, a different threshold may be more appropriate.

### Main Parameters

The following parameters can be passed directly to
`get_faradaic_current()`:

| Parameter | Default | Description |
| --- | ---: | --- |
| `voltage_threshold` | `1.229 V` | Minimum voltage used to identify an ON segment |
| `current_threshold` | `0.0 mA` | Minimum current used to identify an ON segment |
| `MeanTimeMin` | `30 s` | Minimum complete averaging-window duration |
| `ShiftTime` | `-30 s` | Shift of the averaging-window center relative to each GC injection |

The voltage and current thresholds should be selected so that they reliably
separate the ON portion of a pulse from the OFF or open-circuit portion.
Thresholds that are too close to the noise level may split a single physical
pulse into several detected segments.

The fitting functions also contain internal parameters:

| Internal parameter | Default | Description |
| --- | ---: | --- |
| `maxFar` | `0.49 s` | Maximum initial duration considered by the exponential-decay fit |
| `max_transient_time` | `0.49 s` | Points later than approximately 0.49 s after pulse onset are forced to be retained as faradaic by `robust_line_fit()` |
| Exponential-fit residual threshold | `0.05` | Residual criterion used by `robust_line_fit_ln()` |
| Inverse-square-root-fit residual threshold | `0.5` | Residual criterion used by `robust_line_fit()` |

These internal fitting parameters are not currently exposed through
`get_faradaic_current()`. Changing them requires calling or editing the helper
functions directly. The residual-threshold values can be important for
determining the non-faradaic and faradaic regions. It is recommended to evaluate
these values initially using representative data and then keep them constant
across datasets that are analyzed together. The diagnostic plots can help assess
the selected values.

## Separation of Transient and Faradaic Current

The beginning of an electrochemical pulse can contain transient current associated with processes such as double-layer charging and changes at the electrode interface.

The code uses two empirical fitting approaches to distinguish this initial response from the current classified as faradaic. When `NormalizeCurrent` is True, every data point can, in theory, contain faradaic and non-faradaic contributions. However, later data points are expected to have near-zero non-faradaic contribution.

### Physical Interpretation and Classification Strategy

The measured current during an ON pulse can be represented conceptually as:

```text
I_total(t) = I_cap(t) + I_far(t)
```

where:

- `I_cap(t)` is the transient current associated primarily with charging or
  discharging the electrochemical double layer.
- `I_far(t)` is the current associated with electrochemical reactions.

Only the faradaic contribution is directly associated with reactant conversion
and product formation. Using the complete measured current without accounting
for a large capacitive contribution can therefore bias a faradaic-efficiency
calculation.

The code does not explicitly fit and subtract `I_cap(t)` and `I_far(t)` at every
time point. Instead, it classifies measured samples as transient/non-faradaic or
faradaic using two complementary linearized fits.

The first fit examines `ln(I)` versus time. A purely exponential transient,

```text
I_cap(t) = I₀ exp(-t/τ),
```

appears as a straight line after taking the natural logarithm:

```text
ln(I_cap) = ln(I₀) - t/τ.
```

The second fit examines current versus `t⁻¹ᐟ²`. This is motivated by the
Cottrell-type response associated with diffusion-controlled current:

```text
I(t) ∝ t⁻¹ᐟ².
```

The two classification masks are combined using a logical OR:

```python
faradaic_mask = inliers | inliers2
```

A sample is therefore retained as faradaic if either fitting method classifies it
as faradaic. This union favors retaining possible reaction current rather than
aggressively discarding it. However, it may also retain some transient current
when the measurements are noisy or when the models do not adequately describe
the pulse.

Because the measured current is a sum of contributions,

```text
ln(I_cap + I_far)
```

is generally not exactly linear, even when `I_cap` follows a perfect exponential
decay. The fits should therefore be interpreted as empirical classification
tools rather than as a unique physical decomposition of the measured current.

### 1. Exponential-Decay Fit

The initial current transient is treated as an exponential decay:

```text
I(t) = I₀ exp(kt)
```

Taking the natural logarithm gives:

```text
ln(I) = ln(I₀) + kt
```

An exponential current decay therefore appears linear when `ln(current)` is plotted against time.

The code performs a robust linear fit to `ln(current)` versus time and identifies the initial fitted region as non-faradaic. By default, the fitting interval is limited to approximately the first `0.49 s` of each pulse.

### 2. Inverse-Square-Root-Time Fit

The second fit uses current as a function of inverse square-root time:

```text
I(t) ∝ t⁻¹ᐟ²
```

This behavior is associated with the Cottrell equation for diffusion-controlled electrolysis:

```text
I(t) = n F A C √D / √(πt)
```

where:

- `n` is the number of electrons transferred.
- `F` is the Faraday constant.
- `A` is the electrode area.
- `C` is the bulk reactant concentration.
- `D` is the diffusion coefficient.
- `t` is time.

The code robustly fits current against `t⁻¹ᐟ²` and uses the fitted data-point mask as a second indication of the faradaic region.

A data point is classified as faradaic when it is selected by either the inverted exponential-fit mask or the inverse-square-root-time fit mask.

## Averaging Around GC Injections

The code calculates a complete averaging-window duration called `MeanTime`.

The default minimum averaging duration is:

```python
MeanTimeMin = 30.0  # seconds
```

The pulse-cycle length is estimated from the start-to-start time between consecutive ON segments. A start-to-start interval contains both the ON and OFF portions of the pulse cycle.

`MeanTime` is rounded up to a whole-number multiple of the maximum detected pulse-cycle length:

```text
MeanTime = ceil(MeanTimeMin / cycle_length) × cycle_length
```

For example, for a 12-second pulse cycle:

```text
MeanTime = ceil(30 / 12) × 12
MeanTime = 36 seconds
```

The averaging window extends over half of `MeanTime` before and half after its center. In this example, it extends 18 seconds in each direction.

By default, the center of the averaging window is shifted 30 seconds earlier than the supplied GC injection time:

```python
ShiftTime = -30.0  # seconds
```

This shift can be changed using the `ShiftTime` argument.

## General Example

```python
# General example
import pandas as pd
FileName = ""
ExcelFile = pd.read_excel(FileName, sheet_name=None)

Data = ExcelFile["SheetName"]
Time = Data.Time # [s]
Voltage = Data.Voltage # [V]
Current = Data.Current # [mA]

import faradaic_separation_final as fase
injection_times_s = pd.Series([237, 747, 1258, 1768, 2278, 2788, 3298, 3808])

(   injection_results,
    fit_results,
    segments,
    MeanTime,
    cycle_length,
    cycle_lengths,
) = fase.get_faradaic_current(
    TimeData=Time,
    VoltageData=Voltage,
    CurrentData=Current,
    GCInjectionTimes=injection_times_s
)

# print(f"Detected segments: {segments}")
# print(f"Detected cycle lengths: {cycle_lengths} s")
# print(f"Cycle length used: {cycle_length} s")
# print(f"Complete averaging window: {MeanTime} s")

print("")
print(injection_results)
```

## Optional Parameters

The pulse-detection thresholds, minimum averaging duration, and GC time shift can be specified explicitly:

```python
results = fase.get_faradaic_current(
    TimeData=Time,
    VoltageData=Voltage,
    CurrentData=Current,
    GCInjectionTimes=injection_times_s,
    voltage_threshold=1.229,  # V
    current_threshold=0.0,    # mA
    MeanTimeMin=30.0,         # s
    ShiftTime=-30.0,          # s
)
```

## Returned Values

`get_faradaic_current()` returns six values.

### 1. `injection_results`

A pandas DataFrame containing one row per GC injection and the following columns:

- `Time_s`: shifted GC injection time and center of the averaging window
- `Current_A`: average current classified as faradaic, converted to amperes
- `Voltage_V`: average voltage over the selected faradaic data points
- `FaradaicTime_fct`: fraction of the complete averaging window classified as faradaic

`FaradaicTime_fct` is returned as a fraction between approximately `0` and `1`, rather than as a value between `0` and `100` (based on retained/classified points `Time_s`).

To convert it to percent:

```python
injection_results["FaradaicTime_percent"] = (
    injection_results["FaradaicTime_fct"] * 100
)
```

`FaradaicTime_fct` is calculated conceptually as:

```text
                    estimated faradaic-current duration
FaradaicTime_fct = --------------------------------------
                    complete averaging-window duration
```

The denominator includes the complete averaging window, including the OFF time
between pulses. The code inserts `NaN` separators between separate pulse
segments so that the OFF interval between two pulses is not accidentally counted
as faradaic time.

The result can be used to describe the fraction of wall-clock time during the GC
averaging window for which current was classified as faradaic. It is different
from the fraction of the ON segment that is classified as faradaic because its
denominator also includes the OFF portions of the pulse cycles.

If the retained faradaic points within one pulse are not contiguous, the
time-difference calculation may count an excluded internal gap as faradaic time.
Warnings about multiple outlier segments should therefore be investigated.

- `Current_far_A`: average estimated faradaic current contribution (A) from the exponential+linear decomposition (`NormalizeCurrent=True` path).  
  In workflows that do not produce decomposition-based values, this column may be `NaN`.
- `PulseON_fct`: fraction of the complete averaging window during which the pulse is ON (based on detected ON-segment time coverage `Time_far_s`).

### 2. `fit_results`

A list containing detailed pulse-by-pulse fitting results, including:

- Fitted models
- Fitting masks
- Pulse times
- Faradaic current values
- Voltage values
- Original pulse segment indices

The dictionaries in `fit_results` contain different fields depending on
`NormalizeCurrent`.

### When `NormalizeCurrent=False`

| Key | Description |
| --- | --- |
| `segment` | `(start, end)` indices of the detected pulse in the original data |
| `model` | Linear regression model fitted to `ln(I)` versus time (initial transient fit) |
| `inliers` | Inverted exponential-fit mask; `True` indicates a point classified as faradaic by the `ln(I)` method |
| `x` | Time values used for the `ln(I)` fit |
| `y` | Natural logarithm of current used for the exponential-decay fit |
| `model2` | Linear regression model fitted to current versus `t⁻¹ᐟ²` |
| `inliers2` | Mask from the inverse-square-root-time fit; `True` indicates a point classified as faradaic by that method |
| `x2` | Inverse-square-root-time values used for the second fit |
| `y2` | Current values used for the second fit, in mA |
| `AveTime_s` | Mean time of the retained faradaic samples after combining masks |
| `Time_s` | Time values retained as faradaic after combining masks |
| `Current_mA` | Current values retained as faradaic after combining masks |
| `Voltage_V` | Voltage values corresponding to retained faradaic samples |
| `Voltage_ON_V` | Voltage values for all valid fitting points in the detected ON segment, before final faradaic masking |

### When `NormalizeCurrent=True`

| Key | Description |
| --- | --- |
| `segment` | `(start, end)` indices of the detected pulse in the original data |
| `model` | Linear regression model from `robust_line_fit()` applied to normalized current versus time |
| `inliers` | Mask from `robust_line_fit()`; in this mode it is used as the retained/faradaic mask for `Time_s`, `Current_mA`, and `Voltage_V` |
| `x` | Time values (s) used for normalized-current fitting |
| `y` | Normalized current values used for fitting |
| `AveTime_s` | Mean time of retained points (`inliers=True`) |
| `Time_s` | Retained time values (`inliers=True`) |
| `Current_mA` | Retained measured current values (mA), corresponding to `Time_s` |
| `Voltage_V` | Retained voltage values (V), corresponding to `Time_s` |
| `Voltage_ON_V` | Voltage values for all valid points in the detected ON segment (before `inliers` filtering) |
| `Time_far_s` | Time values for all valid ON-segment points used in the exponential+linear decomposition |
| `Current_far_mA` | Estimated faradaic current contribution in mA for all valid ON-segment points, computed as measured current × `fct_lin` |
| `fct_lin` | Estimated linear (faradaic) fraction of the fitted normalized current at each point |
| `model_params` | Parameters `[a, k, m, b]` of `a·exp(-k·(t-t0)) + m·(t-t0) + b` |

The current implementation does not return an individual pulse-level mean
current, charge integral, or fitted RC time constant. These values
would need to be calculated separately from the returned pulse data and fitted
models.

### 3. `segments`

A list of index ranges for the detected ON segments:

```python
[(start_index, end_index), ...]
```

### 4. `MeanTime`

The complete averaging-window duration in seconds.

### 5. `cycle_length`

The maximum detected start-to-start pulse-cycle duration used to calculate `MeanTime`.

### 6. `cycle_lengths`

A list containing all valid start-to-start pulse-cycle durations.

## Diagnostics

The code prints progress information while processing each pulse and may issue
warnings when a classification appears irregular.

### Multiple outlier segments

A warning similar to the following may be printed:

```text
Warning: Multiple outlier segments detected in pulse #... with ln current fit.
```

or:

```text
Warning: Multiple outlier segments detected in pulse #... with sqrt time fit.
```

This means that the classification mask changes between faradaic and
non-faradaic more than once within a pulse. A physically reasonable
classification will often contain one initial transient region followed by one
contiguous faradaic region.

Multiple transitions may indicate:

- Measurement noise.
- Insufficient sampling of the initial transient.
- An irregular or interrupted pulse.
- Threshold crossings within a physical pulse.
- Fitting parameters that are unsuitable for the measured system.
- Behavior that is not well represented by either fitting model.

### Different numbers of data points

The following warning indicates that the two fitting masks are unexpectedly
misaligned:

```text
Warning: Different numbers of data points were found for pulse #...
```

The corresponding pulse-level results should be inspected before they are used
for quantitative calculations.

### Skipped pulses

A pulse must contain at least three valid data points after removal of `NaN` and
infinite values. Pulses with insufficient valid data are not added to
`fit_results`.

The number of successfully fitted pulses can be checked using:

```python
print(f"Detected pulse segments: {len(segments)}")
print(f"Successfully fitted pulses: {len(fit_results)}")
```

If these values differ, one or more detected segments did not contain enough
valid data for fitting.

### Recommended validation

The classification should be checked using representative pulses from each
experiment. In particular, verify that:

- The detected segments correspond to the physical ON portions of the pulses.
- The initial high-current transient is not incorrectly retained as faradaic.
- The faradaic points form a physically reasonable region.
- The detected cycle lengths agree with the programmed pulse period.
- `MeanTime` is appropriate for the GC measurement timing.
- `FaradaicTime_fct` is consistent across comparable GC injections.

The module includes `plot_faradaic_diagnostics()`, which uses the returned
`fit_results` to visualize the pulse classification and retained current values.

### Diagnostic Plots

The `plot_faradaic_diagnostics()` function generates several plots:

1. Average retained faradaic current for every fitted pulse.
2. Simple linear fit of normalized current versus time.
3. Exp. + linear fit of normalized current versus time.
4. Plot faradaic current from exp. + linear fit.
5. Exponential-transient classification using ln(current) versus time.
6. Inverse-square-root-time classification using current versus t^(-1/2).
7. All retained faradaic-current samples from all fitted pulses.

Plots 1 and 7 are always generated.
If the flag `NormalizeCurrent` is True, plots 2-4 are generated.
If the flag `NormalizeCurrent` is False, plots 5 and 6 are generated.

Example:

```python
fase.plot_faradaic_diagnostics(
    fit_results=fit_results,
    plot_index=0,
    current_average="mean",
    NormalizeCurrent=True,
)
```

`plot_index` is zero-based and refers to the position of a successfully fitted
pulse in `fit_results`. It does not necessarily equal the original pulse number
because detected pulses with fewer than three valid fitting points are skipped.

For example, the following selects the 16th successfully fitted pulse:

```python
fase.plot_faradaic_diagnostics(
    fit_results=fit_results,
    plot_index=15,
)
```

The `current_average` argument controls the summary shown in the first plot:

- `"mean"` calculates the arithmetic mean of the retained faradaic-current
  samples.
- `"rms"` calculates their root-mean-square value.

The current summary is calculated from `Current_mA` because the current
implementation does not store `AveCurrent_mA` in `fit_results`.

#### Interpreting the exponential-transient plot

The exponential-transient plot displays:

- **Red points:** samples classified as belonging to the initial transient.
- **Blue points:** samples classified as faradaic by the exponential-fit
  criterion.
- **Green line:** the fitted initial exponential behavior after linearization
  as `ln(current)` versus time.

A typical classification should contain an initial group of red points followed
by a group of blue points. Alternating red and blue regions can indicate noise,
insufficient sampling, or a poor fit.

The green line is a fit in logarithmic-current coordinates. It represents an
exponential current decay in the original current coordinates.

#### Interpreting the inverse-square-root-time plot

The inverse-square-root-time plot displays:

- **Blue points:** samples retained as inliers by the `I` versus `t⁻¹ᐟ²` fit.
- **Red points:** samples rejected by that fit.
- **Green line:** the fitted linear response in `t⁻¹ᐟ²` coordinates.

The horizontal axis is inverted because `t⁻¹ᐟ²` decreases as physical time
increases. This makes physical time progress visually from left to right.

#### Interpreting the combined current plot

The final plot contains only samples retained as faradaic after combining both
classification masks (when `NormalizeCurrent` is False). It is useful for checking:

- Changes in retained current over the complete experiment.
- Pulses with unexpectedly high retained current.
- Gaps caused by skipped or rejected data.
- Long-term drift in the electrochemical response.

## Limitations

- The separation is an empirical classification method. It does not independently measure faradaic and non-faradaic charge.

- The method classifies complete measured samples rather than explicitly
  separating every current value into capacitive and faradaic components.
  Current samples assigned to the initial transient are excluded from the
  per-injection average.

- Faradaic current may still flow during the interval classified as transient.
  That contribution is not included in the average of the retained samples.
  Conversely, transient current may be retained if either fitting method
  classifies it as faradaic. The result should therefore not automatically be
  interpreted as a strict lower or upper bound on the true faradaic current.

- The two masks are combined using a logical OR. This reduces the risk of
  discarding reaction current, but it increases the possibility of retaining
  non-faradaic transient current.

- The logarithm of the total current is generally not perfectly linear:

  ```text
  ln(I_cap + I_far) ≠ ln(I_cap) + ln(I_far)
  ```

  Even if the capacitive contribution is exponential, the simultaneous
  faradaic contribution introduces curvature into `ln(I_total)`. The transition
  identified by the exponential fit is therefore approximate and can be
  sensitive to noise.

- The code does not enforce one contiguous non-faradaic region followed by one
  contiguous faradaic region. Irregular masks are reported as warnings but are
  still used in the calculation.

- The input time axis is not explicitly checked for monotonicity. Unsorted or
  repeated time values can produce invalid cycle lengths, fits, and time
  fractions.

- Only positive current values can be represented by `ln(current)`. Zero or
  negative current values produce invalid logarithms and are removed from both
  fitting datasets by the combined validity mask.

- The exponential and Cottrell-type behaviors are simplified models. Real electrochemical transients may include multiple time constants, adsorption, resistance effects, changing electrode area, convection, bubble formation, or coupled reaction and mass-transport processes.

- The Cottrell relationship is normally expressed using the time elapsed since the beginning of a potential step. The current implementation uses the absolute values in `TimeData` when calculating `t⁻¹ᐟ²`, rather than resetting time to zero at the beginning of each pulse.

- Pulse detection depends strongly on the selected voltage and current thresholds. Noise or threshold crossings within a pulse can incorrectly split one pulse into multiple segments.

- At least two detected pulses are required to determine the complete ON/OFF pulse-cycle length.

- The code uses the maximum detected start-to-start cycle duration. A single missed or irregular pulse can therefore make `MeanTime` unnecessarily long.

- Cycle durations are rounded to whole seconds, which may be inappropriate for short or high-frequency pulses.

- The default `0.49 s` limit for the initial transient is fixed and may not be suitable for every electrochemical system or sampling frequency.

- The fits require at least three valid data points in each pulse. Short pulses or low-frequency measurements may not contain enough data.

- `FaradaicTime_fct` is estimated using discrete sampling intervals. Its accuracy depends on the acquisition frequency and regularity of the time spacing.

- GC injection windows that extend beyond the available fitted data return `NaN`.

- The current implementation shifts `GCInjectionTimes` in place using:

  ```python
  GCInjectionTimes += ShiftTime
  ```

  This may modify the original Series supplied by the user. Pass a copy if the original injection times must be retained:

  ```python
  GCInjectionTimes=injection_times_s.copy()
  ```

- For completed pulse segments, the stored end index is the last point satisfying the thresholds, while Python slicing excludes the end index. Consequently, the final qualifying data point of a completed segment may be omitted from fitting.

- The method should be validated using representative experimental data before its results are used for quantitative faradaic-efficiency calculations.

- The diagnostic plot index refers to a position in `fit_results`, not directly
  to the pulse number in the original data. These can differ when a detected
  pulse is skipped because it contains fewer than three valid data points.

- The first diagnostic plot summarizes only the current samples retained as
  faradaic. It does not calculate a current averaged over the complete ON/OFF
  pulse cycle.

- The green line in the exponential-transient plot is shown in `ln(current)`
  coordinates. It should not be interpreted as a linear current decay in the
  original current coordinates.

- The diagnostic plots support visual inspection but do not automatically
  determine whether a classification is physically correct.
