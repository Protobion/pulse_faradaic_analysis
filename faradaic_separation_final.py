import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def get_pulse_segments(
    current_threshold,
    voltage,
    voltage_threshold,
    current,
):
    """
    Detect pulse segments using voltage and current thresholds.

    A data point belongs to a pulse segment when both of the following
    conditions are true:

        voltage > voltage_threshold
        current > current_threshold

    Parameters
    ----------
    current_threshold : float
        Minimum current in mA. The fallback value used by the main function
        is 0.0 mA.

    voltage : numpy.ndarray
        Voltage values in V.

    voltage_threshold : float
        Minimum voltage in V. The fallback value used by the main function
        is 1.229 V (minimum required voltage for water splitting at standard
        conditions).

    current : numpy.ndarray
        Current values in mA.

    Returns
    -------
    segments : list of tuple
        List of ``(start_idx, end_idx)`` tuples identifying detected pulses.

        Index-convention note:
        - For segments that end before the dataset ends, ``end_idx`` is stored
          as the last index that satisfies both threshold conditions
          (inclusive end index).
        - For a segment that continues to the end of the dataset, ``end_idx``
          is stored as ``len(data)`` (exclusive-style sentinel).

        Because later processing uses Python slicing ``[start:end]`` (exclusive
        upper bound), completed segments and trailing segments do not follow one
        uniform end-index convention. See README limitations for details.
    """
    above = (
        (voltage > voltage_threshold)
        & (current > current_threshold)
    )

    segments = []
    start = None

    for i, is_above in enumerate(above):
        if is_above and start is None:
            # Beginning of a new pulse segment.
            start = i

        elif not is_above and start is not None:
            # The pulse ended at the preceding data point.
            segments.append((start, i - 1))
            start = None

    if start is not None:
        # The final pulse continues through the end of the data.
        segments.append((start, len(voltage)))

    return segments


def calculate_mean_time(
    TimeData,
    segments,
    MeanTimeMin=30.0,
):
    """
    Calculate the complete averaging-window duration.

    MeanTime is selected so that it is:

    1. At least ``MeanTimeMin`` seconds.
    2. A whole-number multiple of the pulse-cycle length.

    The pulse-cycle length is determined from the elapsed time between the
    start of consecutive detected pulse segments. A start-to-start interval
    includes both the ON and OFF portions of a pulse cycle.

    ``get_current_GC_injection()`` divides MeanTime by two internally 
    because it uses half of the complete window before the GC injection
    and half after it.

    For example, if MeanTimeMin is 30 seconds and the pulse cycle is
    12 seconds:

        MeanTime = ceil(30 / 12) * 12 = 36 seconds

    The averaging range will then be 18 seconds before to 18 seconds after
    each GC injection (the center point of this averaging window can be
    shifted by ShiftTime, see below).

    Parameters
    ----------
    TimeData : pandas.Series
        Time values in seconds.

    segments : list of tuple
        Pulse segments returned by ``get_pulse_segments()``.

    MeanTimeMin : float, optional
        Minimum complete averaging-window duration in seconds. Default is
        30 seconds.

    Returns
    -------
    MeanTime : float
        Complete averaging-window duration in seconds.

    cycle_length : float
        Detected pulse-cycle length in seconds.

    cycle_lengths : list of float
        Start-to-start durations for all available pulse cycles.

    Raises
    ------
    ValueError
        If fewer than two pulse segments are available, or if a valid
        positive pulse-cycle length cannot be determined.
    """
    if MeanTimeMin <= 0:
        raise ValueError("MeanTimeMin must be greater than zero.")

    if len(segments) < 2:
        raise ValueError(
            "At least two pulse segments are required to determine the "
            "complete ON/OFF pulse-cycle length."
        )

    # Extract the time at which each detected pulse begins.
    pulse_start_times = [
        float(TimeData.iloc[start_idx])
        for start_idx, _ in segments
    ]

    # Start-to-start durations include both the ON and OFF pulse portions.
    cycle_lengths = [
        round(pulse_start_times[i] - pulse_start_times[i - 1])
        for i in range(1, len(pulse_start_times))
    ]

    # Ignore nonpositive durations, which cannot represent a pulse cycle.
    cycle_lengths = [
        length for length in cycle_lengths
        if np.isfinite(length) and length > 0
    ]

    if not cycle_lengths:
        raise ValueError(
            "A positive pulse-cycle length could not be determined from "
            "the detected segment start times."
        )

    # Preserve the original use of the maximum detected cycle length.
    cycle_length = max(cycle_lengths)

    # MeanTime is at least MeanTimeMin and is a whole multiple of the cycle.
    MeanTime = (
        math.ceil(MeanTimeMin / cycle_length)
        * cycle_length
    )

    return MeanTime, cycle_length, cycle_lengths


def robust_line_fit(
    x,
    y,
    NormalizeCurrent,
    max_iterations=10,
    threshold=0.5,
    max_transient_time=0.49
):
    """
    Fit a line while iteratively removing points with large residuals.

    This is used for the current-versus-inverse-square-root-time fit.

    Parameters
    ----------
    x : numpy.ndarray
        Fit x-coordinates.

    y : numpy.ndarray
        Fit y-coordinates.

    max_iterations : int, optional
        Maximum number of fitting iterations.
        Default: 10

    threshold : float, optional
        Residual threshold as a fraction of the residual standard deviation.
        Default: 0.5

    max_transient_time : float, optional
        Any data points after this time are considered faradaic.
        Default: 0.49 [s]

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model.

    inliers : numpy.ndarray
        Boolean mask identifying points retained by the fit.
    """
    inliers = np.ones(len(x), dtype=bool)

    for _ in range(max_iterations):
        # Fit a line through the currently selected points.
        x_inliers = x[inliers].reshape(-1, 1)
        y_inliers = y[inliers]

        model = LinearRegression().fit(x_inliers, y_inliers)

        # Calculate residuals for all data points.
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = np.abs(y - y_pred)

        # Mark points with large residuals as outliers.
        outlier_mask = residuals > threshold * np.std(residuals)
        new_inliers = ~outlier_mask

        # Stop when the selected points no longer change.
        if np.all(inliers == new_inliers):
            break

        inliers = new_inliers

        # The final point is treated as faradaic if too few points are available.
        inliers[-1] = True

    # The first point is always an outlier.
    inliers[0] = False

    # Allow no more than approximately max_transient_time seconds of non-faradaic current.
    if NormalizeCurrent:
        inliers[x > x[0] + max_transient_time] = True
    else:
        inliers[x**-2 > x[0]**-2 + max_transient_time] = True

    return model, inliers


def robust_line_fit_ln(
    x,
    y,
    min_focus_fraction=0,
    max_focus_fraction=0.2,
    threshold=0.05,
    maxFar=0.49,
):
    """
    Fit the initial exponential current decay by performing a linear fit
    to ln(current) versus time.

    The beginning of each pulse is treated as the non-faradaic region. The
    function identifies the points associated with that initial exponential
    decay.

    Parameters
    ----------
    x : numpy.ndarray
        Time values in seconds.

    y : numpy.ndarray
        Natural logarithm of current in mA.

    min_focus_fraction : float, optional
        Minimum fraction of points considered during fitting.

    max_focus_fraction : float, optional
        Maximum fraction of points considered during fitting.

    threshold : float, optional
        Residual threshold as a fraction of the residual standard deviation.

    maxFar : float, optional
        Maximum duration, in seconds, considered part of the initial
        non-faradaic decay.

    Returns
    -------
    best_model : sklearn.linear_model.LinearRegression or None
        Best fitted model.

    best_inliers : numpy.ndarray
        Boolean mask identifying the initial non-faradaic points.
    """
    xBegin = x[0]
    xFull = x
    nPointsRemoved = 0

    # Ensure the fit begins near the peak current and first current drop.
    while (
        (y[0] < y[1] and len(y) > 2)
        or (
            y[0] - y[1] < y[1] - y[2]
            and len(y) > 3
        )
    ):
        y = y[1:]
        x = x[1:]
        nPointsRemoved += 1

    # Only use the initial maxFar seconds for the non-faradaic fit.
    if any(x < xBegin + maxFar):
        keep = x < xBegin + maxFar
        y = y[keep]
        x = x[keep]

        nPointsRemovedEnd = (
            len(xFull)
            - len(x)
            - nPointsRemoved
        )
    else:
        # No points are available for fitting.
        best_model = None
        best_inliers = np.zeros(len(xFull), dtype=bool)
        best_inliers[0] = True
        return best_model, best_inliers

    best_model = None
    best_inliers = None
    max_iterations = 100

    for focus_fraction in np.arange(
        min_focus_fraction,
        max_focus_fraction,
        0.01,
    ):
        inliers = np.ones(len(x), dtype=bool)
        ignorePoints = 0
        iteration = 0

        while iteration < max_iterations:
            # At least three points are required for this fitting procedure.
            if len(x) - ignorePoints < 3:
                break

            focus_size = int(focus_fraction * len(x))
            if focus_size < 3:
                focus_size = 3

            # Fit the first selected points in the pulse.
            focus_x = x[inliers][ignorePoints:focus_size]
            focus_y = y[inliers][ignorePoints:focus_size]

            if len(focus_x) < 2:
                break

            model = LinearRegression().fit(
                focus_x.reshape(-1, 1),
                focus_y,
            )

            y_pred = model.predict(x.reshape(-1, 1))
            residuals = np.abs(y - y_pred)

            new_inliers = (
                residuals
                < threshold * np.std(residuals)
            )

            # The pulse begins with a non-faradaic contribution.
            if ignorePoints == 0:
                new_inliers[0] = True
            else:
                new_inliers[0:ignorePoints] = (
                    [True] * ignorePoints
                )

            # Keep an isolated outlier if it lies between two inliers.
            for i in range(1, len(new_inliers) - 1):
                if (
                    not new_inliers[i]
                    and new_inliers[i - 1]
                    and new_inliers[i + 1]
                ):
                    new_inliers[i] = True

            # Save the result if the fitting mask has converged.
            if np.all(inliers == new_inliers):
                if (
                    best_inliers is None
                    or np.sum(new_inliers) > np.sum(best_inliers)
                ):
                    best_model = model
                    best_inliers = new_inliers

                break

            inliers = new_inliers
            iteration += 1

    if best_inliers is None:
        best_inliers = np.zeros(len(x), dtype=bool)
        best_inliers[0] = True

    # Reinsert points removed from the beginning.
    for _ in range(nPointsRemoved):
        best_inliers = np.insert(
            best_inliers,
            0,
            True,
        )

    # Reinsert points removed from the end.
    for _ in range(nPointsRemovedEnd):
        best_inliers = np.append(
            best_inliers,
            False,
        )

    return best_model, best_inliers

def exp_lin_fit(t, y):

    """
    Fit normalized current with an exponential-decay plus linear model.

    The model is:

        y(t) = a * exp(-k * (t - t0)) + m * (t - t0) + b

    where ``t0`` is the first time value in the provided segment. The fit is
    performed with bounded nonlinear least squares (``scipy.optimize.curve_fit``)
    using ``k >= 0``.

    In addition to fitted parameters, the function returns pointwise fractional
    contributions of the exponential and linear components relative to the
    fitted total signal.

    Parameters
    ----------
    t : array-like
        Time values for one pulse segment (seconds).

    y : array-like
        Normalized current values corresponding to ``t``.

    Returns
    -------
    f_exp : numpy.ndarray
        Fractional exponential contribution at each point:

            f_exp = exp_part / y_fit

        where ``exp_part = a * exp(-k * (t - t0))``.

    f_lin : numpy.ndarray
        Fractional linear contribution at each point:

            f_lin = lin_part / y_fit

        where ``lin_part = m * (t - t0) + b``.

    popt : numpy.ndarray
        Fitted model parameters ``[a, k, m, b]``.

    Notes
    -----
    - Time is shifted internally as ``x = t - t0`` for numerical stability.
    - If ``y_fit`` is very close to zero at any point, fractions at those
      points are set to ``NaN``.
    - Fractional contributions are based on the signed model components and are
      not constrained to [0, 1] in all cases.
    """

    from scipy.optimize import curve_fit

    t = np.asarray(t)
    y = np.asarray(y)

    t0 = t[0]
    x = t - t0  # shifted time for numerical stability

    def exp_linear(x, a, k, m, b):
        return a * np.exp(-k * x) + m * x + b

    # ---- reasonable initial guesses ----
    # linear guess from tail (last 30%)
    n_tail = max(5, int(0.3 * len(x)))
    m0, b0_tail = np.polyfit(x[-n_tail:], y[-n_tail:], 1)

    # transient amplitude guess: early value minus tail line at x=0
    a0 = max(y[0] - (m0 * x[0] + b0_tail), 1e-6)

    # decay-rate guess: rough timescale ~ first 10–20% of window
    window = x[-1] - x[0]
    k0 = 3.0 / max(window, 1e-6)   # mild default

    p0 = [a0, k0, m0, b0_tail]

    # bounds: k >= 0 for decay; others mostly free
    lb = [-np.inf, 0.0, -np.inf, -np.inf]
    ub = [ np.inf, np.inf, np.inf,  np.inf]

    popt, _ = curve_fit(
        exp_linear, x, y, p0=p0, bounds=(lb, ub), maxfev=20000
    )

    a, k, m, b = popt
    y_fit = exp_linear(x, *popt)

    exp_part = a * np.exp(-k * x)
    lin_part = m * x + b

    eps = 1e-12
    den = np.where(np.abs(y_fit) < eps, np.nan, y_fit)

    f_exp = exp_part / den
    f_lin = lin_part / den

    return f_exp, f_lin, popt


def _fit_faradaic_current(
    TimeData,
    VoltageData,
    CurrentData,
    segments,
    NormalizeCurrent,
    Voltages,
    Currents,
):
    """
    Identify the faradaic-current portion of each detected pulse.

    This function is an internal helper so the public 
    ``get_faradaic_current()`` function can also perform segment
    detection, MeanTime calculation, and per-injection averaging.

    - ``NormalizeCurrent=False``:
      Uses two classification fits:
        1) robust fit of ``ln(current)`` versus time (initial transient),
        2) robust fit of current versus ``t^(-1/2)``.
      The faradaic mask is the logical OR of the two fits.

    - ``NormalizeCurrent=True``:
      Normalizes current using the IV reference, performs an
      exponential+linear fit of normalized current versus time, and
      computes a pointwise linear fraction used to estimate faradaic
      current contribution (``Current_far_mA``).

    Parameters
    ----------
    TimeData : pandas.Series
        Time values in seconds.
    VoltageData : pandas.Series
        Voltage values in V.
    CurrentData : pandas.Series
        Current values in mA.
    segments : list of tuple
        Pulse segments returned by ``get_pulse_segments()``.
    NormalizeCurrent : bool
        Selects classification/decomposition path as described above.
    Voltages : array-like
        IV reference voltages used when ``NormalizeCurrent=True``.
    Currents : array-like
        IV reference currents used when ``NormalizeCurrent=True``.

    Returns
    -------
    fit_results : list of dict
        Pulse-by-pulse fitting/classification results. Returned keys differ
        between normalized and non-normalized paths (see README).
    """

    fit_results = []
    iPulse = 0

    for start, end in segments:
        if NormalizeCurrent:
            Time_norm = TimeData.values[start:end]
            Current_norm = (CurrentData.values[start:end]/
                            get_current_from_IV(VoltageData.values[start:end],
                            Voltages, Currents))
            mask = ((~np.isnan(Time_norm)) & (~np.isnan(Current_norm)))
            Time_norm = Time_norm[mask]
            Current_norm = Current_norm[mask]

            if(len(Time_norm) > 2 and len(Current_norm) > 2):
                f_exp, f_lin, popt = exp_lin_fit(Time_norm, Current_norm)
                Current_far = CurrentData.values[start:end][mask]*f_lin

                model, inliers = robust_line_fit(Time_norm, Current_norm, NormalizeCurrent)

                if (
                    np.count_nonzero(
                        np.diff(inliers.astype(int))
                    )
                    > 1
                ):   
                    inliers[inliers.argmax()+1:] = True #Make all True after first True
                    print(
                        "Warning: Multiple outlier segments detected "
                        f"in pulse #{iPulse} with linear fit. All outliers"
                        " following the first inlier are treated as inliers."
                    )

                AveTime = np.mean(Time_norm[inliers])
                allTime = Time_norm[inliers]
                allVoltage = VoltageData.values[start:end][mask][inliers]
                allCurrent = CurrentData.values[start:end][mask][inliers]

                fit_results.append(
                    {
                        "segment": (start, end),
                        "model": model,
                        "inliers": inliers,
                        "x": Time_norm,
                        "y": Current_norm,
                        "AveTime_s": AveTime,
                        "Time_s": allTime,
                        "Current_mA": allCurrent,
                        "Voltage_V": allVoltage,
                        "Voltage_ON_V": VoltageData.values[start:end][mask],
                        "Time_far_s": Time_norm,
                        "Current_far_mA": Current_far,
                        "fct_lin": f_lin,
                        "model_params": popt, # [a, k, m, b] for y = a*exp(-k*(t - t0)) + m*(t - t0) + b
                    }
                )
        else:
            # First fit: logarithm of current versus time.
            x_seg = TimeData.values[start:end]
            # if NormalizeCurrent:
            #     y_seg = np.log(CurrentData.values[start:end]/
            #                 get_current_from_IV(VoltageData.values[start:end],
            #                 Voltages, Currents))
            # else:
            y_seg = np.log(CurrentData.values[start:end])

            mask = (
                (~np.isnan(x_seg))
                & (~np.isnan(y_seg))
                & (~np.isinf(x_seg))
                & (~np.isinf(y_seg))
            )

            # Second fit: current versus inverse square root of time.
            x_seg2 = TimeData.values[start:end] ** (-0.5)
            # if NormalizeCurrent:
            #     y_seg2 = (CurrentData.values[start:end]/
            #             get_current_from_IV(VoltageData.values[start:end],
            #             Voltages, Currents))
            # else:
            y_seg2 = (CurrentData.values[start:end])
            V_seg2 = VoltageData.values[start:end]

            mask2 = (
                (~np.isnan(x_seg2))
                & (~np.isnan(y_seg2))
                & (~np.isinf(x_seg2))
                & (~np.isinf(y_seg2))
            )

            # Use only points valid for both fits.
            mask_overall = mask & mask2

            x_seg = x_seg[mask_overall]
            y_seg = y_seg[mask_overall]

            x_seg2 = x_seg2[mask_overall]
            y_seg2 = y_seg2[mask_overall]
            V_seg2 = V_seg2[mask_overall]
            # Store the voltage for all valid points in the detected ON segment,
            # before filtering the points using the faradaic classification.
            # This array is aligned with x_seg, y_seg, x_seg2, and y_seg2.
            Voltage_ON_V = V_seg2.copy()

            # At least three points are required.
            if len(x_seg) > 2 and len(x_seg2) > 2:
                model, inliers = robust_line_fit_ln(
                    x_seg,
                    y_seg,
                )

                # robust_line_fit_ln identifies the initial non-faradaic points.
                # Invert its result to identify the faradaic points.
                inliers = ~inliers

                if (
                    np.count_nonzero(
                        np.diff(inliers.astype(int))
                    )
                    > 1
                ):
                    print(
                        "Warning: Multiple outlier segments detected "
                        f"in pulse #{iPulse} with ln current fit."
                    )

                model2, inliers2 = robust_line_fit(
                    x_seg2,
                    y_seg2,
                    NormalizeCurrent,
                )

                if (
                    np.count_nonzero(
                        np.diff(inliers2.astype(int))
                    )
                    > 1
                ):
                    print(
                        "Warning: Multiple outlier segments detected "
                        f"in pulse #{iPulse} with sqrt time fit."
                    )

                if len(inliers) != len(inliers2):
                    print(
                        "Warning: Different numbers of data points "
                        f"were found for pulse #{iPulse}."
                    )

                # A point is treated as faradaic when selected by either fit.
                faradaic_mask = inliers | inliers2

                AveTime = np.mean(x_seg[faradaic_mask])
                allTime = x_seg[faradaic_mask]
                allVoltage = V_seg2[faradaic_mask]
                # if NormalizeCurrent:
                #     allCurrent = (y_seg2[faradaic_mask]*
                #                 get_current_from_IV(
                #                     allVoltage, Voltages, Currents))
                # else:
                allCurrent = y_seg2[faradaic_mask]

                fit_results.append(
                    {
                        "segment": (start, end),
                        "model": model,
                        "inliers": inliers,
                        "x": x_seg,
                        "y": y_seg,
                        "model2": model2,
                        "inliers2": inliers2,
                        "x2": x_seg2,
                        "y2": y_seg2,
                        "AveTime_s": AveTime,
                        "Time_s": allTime,
                        "Current_mA": allCurrent,
                        "Voltage_V": allVoltage,
                        "Voltage_ON_V": Voltage_ON_V,
                    }
                )

        print(
            "Please wait. Calculating faradaic current "
            f"for pulse #{iPulse}.",
            end="\r",
        )

        iPulse += 1

    return fit_results


def get_current_GC_injection(
    TimeData,
    VoltageData,
    CurrentData,
    GCInjectionTimes,
    MeanTime,
    segments,
    NormalizeCurrent,
    Voltages,
    Currents,
):
    """
    Calculate current and voltage results for each GC injection.

    Only points identified as faradaic by ``_fit_faradaic_current()`` are
    included in the current and voltage averages.

    Parameters
    ----------
    TimeData : pandas.Series
        Time values in seconds.

    VoltageData : pandas.Series
        Voltage values in V.

    CurrentData : pandas.Series
        Current values in mA.

    GCInjectionTimes : pandas.Series
        GC injection times in seconds, using the same time reference as
        TimeData. These values can be shifted by ShiftTime. For example,
        the time for averaging relevant values can be shifted to ShiftTime
        seconds before the GC injection.

    MeanTime : float
        Complete averaging-window duration in seconds. Half of this duration
        is used before the injection and half after the injection.

    segments : list of tuple
        Automatically detected pulse segments.

    Returns
    -------
    injection_results : pandas.DataFrame
        Per-injection results containing:

        - ``Time_s``
        - ``Current_A``
        - ``Voltage_V``
        - ``FaradaicTime_fct``
        - ``Current_far_A``
        - ``PulseON_fct``

    ``Current_far_A`` and ``PulseON_fct`` are populated only when
    ``NormalizeCurrent=True`` and are ``NaN`` otherwise.

    fit_results : list of dict
        Pulse-by-pulse faradaic fitting results.
    """
    # MeanTime represents the complete window. Divide it by two to obtain
    # the duration used on either side of each injection.
    MeanTimeHalf = MeanTime / 2

    fit_results = _fit_faradaic_current(
        TimeData,
        VoltageData,
        CurrentData,
        segments,
        NormalizeCurrent,
        Voltages,
        Currents,
    )

    if not fit_results:
        raise ValueError(
            "No pulse contained enough valid data points to calculate "
            "faradaic current."
        )

    # Add NaN between pulses so subtraction does not count the OFF time
    # between two separate pulse segments.
    times_ON = []
    times_ON_far = []

    for result in fit_results:
        times_ON.append(
            np.append(
                result["Time_s"],
                np.nan,
            )
        )

        if NormalizeCurrent:
            times_ON_far.append(
                np.append(
                    result["Time_far_s"],
                    np.nan,
                )
            )

    times_ON = pd.Series(
        np.concatenate(times_ON)
    )

    if NormalizeCurrent:
        times_ON_far = pd.Series(
            np.concatenate(times_ON_far)
        )

    # Combine all faradaic-current points from all fitted pulses.
    TimeEC = pd.Series(
        np.concatenate(
            [
                result["Time_s"]
                for result in fit_results
            ]
        )
    )

    CurrentEC = pd.Series(
        np.concatenate(
            [
                result["Current_mA"]
                for result in fit_results
            ]
        )
    )

    VoltageEC = pd.Series(
        np.concatenate(
            [
                result["Voltage_V"]
                for result in fit_results
            ]
        )
    )

    if NormalizeCurrent:
        TimeFar = pd.Series(
            np.concatenate(
                [
                    result["Time_far_s"]
                    for result in fit_results
                ]
            )
        )

        CurrentFar = pd.Series(
            np.concatenate(
                [
                    result["Current_far_mA"]
                    for result in fit_results
                ]
            )
        )

    GCCurrent = np.full(
        len(GCInjectionTimes),
        np.nan,
    )

    GCVoltage = np.full(
        len(GCInjectionTimes),
        np.nan,
    )

    fct_on = np.full(
        len(GCInjectionTimes),
        np.nan,
    )

    GCCurrent_far = np.full(
        len(GCInjectionTimes),
        np.nan,
    )

    fct_on_far = np.full(
        len(GCInjectionTimes),
        np.nan,
    )

    for m in range(len(GCInjectionTimes)):
        injection_time = GCInjectionTimes.iloc[m]
        window_start = injection_time - MeanTimeHalf
        window_end = injection_time + MeanTimeHalf

        # Skip an injection if the complete averaging window is not covered
        # by the available faradaic-current results.
        if (
            window_end > TimeEC.iloc[-1]
            or window_start < TimeEC.iloc[0]
        ):
            continue

        fullWindow = MeanTime

        minIndex = next(
            (
                idx
                for idx, value in times_ON.items()
                if value > window_start
            ),
            None,
        )

        maxIndex = next(
            (
                idx
                for idx, value in times_ON.items()
                if value > window_end
            ),
            None,
        )

        if minIndex is not None:
            onTime = times_ON.iloc[minIndex:maxIndex]

            # NaN values separating pulses prevent OFF intervals from being
            # included in the summed faradaic time.
            onTimeSum = np.sum(
                onTime.shift(-1) - onTime
            )

            fct_on[m] = onTimeSum / fullWindow

        if NormalizeCurrent:
            minIndex_far = next(
                (
                    idx
                    for idx, value in times_ON_far.items()
                    if value > window_start
                ),
                None,
            )

            maxIndex_far = next(
                (
                    idx
                    for idx, value in times_ON_far.items()
                    if value > window_end
                ),
                None,
            )

            if minIndex_far is not None:
                onTime_far = times_ON_far.iloc[minIndex_far:maxIndex_far]

                # NaN values separating pulses prevent OFF intervals from being
                # included in the summed ON time.
                onTimeSum_far = np.sum(
                    onTime_far.shift(-1) - onTime_far
                )

                fct_on_far[m] = onTimeSum_far / fullWindow

        points_in_window = (
            (TimeEC > window_start)
            & (TimeEC < window_end)
        )

        GCCurrent[m] = np.mean(
            CurrentEC[points_in_window]
        )

        GCVoltage[m] = np.mean(
            VoltageEC[points_in_window]
        )

        if NormalizeCurrent:
            points_in_window_far = (
                (TimeFar > window_start)
                & (TimeFar < window_end)
            )

            GCCurrent_far[m] = np.mean(
                CurrentFar[points_in_window_far]
            )

    injection_results = pd.DataFrame(
        {
            "Time_s": GCInjectionTimes.values,
            # Convert current from mA to A.
            "Current_A": GCCurrent * 1e-3,
            "Voltage_V": GCVoltage,
            "FaradaicTime_fct": fct_on,
            "Current_far_A": GCCurrent_far * 1e-3,
            "PulseON_fct": fct_on_far,
        }
    )

    return injection_results, fit_results


def get_faradaic_current(
    TimeData,
    VoltageData,
    CurrentData,
    GCInjectionTimes,
    NormalizeCurrent=False,
    Voltages=None,
    Currents=None,
    voltage_threshold=None,
    current_threshold=None,
    MeanTimeMin=30.0,
    ShiftTime=-30.0
):
    """
    Detect pulses and calculate faradaic-current results.

    This is the main standalone function.

    Input requirement
    -----------------
    TimeData, VoltageData, CurrentData, and GCInjectionTimes are assumed to
    be one-dimensional pandas Series. TimeData, VoltageData, and CurrentData
    must have the same length and must be positionally aligned.

    When ``NormalizeCurrent=True``, the measured current and IV-reference
    current must use the same sign convention and current units. The
    interpolated IV-reference current must also be positive and sufficiently
    different from zero wherever normalization is performed.

    Parameters
    ----------
    TimeData : pandas.Series
        Electrochemical measurement times in seconds.

    VoltageData : pandas.Series
        Measured voltage in V.

    CurrentData : pandas.Series
        Measured current in mA.

    GCInjectionTimes : pandas.Series
        GC injection times in seconds. These must use the same time reference
        as TimeData.

    NormalizeCurrent : bool, optional
        If True, normalize the measured pulse current by a reference current
        obtained from the supplied IV curve:

            normalized current = measured current / IV-reference current

        The IV-reference current is evaluated at the measured pulse voltage
        using linear interpolation. Voltages up to 0.4 V above the IV-curve
        range are evaluated using linear extrapolation from the last two IV
        points.

        Normalization is used only for fitting and classification. The
        current values returned in ``fit_results["Current_mA"]`` and
        ``injection_results["Current_A"]`` remain measured current values in
        mA and A, respectively.

        Default is False.

    Voltages : array-like or None, optional
        Reference IV-curve voltage values in V. Values must be finite,
        one-dimensional, and strictly increasing. Required when
        ``NormalizeCurrent=True``.

    Currents : array-like or None, optional
        Reference IV-curve current values in mA corresponding to
        ``Voltages``. Required when ``NormalizeCurrent=True``.

        The reference current should represent the expected faradaic current
        at each voltage with negligible capacitive or other non-faradaic
        contribution.

    voltage_threshold : float or None, optional
        Voltage threshold used for pulse detection, in V. If omitted or
        explicitly set to None, 1.229 V is used.

    current_threshold : float or None, optional
        Current threshold used for pulse detection, in mA. If omitted or
        explicitly set to None, 0.0 mA is used.

    MeanTimeMin : float, optional
        Minimum complete averaging-window duration in seconds.

        The final MeanTime is at least this long and is rounded up to a
        whole-number multiple of the detected ON/OFF pulse-cycle length.
        Default is 30 seconds.

    ShiftTime : float, optional
        Number of seconds to shift the window for averaging relevant values,
        compared to the provided GC injection times. To move the averaging
        window to an earlier time, the value needs to be negative.
        Default is -30 seconds.

    Returns
    -------
    injection_results : pandas.DataFrame
        Per-GC-injection current, voltage, and faradaic-time results.

    fit_results : list of dict
        Detailed pulse-by-pulse fit and faradaic-current results.

    segments : list of tuple
        Automatically detected pulse segments.

    MeanTime : float
        Complete averaging-window duration in seconds.

    cycle_length : float
        Pulse-cycle length used to determine MeanTime.

    cycle_lengths : list of float
        All detected start-to-start pulse-cycle lengths.
    """
    # Apply fallback thresholds only when no value was provided.
    if voltage_threshold is None:
        voltage_threshold = 1.229  # V

    if current_threshold is None:
        current_threshold = 0.0  # mA

    # Shift the time window for averaging relevant values relative to GC
    # injection times
    GCInjectionTimes += ShiftTime

    # Confirm the required pandas Series input type.
    named_inputs = {
        "TimeData": TimeData,
        "VoltageData": VoltageData,
        "CurrentData": CurrentData,
        "GCInjectionTimes": GCInjectionTimes,
    }

    for name, values in named_inputs.items():
        if not isinstance(values, pd.Series):
            raise TypeError(
                f"{name} must be a one-dimensional pandas Series."
            )

        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

    # Time, voltage, and current describe the same measurement points.
    if not (
        len(TimeData)
        == len(VoltageData)
        == len(CurrentData)
    ):
        raise ValueError(
            "TimeData, VoltageData, and CurrentData must have "
            "the same length."
        )

    if len(TimeData) < 2:
        raise ValueError(
            "At least two electrochemical data points are required."
        )

    # Detect segments directly from voltage and current.
    segments = get_pulse_segments(
        current_threshold=current_threshold,
        voltage=VoltageData.values,
        voltage_threshold=voltage_threshold,
        current=CurrentData.values,
    )

    if not segments:
        raise ValueError(
            "No pulse segments were detected using "
            f"voltage > {voltage_threshold} V and "
            f"current > {current_threshold} mA."
        )

    # Determine a complete averaging window that is at least MeanTimeMin
    # and is a whole multiple of the ON/OFF pulse-cycle length.
    MeanTime, cycle_length, cycle_lengths = calculate_mean_time(
        TimeData=TimeData,
        segments=segments,
        MeanTimeMin=MeanTimeMin,
    )

    # Calculate pulse fits and per-injection averages.
    injection_results, fit_results = get_current_GC_injection(
        TimeData=TimeData,
        VoltageData=VoltageData,
        CurrentData=CurrentData,
        GCInjectionTimes=GCInjectionTimes,
        MeanTime=MeanTime,
        segments=segments,
        NormalizeCurrent=NormalizeCurrent,
        Voltages=Voltages,
        Currents=Currents,
    )

    return (
        injection_results,
        fit_results,
        segments,
        MeanTime,
        cycle_length,
        cycle_lengths,
    )


def plot_faradaic_diagnostics(
    fit_results,
    plot_index=0,
    current_average="mean",
    NormalizeCurrent=False,
):
    """
    Plot diagnostic results for the faradaic-current classification.

    Several plots are generated:

    1. Average retained faradaic current for every fitted pulse.
    2. Simple linear fit of normalized current versus time.
    3. Exp. + linear fit of normalized current versus time.
    4. Plot faradaic current from exp. + linear fit.
    5. Exponential-transient classification using ln(current) versus time.
    6. Inverse-square-root-time classification using current versus t^(-1/2).
    7. All retained faradaic-current samples from all fitted pulses.

    Plots 1 and 7 are always generated.
    If the flag ``NormalizeCurrent`` is True, plots 2-4 are generated.
    If the flag ``NormalizeCurrent`` is False, plots 5 and 6 are generated.

    Parameters
    ----------
    fit_results : list of dict
        Pulse-level results returned by ``get_faradaic_current()``.

    plot_index : int, optional
        Zero-based position of the pulse to inspect in ``fit_results``.
        The default is 0, corresponding to the first fitted pulse.

        This is a position in ``fit_results``, not necessarily the original
        pulse number. Pulses with insufficient valid data are skipped and do
        not appear in ``fit_results``.

        It is recommended to pick plot_index according to any warnings that 
        may appear to inspect potentially faulty fits.

    current_average : {"mean", "rms"}, optional
        Method used to summarize the retained faradaic current in the first
        plot:

        - ``"mean"``: arithmetic mean current
        - ``"rms"``: root-mean-square current

        The default is ``"mean"``.

    NormalizeCurrent : bool, optional
        Selects which diagnostic branch to plot.

        - If True, diagnostics are based on normalized-current fitting and
          exponential+linear decomposition.
        - If False, diagnostics are based on the ln(current) and
          inverse-square-root-time classification fits.

        This should match the mode used when generating ``fit_results``.

    Raises
    ------
    ValueError
        If fit_results is empty or current_average is invalid.

    IndexError
        If plot_index is outside the available range.
    """
    if not fit_results:
        raise ValueError(
            "fit_results is empty. No successfully fitted pulses are "
            "available for plotting."
        )

    if not isinstance(plot_index, (int, np.integer)):
        raise TypeError("plot_index must be an integer.")

    if plot_index < 0 or plot_index >= len(fit_results):
        raise IndexError(
            f"plot_index={plot_index} is outside the available range "
            f"0 to {len(fit_results) - 1}."
        )

    if current_average not in {"mean", "rms"}:
        raise ValueError(
            "current_average must be either 'mean' or 'rms'."
        )

    # ------------------------------------------------------------------
    # Plot 1: average retained faradaic current for every fitted pulse
    # ------------------------------------------------------------------
    average_times = []
    average_currents = []

    for result in fit_results:
        current = np.asarray(result["Current_mA"], dtype=float)

        # Ignore non-finite values when calculating a representative current.
        current = current[np.isfinite(current)]

        if current.size == 0:
            average_current = np.nan
        elif current_average == "rms":
            average_current = np.sqrt(np.mean(current**2))
        else:
            average_current = np.mean(current)

        average_times.append(result["AveTime_s"])
        average_currents.append(average_current)

    average_label = (
        "RMS Faradaic Current [mA]"
        if current_average == "rms"
        else "Mean Faradaic Current [mA]"
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        average_times,
        average_currents,
        color="tab:blue",
    )
    plt.xlabel("Average Time [s]")
    plt.ylabel(average_label)
    plt.title("Retained Faradaic Current by Pulse")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Select one fitted pulse for detailed inspection.
    result = fit_results[plot_index]
    segment = result["segment"]

    # ------------------------------------------------------------------
    # Plot 2: Simple linear fit of normalized current versus time
    # ------------------------------------------------------------------
    if NormalizeCurrent:
        x = np.asarray(result["x"], dtype=float)
        y = np.asarray(result["y"], dtype=float)

        faradaic_mask = np.asarray(
            result["inliers"],
            dtype=bool,
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x[faradaic_mask],
            y[faradaic_mask],
            color="tab:blue",
            label="Classified as faradaic",
        )

        plt.scatter(
            x[~faradaic_mask],
            y[~faradaic_mask],
            color="tab:red",
            label="Initial transient",
        )

        model = result["model"]
        if model is not None and x.size > 0:
            sort_index = np.argsort(x)
            x_sorted = x[sort_index]

            plt.plot(
                x_sorted,
                model.predict(x_sorted.reshape(-1, 1)),
                color="tab:green",
                label="Linear fit of normalized current",
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Normalized Current")
        plt.title(
            "Normalized Current Fit\n"
            f"fit_results index {plot_index}, pulse segment {segment}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim([0, 5])
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 3: Exp. + linear fit of normalized current versus time
    # ------------------------------------------------------------------

    if NormalizeCurrent:
        t = np.asarray(result["x"], dtype=float)
        x = t - t[0]  # Shift time to start at zero for better visualization
        y = np.asarray(result["y"], dtype=float)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            t,
            y,
            label="Normalized current",
        ) 

        a, k, m, b = result["model_params"]
        y_exp = a * np.exp(-k * x)
        y_lin = m * x + b
        y_fit = y_exp + y_lin

        plt.plot(t, y_fit, color='tab:blue', lw=2.5, label='Combined fit (exp + line)')
        plt.plot(t, y_exp, color='tab:red', lw=2, ls='--', label='Exponential only')
        plt.plot(t, y_lin, color='tab:green', lw=2, ls='-.', label='Linear only')

        plt.xlabel('Time [s]')
        plt.ylabel('Normalized current')
        plt.title(
            "Exponential + Linear Fit of Normalized Current\n"
            f"fit_results index {plot_index}, pulse segment {segment}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 4: Plot faradaic current from exp. + linear fit
    # ------------------------------------------------------------------
    if NormalizeCurrent:
        t = np.asarray(result["x"], dtype=float)
        x = t - t[0]  # Shift time to start at zero for better visualization
        y = np.asarray(result["y"], dtype=float)

        fct_lin = result["fct_lin"]
        faradaic_current = result["Current_far_mA"]

        plt.figure(figsize=(8, 5))
        plt.scatter(
            t,
            faradaic_current,
            color='tab:blue',
            label='Faradaic current (from exp + linear fit)',
        )
        plt.xlabel('Time [s]')
        plt.title(
            "Faradaic Current from Exponential + Linear Fit\n"
            f"fit_results index {plot_index}, pulse segment {segment}"
        )
        ax1 = plt.gca()
        ax1.set_ylabel('Faradaic current [mA]', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = plt.gca().twinx()
        ax2.set_ylabel('Faradaic fraction (linear fit)', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        plt.plot(t, fct_lin, color='tab:green', lw=2.5, label='Faradaic fraction (linear fit)')       
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 5: ln(current) versus time
    # ------------------------------------------------------------------
    if not NormalizeCurrent:
        x = np.asarray(result["x"], dtype=float)
        y = np.asarray(result["y"], dtype=float)

        # In _fit_faradaic_current(), this mask has already been inverted.
        # True therefore means that the point is classified as faradaic.
        faradaic_mask_ln = np.asarray(
            result["inliers"],
            dtype=bool,
        )

        model = result["model"]

        plt.figure(figsize=(8, 5))

        plt.scatter(
            x[faradaic_mask_ln],
            y[faradaic_mask_ln],
            color="tab:blue",
            label="Classified as faradaic",
        )

        plt.scatter(
            x[~faradaic_mask_ln],
            y[~faradaic_mask_ln],
            color="tab:red",
            label="Initial transient",
        )

        # The model was fitted to points identified as belonging to the initial
        # exponential transient. Plot the prediction in increasing time order.
        if model is not None and x.size > 0:
            sort_index = np.argsort(x)
            x_sorted = x[sort_index]

            plt.plot(
                x_sorted,
                model.predict(x_sorted.reshape(-1, 1)),
                color="tab:green",
                label="Exponential-transient fit",
            )

        plt.xlabel("Time [s]")
        plt.ylabel("ln(Current [mA])")
        plt.title(
            "Exponential-Transient Classification\n"
            f"fit_results index {plot_index}, pulse segment {segment}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 6: current versus inverse square-root time
    # ------------------------------------------------------------------
    if not NormalizeCurrent:
        x2 = np.asarray(result["x2"], dtype=float)
        y2 = np.asarray(result["y2"], dtype=float)

        # In the current implementation, True values in inliers2 are treated
        # as faradaic when the two fitting masks are combined.
        faradaic_mask_sqrt = np.asarray(
            result["inliers2"],
            dtype=bool,
        )

        model2 = result["model2"]

        plt.figure(figsize=(8, 5))

        plt.scatter(
            x2[faradaic_mask_sqrt],
            y2[faradaic_mask_sqrt],
            color="tab:blue",
            label="Classified as faradaic",
        )

        plt.scatter(
            x2[~faradaic_mask_sqrt],
            y2[~faradaic_mask_sqrt],
            color="tab:red",
            label="Rejected by fit",
        )

        if model2 is not None and x2.size > 0:
            sort_index = np.argsort(x2)
            x2_sorted = x2[sort_index]

            plt.plot(
                x2_sorted,
                model2.predict(x2_sorted.reshape(-1, 1)),
                color="tab:green",
                label=r"Linear fit in $t^{-1/2}$ coordinates",
            )

        plt.xlabel(r"Time$^{-0.5}$ [s$^{-0.5}$]")
        plt.ylabel("Current [mA]")
        plt.title(
            "Inverse-Square-Root-Time Classification\n"
            f"fit_results index {plot_index}, pulse segment {segment}"
        )
        plt.legend()
        plt.grid(alpha=0.3)

        # For increasing physical time, t^(-1/2) decreases. Inverting the axis
        # makes physical time progress visually from left to right.
        plt.gca().invert_xaxis()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 7: all retained faradaic-current samples
    # ------------------------------------------------------------------
    valid_time_arrays = []
    valid_current_arrays = []

    for result_item in fit_results:
        time_values = np.asarray(
            result_item["Time_s"],
            dtype=float,
        )
        current_values = np.asarray(
            result_item["Current_mA"],
            dtype=float,
        )

        valid = (
            np.isfinite(time_values)
            & np.isfinite(current_values)
        )

        if np.any(valid):
            valid_time_arrays.append(time_values[valid])
            valid_current_arrays.append(current_values[valid])

    if valid_time_arrays:
        all_times = np.concatenate(valid_time_arrays)
        all_currents = np.concatenate(valid_current_arrays)

        plt.figure(figsize=(10, 5))
        plt.scatter(
            all_times,
            all_currents,
            s=10,
            color="tab:blue",
            label="Retained faradaic samples",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Current [mA]")
        plt.title("Retained Faradaic-Current Samples")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(
            "No finite retained faradaic-current samples are available "
            "for the combined plot."
        )

def get_current_from_IV(voltage, voltages, currents):
    """
    Calculate current values by linearly interpolating or extrapolating
    measured voltage-current data.

    Parameters
    ----------
    voltage : float, list, or numpy.ndarray
        Voltage value or values for which currents are requested.

    voltages : list or numpy.ndarray
        Measured voltage data. Values must be in strictly increasing order.

    currents : list or numpy.ndarray
        Measured current data corresponding to each measured voltage.

    Returns
    -------
    float or numpy.ndarray
        A float is returned for a single input voltage.
        A NumPy array is returned for multiple input voltages.

    Raises
    ------
    ValueError
        If:
        - The measured arrays are invalid.
        - An input voltage is below the minimum measured voltage.
        - An input voltage is more than 0.4 V above the maximum
          measured voltage.
    """

    # Convert all inputs to NumPy arrays so that both lists and arrays work.
    voltage = np.asarray(voltage, dtype=float)
    voltages = np.asarray(voltages, dtype=float)
    currents = np.asarray(currents, dtype=float)

    # The measured data must be one-dimensional.
    if voltages.ndim != 1 or currents.ndim != 1:
        raise ValueError("voltages and currents must be one-dimensional")

    # Each voltage must have one corresponding current.
    if len(voltages) != len(currents):
        raise ValueError(
            "voltages and currents must contain the same number of values"
        )

    # At least two points are needed to calculate the extrapolation slope.
    if len(voltages) < 2:
        raise ValueError(
            "At least two voltage-current data points are required"
        )

    # NaN or infinite values would make interpolation unreliable.
    if (
        not np.all(np.isfinite(voltages))
        or not np.all(np.isfinite(currents))
        or not np.all(np.isfinite(voltage))
    ):
        raise ValueError("All voltage and current values must be finite")

    # np.interp expects the measured voltages to be strictly increasing.
    if np.any(np.diff(voltages) <= 0):
        raise ValueError(
            "Measured voltages must be in strictly increasing order"
        )

    # Determine the allowed input range from the measured voltage data.
    minimum_voltage = voltages[0]
    maximum_voltage = voltages[-1]
    maximum_allowed_voltage = maximum_voltage + 0.4

    # Reject values below the measured range or more than 0.4 V above it.
    if np.any(
        (voltage < minimum_voltage)
        | (voltage > maximum_allowed_voltage)
    ):
        raise ValueError(
            f"Voltage values must be between {minimum_voltage:g} V "
            f"and {maximum_allowed_voltage:g} V"
        )

    # Interpolate all values inside the measured voltage range.
    current = np.interp(voltage, voltages, currents)

    # Calculate the slope between the final two measured points.
    # This slope is used to extrapolate above the measured maximum.
    final_slope = (
        (currents[-1] - currents[-2])
        / (voltages[-1] - voltages[-2])
    )

    # For voltages above the measured maximum, replace np.interp's result
    # with a linear extrapolation based on the final two measured points.
    current = np.where(
        voltage > maximum_voltage,
        currents[-1] + final_slope * (voltage - maximum_voltage),
        current
    )

    # Return a regular float for scalar input and an array otherwise.
    return float(current) if current.ndim == 0 else current