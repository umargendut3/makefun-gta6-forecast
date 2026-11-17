"""gta6-twitch-ccu-forecast-01 entry, variant D (umargendut3).

A log-scale quantile forecaster anchored on the first hour's print.

The model, in prose.  Every past big launch carried in the case's own
comparables has an hourly curve for its release day.  For a launch write
r_h for the log ratio of hour h's viewers to hour 1's viewers.  Two facts
the comparables show, and this program fits from them at run time:

  1. The shape of r_h over the day is an S-curve in two parts: a fast
     early dip (the midnight spike fading as the first time zones go to
     sleep) and then a slow logistic rise to a plateau as the rolling
     midnight unlock reaches Europe and the Americas, easing off a little
     in the last hours.  It is a Gompertz/logistic shape in log space:
     a quick move early, a flat tail late.
  2. The size of the rise shrinks with the size of the launch.  A small
     launch grows several-fold through the day; a mega launch is already
     near its peak in hour 1.  So r_h is regressed, hour by hour, on the
     log of hour 1's print, and the regression is read at this case's own
     hour-1 print (clamped to three log units either side of the fitted
     launches' median log print, so an unprecedented launch saturates
     instead of extrapolating wildly).

Only launches like this one are fitted: console-only releases (no PC in
their platform list) whose release-day curve is (nearly) complete, and
never the target game's own pre-release category curve.  A PC launch
unfolds on a different clock (one simultaneous global unlock), and mixing
it in wrecks the shape.  The spread is the honest one: the spread of the
fitted launches' log ratios at each hour, floored, so the 5%..95% band is
as wide as past launches say it should be, no wider.

Quantiles: q_tau(h) = p1 * exp(m_h + z_tau * s_h), p1 the hour-1 print,
m_h the fitted median log ratio, s_h the fitted log spread, z_tau the
standard normal quantile at the level.  Python 3.13 standard library
only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# How far, in natural-log units, the case's hour-1 print may sit from the
# fitted launches' median log print before the regression is clamped.
LOG_PRINT_CLAMP = 3.0
# The narrowest log spread any hour may carry.
SPREAD_FLOOR = 0.10
# A launch counts as fitted when this many of its 24 release-day hours
# are present with positive viewers.
MIN_DAY_HOURS = 20
# How many console-only launches must be fittable before they are used
# alone; below that, every complete launch is used.
MIN_CONSOLE_POOL = 4


def norm_ppf(p):
    """Acklam's approximation to the inverse standard normal CDF."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    """An ISO 8601 timestamp to an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def clean_quantiles(qs):
    """Clamp to (0, 1e9] and force non-decreasing, left to right."""
    out = []
    prev = 0.01
    for q in qs:
        value = min(max(float(q), 0.01), 1e9)
        if value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def is_console_only(comp):
    """True when the platform list names no PC release."""
    platforms = comp.get("platforms") or ""
    tokens = str(platforms).replace("/", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
    return not any(token.upper() == "PC" for token in tokens)


def release_day_hours(comp):
    """Hour 1..24 of the comparable's release day: {hour: viewers}."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            when = parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        offset = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= offset <= 24:
            hours[offset] = float(viewers)
    return hours


def select_pool(case):
    """The launches to fit: console-only ones with a (nearly) full day.

    The target game's own pre-release category curve is never a launch,
    so it is left out by name.  When too few console-only launches
    qualify, every complete launch is used instead.
    """
    game_name = str((case.get("game") or {}).get("name", "")).strip().lower()
    complete = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name", ""))
        lowered = name.lower()
        if game_name and (game_name in lowered or lowered in game_name):
            continue
        hours = release_day_hours(comp)
        if hours.get(1) and sum(1 for h in range(1, 25) if h in hours) >= MIN_DAY_HOURS:
            complete.append({
                "name": name,
                "p1": hours[1],
                "hours": hours,
                "console": is_console_only(comp),
            })
    console = [c for c in complete if c["console"]]
    if len(console) >= MIN_CONSOLE_POOL:
        return console
    return complete


def fit_shape(pool, log_p1_case):
    """Per hour, the OLS line of log ratio on log hour-1 print.

    Returns {hour: (median log ratio at this case's print, log spread)}.
    The spread is the raw spread of the fitted launches' log ratios at
    that hour, so it carries the slope's own uncertainty too.
    """
    log_p1s = [math.log(c["p1"]) for c in pool]
    centre = median(log_p1s)
    log_p1 = min(max(log_p1_case, centre - LOG_PRINT_CLAMP),
                 centre + LOG_PRINT_CLAMP)
    shape = {}
    for hour in range(2, 25):
        xs = []
        ys = []
        for comp in pool:
            viewers = comp["hours"].get(hour)
            if viewers is not None:
                xs.append(math.log(comp["p1"]))
                ys.append(math.log(viewers / comp["p1"]))
        if not xs:
            shape[hour] = (0.0, 0.30)
            continue
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = 0.0
        if denom > 0:
            slope = sum((xs[i] - mean_x) * (ys[i] - mean_y)
                        for i in range(n)) / denom
        median_log_ratio = (mean_y - slope * mean_x) + slope * log_p1
        if n > 1:
            spread = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
        else:
            spread = 0.30
        shape[hour] = (median_log_ratio, max(SPREAD_FLOOR, spread))
    return shape


def hour_one_print(case):
    """Hour 1's print, falling back to the last whole-hour mean, then 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    for row in reversed(case.get("history_hourly") or []):
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    return 1.0


def forecast_case(case, name):
    levels = [float(v) for v in case.get("levels", LEVELS)]
    if len(levels) != len(LEVELS) or any(not 0.0 < v < 1.0 for v in levels):
        levels = list(LEVELS)
    zetas = [norm_ppf(level) for level in levels]
    p1 = hour_one_print(case)
    wanted = [row["hour"] for row in case.get("forecast") or []]
    hours = sorted(h for h in wanted if isinstance(h, int)) or list(range(2, 25))
    pool = select_pool(case)
    rows = []
    if len(pool) >= 2:
        shape = fit_shape(pool, math.log(p1))
        for hour in hours:
            med, spread = shape.get(hour, (0.0, 0.30))
            quantiles = [round2(p1 * math.exp(med + z * spread)) for z in zetas]
            rows.append({"hour": hour, "q": clean_quantiles(quantiles)})
    else:
        # Degenerate fallback, used only when the case carries almost no
        # usable past launch: a flat median with a widening random-walk
        # spread, so the output is still valid.
        for hour in hours:
            quantiles = [round2(p1 * math.exp(z * 0.20 * math.sqrt(hour - 1)))
                         for z in zetas]
            rows.append({"hour": hour, "q": clean_quantiles(quantiles)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
