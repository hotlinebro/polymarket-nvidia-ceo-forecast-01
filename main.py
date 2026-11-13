#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Reads one case at a time from /input/<id>.in (or $MAKEFUN_INPUT_DIR):
every trading market of the event, each with its Yes price on a 1-minute
grid for the 72 hours up to a checkpoint. Writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR): for each market and each of the 240 minutes after the
checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

Method -- price-only, no news, no network, fully deterministic:
- Median (50% quantile): the market's last price (persistence). A Yes price
  is a market martingale, so the last price is the best point forecast; any
  drift we added would be a bet against market efficiency, so we add none.
- Spread: per-market, from recent realised volatility. For each market we
  take the 1-minute price differences over the last 4 hours, compute an
  exponentially weighted RMS (halflife 1 hour), shrink it 60/40 toward the
  house spread 0.0025, and clamp it to [0.0008, 0.005]. Calm markets get a
  much sharper forecast than the house baseline; wild markets get a wider
  one. This is the whole edge over the house baseline, which uses a fixed
  0.0025 spread for every market at every time.
- Horizon: the spread grows like sqrt(h), the random-walk scaling. Quantiles
  come from the standard normal, are clamped to [0, 1] and rounded to
  6 decimals. Two runs on the same case write identical bytes.

The house baseline (the kit's starter) is persistence with a fixed 0.0025
spread. We keep its median and beat it on calibration, not on direction.

Standard library only. Runs in well under the 8-second per-case cap.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The spread of a minute's price move the house baseline assumes, in price
# units per square-root minute.
HOUSE_SIGMA = 0.0025
# How much of each market's recent history sets its spread.
WINDOW_MINUTES = 240          # last 4 hours of 1-minute differences
HALFLIFE_MINUTES = 60         # EWMA halflife
SHRINK_TO_HOUSE = 0.4         # weight on HOUSE_SIGMA in the blend
SIGMA_MIN = 0.0008            # floor: never forecast a dead-calm market
SIGMA_MAX = 0.005             # cap: never forecast a wider band than this
MIN_DIFFS = 30                # fewer differences -> fall back to HOUSE_SIGMA


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, |err| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3])
                * r + b[4]) * r + 1)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def market_sigma(prices):
    """Per-minute spread of one market from its recent price moves."""
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    diffs = diffs[-WINDOW_MINUTES:]
    if len(diffs) < MIN_DIFFS:
        return HOUSE_SIGMA
    # Start the EWMA at the window's mean square, then weight recent moves
    # more heavily: a market that just jumped stays wide for a while.
    var = sum(d * d for d in diffs) / len(diffs)
    alpha = 1.0 - 0.5 ** (1.0 / HALFLIFE_MINUTES)
    for d in diffs:
        var = alpha * d * d + (1.0 - alpha) * var
    raw = math.sqrt(var)
    blended = (1.0 - SHRINK_TO_HOUSE) * raw + SHRINK_TO_HOUSE * HOUSE_SIGMA
    return min(SIGMA_MAX, max(SIGMA_MIN, blended))


def forecast_market(market, horizon, quantiles):
    """horizon rows of quantiles for one market, around its last price."""
    prices = market.get("prices") or []
    x = prices[-1] if prices else 0.5
    sigma = market_sigma(prices)
    z = [norm_ppf(t) for t in quantiles]
    rows = []
    for h in range(1, horizon + 1):
        s = sigma * math.sqrt(h)
        rows.append([price6(x + zi * s) for zi in z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        quantiles = [float(t) for t in case["quantiles"]]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon, quantiles)}
            for m in case["markets"]
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
