#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Reads one case at a time from /input/<id>.in (or $MAKEFUN_INPUT_DIR):
every trading market of the event, each with its Yes price on a 1-minute
grid for the 72 hours up to a checkpoint. Writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR): for each market and each of the 240 minutes after the
checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

Method -- price-only, no news, no network, fully deterministic (variant F):
- Median: the market's last price, pulled toward its board-sum-normalised
  share. The Yes prices sum to ~1, so a market's fair share is
  x_i + (1 - S) * w_i / W, where S is the board sum and the adjustment is
  shared across markets in proportion to w_i = x_i * sigma_i (liquid
  markets absorb more of the gap). The pull is asymmetric: when the board
  is under-priced (gap > 0, sum < 1) it closes only 0.38 as fast as the
  symmetric model assumes; when over-priced (gap < 0) it closes 1.6x.
  Pull strength grows with horizon as (1 - exp(-h/110)) and is scaled by
  the volatility regime rscale (see below); a small pull-down of
  0.015 * x_i * pull(h) is also applied.
- Give-back: after a 1-minute rise of >= 0.02 the price gives back the
  whole rise with time constant 90 minutes; after a fall of the same size
  it gives back almost nothing (falls persist, rises revert).
- Deadband: when a row's median displacement from the last price is under
  0.0006 the median is set to the last price exactly (most 240-minute
  moves are exactly zero) and that row's spread is compressed x0.8.
- Spread: per-market, a 0.65/0.35 blend of a fast (15-min half-life) and
  a slow (4-hour half-life) EWMA-RMS of 1-minute changes (winsorised
  at +-0.06), with a price-level floor (0.0002 + 0.005 * x) and a 0.008
  cap, then halved (x0.5). The board's mean sigma V measures the
  volatility regime against the practice median Vmed = 0.001988; the pull
  is scaled by rscale = clip(1 + 2.3 * (V - Vmed) / Vmed, 0.6, 2.5).
- Fan: Laplace quantiles (scaled so the 5%/95% points match the Gaussian
  width), with the outer pair widened x1.1 and the inner pair narrowed
  x0.7 for fat tails; the spread grows like sqrt(h).

Verified with the local scorer on the six practice cases: mean pinball
loss 0.240411 (median error 0.6583, 90% coverage 0.8932; per case p00
0.3336, p01 0.2913, p02 0.2838, p03 0.2960, p04 0.1772, p05 0.1019) and
on five fresh cases (2026-11-13): mean loss 0.123218 (median error
0.2901, coverage 0.8977). Two runs are byte-identical.

Standard library only. Runs in well under the 8-second per-case cap.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Laplace inverse CDF at the quantiles (scale b = 1), scaled so the 5%/95%
# points match the Gaussian z-values.
ZL = [math.log(0.1), math.log(0.5), 0.0, -math.log(0.5), -math.log(0.1)]
ZSCALE = 1.644854 / 2.302585
Z = [z * ZSCALE for z in ZL]
# Fan multiplier per quantile level: outer pair widened, inner pair narrowed.
FAN_MULT = [1.1, 0.7, 1.0, 0.7, 1.1]

VMED = 0.001988


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def dual_sigma(prices, x):
    """Raw dual-EWMA spread of one market from its recent price moves.

    Returns the RAW spread (before the x0.5 calibration): the forecast
    spread is raw * 0.5, while the pull weights and the board regime V
    use the raw value.
    """
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    diffs = [max(-0.06, min(0.06, d)) for d in diffs]
    diffs = diffs[-1440:]
    if len(diffs) < 30:
        return 0.0025

    def ewma_rms(halflife):
        # EWMA of squared moves, returned as an RMS (root mean square).
        var = sum(d * d for d in diffs) / len(diffs)
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        for d in diffs:
            var = alpha * d * d + (1.0 - alpha) * var
        return math.sqrt(var)

    # Blend the two RMS values (not the variances).
    raw = 0.65 * ewma_rms(15.0) + 0.35 * ewma_rms(240.0)
    return min(0.008, max(0.0002 + 0.005 * x, raw))


def forecast_case(case):
    """All markets' 240-row quantile forecasts for one case."""
    markets = case["markets"]
    horizon = int(case["horizon_minutes"])
    # Per-market last price and dual-EWMA spread, computed for all markets
    # first: the board-sum pull needs every market's sigma.
    xs = []
    sigs = []
    for m in markets:
        prices = m.get("prices") or []
        x = prices[-1] if prices else 0.5
        xs.append(x)
        sigs.append(dual_sigma(prices, x))
    total = sum(xs)
    weights = [x * s for x, s in zip(xs, sigs)]
    wsum = sum(weights)
    gaps = []
    for x, w in zip(xs, weights):
        if wsum > 0:
            target = x + (1.0 - total) * w / wsum
        else:
            target = x
        gaps.append(target - x)
    vmean = sum(sigs) / len(sigs) if sigs else VMED
    rscale = min(2.5, max(0.6, 1.0 + 2.3 * (vmean - VMED) / VMED))

    forecasts = []
    for m, x, sig, gap in zip(markets, xs, sigs, gaps):
        prices = m.get("prices") or []
        d1 = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
        k = 0.38 if gap > 0 else 1.6
        rows = []
        for h in range(1, horizon + 1):
            pull = 1.0 * (1.0 - math.exp(-h / 110.0))
            med = x + rscale * k * pull * gap - 0.015 * x * pull
            if d1 >= 0.02:
                med -= 1.0 * d1 * (1.0 - math.exp(-h / 90.0))
            spread = sig * 0.5 * math.sqrt(h)
            if abs(med - x) < 0.0006:
                med = x
                spread *= 0.8
            rows.append([price6(med + z * fm * spread)
                         for z, fm in zip(Z, FAN_MULT)])
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = forecast_case(case)
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
