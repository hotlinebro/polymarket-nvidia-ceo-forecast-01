#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Reads one case at a time from /input/<id>.in (or $MAKEFUN_INPUT_DIR):
every trading market of the event, each with its Yes price on a 1-minute
grid for the 72 hours up to a checkpoint. Writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR): for each market and each of the 240 minutes after the
checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

Method -- price-only, no news, no network, fully deterministic:
- Board sum: S is the sum of the last prices of all markets in the case.
  The candidates are mutually exclusive, so the board should sum to ~1;
  when it does not, every price is pulled toward its board-sum-implied
  level x/S, linearly over the horizon, plus a small downward drift of
  3% of the last price over the horizon.
- Median (50% quantile) at minute h: x + (h/240)*(x/S - x) - 0.03*x*(h/240),
  where x is the market's last price.
- Spread: per-market, from recent realised volatility. For each market we
  take the 1-minute price differences over the last 4 hours, compute an
  exponentially weighted RMS (halflife 1 hour), blend it 90/10 toward the
  house spread 0.0025, clamp it to [0.0008, 0.005], and calibrate it by
  0.55. A market with too little history falls back to the house spread.
- Fan: Laplace quantile offsets (inverse Laplace CDF at each level),
  rescaled so the 90% width matches a Gaussian's (x1.644854/2.302585).
  Quantiles are clamped to [0, 1] and rounded to 6 decimals. Two runs on
  the same case write identical bytes.

Practice loss ~0.2699 (house baseline 0.3636, house model 0.313).

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
SIGMA_MIN = 0.0008            # floor: never forecast a dead-calm market
SIGMA_MAX = 0.005             # cap: never forecast a wider band than this
SIGMA_CALIB = 0.55            # calibration multiplier on the spread
MIN_DIFFS = 30                # fewer differences -> fall back to HOUSE_SIGMA
DRIFT = 0.03                  # downward drift, fraction of x over horizon
# Laplace fan rescale: match a Gaussian's 90% width.
ZSCALE = 1.644854 / 2.302585


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def laplace_z(tau):
    """Inverse Laplace CDF (scale 1) at level tau."""
    if tau < 0.5:
        return math.log(2.0 * tau)
    if tau > 0.5:
        return -math.log(2.0 * (1.0 - tau))
    return 0.0


def market_sigma(prices):
    """Per-minute spread of one market from its recent price moves."""
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    diffs = diffs[-WINDOW_MINUTES:]
    if len(diffs) < MIN_DIFFS:
        base = HOUSE_SIGMA
    else:
        # Start the EWMA at the window's mean square, then weight recent
        # moves more heavily.
        var = sum(d * d for d in diffs) / len(diffs)
        alpha = 1.0 - 0.5 ** (1.0 / HALFLIFE_MINUTES)
        for d in diffs:
            var = alpha * d * d + (1.0 - alpha) * var
        raw = math.sqrt(var)
        base = min(SIGMA_MAX, max(SIGMA_MIN, 0.9 * raw + 0.1 * HOUSE_SIGMA))
    return base * SIGMA_CALIB


def forecast_market(market, board_sum, horizon, z):
    """horizon rows of quantiles for one market, around its pulled median."""
    prices = market.get("prices") or []
    x = prices[-1] if prices else 0.5
    x_norm = x / board_sum if board_sum > 0 else x
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        pf = h / float(horizon)
        med = x + pf * (x_norm - x) - DRIFT * x * pf
        s = sigma * math.sqrt(h)
        rows.append([price6(med + zi * s) for zi in z])
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
        z = [laplace_z(t) * ZSCALE for t in quantiles]
        markets = case["markets"]
        board_sum = sum((m.get("prices") or [0.5])[-1] for m in markets)
        if board_sum <= 0:
            board_sum = 1.0
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, board_sum, horizon, z)}
            for m in markets
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
