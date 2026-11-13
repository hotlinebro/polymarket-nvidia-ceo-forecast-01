# Polymarket NVIDIA CEO Forecast

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`: forecast the Yes
prices of every market of Polymarket's "Next CEO of NVIDIA?" event, minute
by minute, 4 hours ahead.

## What the program does

`main.py` (Python 3.13, standard library only) reads one case from
`/input/<id>.in` — each market's Yes price on a 1-minute grid for the 72
hours up to a checkpoint — and writes `/output/<id>.out`: for each market
and each of the 240 minutes after the checkpoint, five quantiles (5%, 25%,
50%, 75%, 95%) of that minute's price. It runs offline, with no network,
and two runs on the same case write identical bytes.

## Method

Price-only, no news. The program cannot know about any scandal or
announcement — it only sees the price history, and the forecast is built
from that alone.

- **Median: the last price.** A Yes price is a market martingale, so the
  last price is the best point forecast. No drift is added: any drift would
  be a bet against market efficiency.
- **Spread: per-market, from recent realised volatility.** For each market,
  the 1-minute price differences over the last 4 hours give an
  exponentially weighted RMS (halflife 1 hour), shrunk 60/40 toward the
  house spread of 0.0025 and clamped to [0.0008, 0.005]. Calm markets get a
  much sharper forecast than the house baseline; wild markets get a wider
  one. This calibration — not direction — is the edge over the house
  baseline, which uses a fixed 0.0025 spread for every market at every
  time.
- **Horizon: spread grows like sqrt(h)**, the random-walk scaling.
  Quantiles come from the standard normal (Acklam's inverse CDF),
  clamped to [0, 1] and rounded to 6 decimals.

Deliberately left out: a drift toward the sum of all Yes prices being 1
(the sum runs a little above 1 in the practice cases, but the gap looks
like a persistent midpoint artefact, so correcting for it is an unverified
bet either way).

## Narrative (not in the program)

The scenario that motivated the entry: a scandal forces Jensen Huang out,
and the successor markets — not the stock — are the prices that move. The
program itself stays purely data-driven.

## Results

Practice grading (6 cases, local, vs house baseline pinball loss 0.363589):
mean pinball loss **0.313232** (baseline 0.363589, **−0.050358**) — beats
the baseline on all 6 cases: p00 0.473830 vs 0.536138, p01 0.337318 vs
0.380352, p02 0.355131 vs 0.408408, p03 0.343498 vs 0.368599, p04 0.236726
vs 0.298794, p05 0.181354 vs 0.238232. Median error identical to the
baseline (0.767503 — same persistence median); 90% coverage 0.971484 vs
baseline 0.949642. Runtime 0.20–0.31 s per case (cap 8 s); output
deterministic (byte-identical across runs), 240×5 quantiles per market,
monotone, in [0, 1], ≤0.22 MiB per case.

## License

MIT (see `LICENSE`).
