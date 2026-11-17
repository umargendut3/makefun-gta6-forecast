[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/umargendut3/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/umargendut3/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — variant D (umargendut3)

A log-scale quantile forecaster anchored on the first hour's print.

## The model

Past big launches carry their release-day hourly curves inside each case.
For a launch, the log ratio of hour h's viewers to hour 1's viewers follows
an S-shape in log space: a fast early dip as the midnight spike fades, then
a slow logistic rise to a plateau as the rolling midnight unlock reaches
Europe and the Americas, easing off a little in the last hours. The size of
that rise shrinks with the size of the launch — a small launch grows
several-fold through the day, a mega launch is already near its peak in
hour 1 — so each hour's log ratio is regressed on the log of hour 1's print
and read at this case's own print (clamped to three log units either side of
the fitted launches' median, so an unprecedented launch saturates instead of
extrapolating wildly).

Only console-only launches (no PC in their platform list) with a (nearly)
complete release-day curve are fitted, never the target game's own
pre-release category curve. The spread at each hour is the honest spread of
those launches' log ratios, floored, so the 5%..95% band is as wide as past
launches say it should be.

Quantiles: `q_tau(h) = p1 * exp(m_h + z_tau * s_h)`.

## Local validation

Leave-one-out over the 8 console-only launches in the practice cases
(each rebuilt as a synthetic case from its own release day, the target left
out of the fit): mean pinball loss **0.1703** against **0.3226** for the flat
persistence baseline — 7 wins out of 8, mean 90% coverage 0.880. Practice
cases run in ~0.25 s each, outputs byte-identical across runs.

## Run

```
python3 -P -s main.py   # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Python 3.13 standard library only, no network, deterministic.
