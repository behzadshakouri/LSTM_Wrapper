# LSTM_Wrapper

## About

LSTM_Wrapper is the C++/Qt research implementation of a long short-term memory
(LSTM) surrogate for Activated Sludge Model No. 1 (ASM1) simulations. The
primary workflow uses mlpack recurrent neural networks to emulate
wastewater-treatment outputs including nitrate (`NO`), ammonia (`NH`),
soluble COD (`sCOD`), total Kjeldahl nitrogen (`TKN`), volatile suspended
solids (`VSS`), and organic nitrogen (`ND`).

The code includes alternate time-series modes and legacy rainfall-runoff and
stock-market example files, but those are secondary examples rather than the
repository's principal scientific purpose.

## Repository contents

- `main.cpp`: ASM-mode selection, target configuration, and LSTM workflow;
- `train_modes.*`: training, loading, continuation, and evaluation modes;
- `lstmtimeseriesset.*`: LSTM-oriented time-series data structures;
- `helpers.*`: shared data preparation and run-reporting utilities;
- `RR*.csv` and `Google2016-2019*.csv`: secondary example datasets; and
- `LSTM_Wrapper.pro`: Qt/qmake project definition.

## Building and data configuration

Open `LSTM_Wrapper.pro` in a compatible Qt environment with the required
mlpack and Armadillo dependencies. Before building, replace machine-specific
data paths and confirm the ASM input columns and selected target in `main.cpp`.

Preserve chronological train/validation/test separation and reuse the training
normalization and sequence construction when evaluating saved models.

## Status

This is research software accompanying an ASM1 surrogate-modeling workflow.
Validate performance on independent influent scenarios before using predictions
for real-time control or digital-twin applications.

## Associated publication

Shakouri, B., Murthy, S., De Clippeleir, H., Lesnik, K., & Massoudieh, A.
(2026). “Surrogate modeling of ASM1 using feedforward and LSTM neural networks.”
*Journal of Water Process Engineering, 81*, 109246.
[https://doi.org/10.1016/j.jwpe.2025.109246](https://doi.org/10.1016/j.jwpe.2025.109246)
· [Google Scholar record](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=0lTtuMoAAAAJ&citation_for_view=0lTtuMoAAAAJ:4JMBOYKVnBMC)
