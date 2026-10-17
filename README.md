# LSTM_Wrapper

## About

LSTM_Wrapper is a C++/Qt research wrapper for training and evaluating LSTM
time-series models. It includes reusable time-series containers, multiple
training modes, helper utilities, and sample rainfall-runoff and other temporal
datasets.

## Repository contents

- `lstmtimeseriesset.*`: LSTM-oriented time-series data structures;
- `train_modes.*`: training and evaluation workflows;
- `helpers.*`: shared data and modeling utilities;
- `RR*.csv` and `Google2016-2019*.csv`: example datasets; and
- `LSTM_Wrapper.pro`: Qt/qmake project definition.

## Building

Open `LSTM_Wrapper.pro` in a compatible Qt environment and build the project.
Review local paths, sequence construction, normalization, train/test separation,
and random seeds before running an experiment.

## Status

This repository contains research code. Model skill should be assessed on
independent time periods before predictions are used operationally.
