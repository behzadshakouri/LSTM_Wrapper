/**
 * @file helpers.cpp
 * @brief Implementation of LSTM Wrapper utility functions.
 *
 * Provides normalization, metric evaluation, shape validation,
 * data cube generation, CSV saving, and configuration utilities.
 *
 * Authors:
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include "helpers.h"
#include <iostream>
#include <sstream>
#include <QDebug>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include "train_modes.h"   // for NormalizationType enum

using namespace std;
using namespace arma;
using namespace mlpack::data;

/* ============================================================
 *                Metric Computation
 * ============================================================ */
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return accu(square(pred - Y)) / Y.n_elem;
}

double ComputeR2(arma::cube& pred, arma::cube& Y)
{
    arma::vec yTrue = vectorise(Y);
    arma::vec yPred = vectorise(pred);

    const double ss_res = accu(square(yTrue - yPred));
    const double mean_y = mean(yTrue);
    const double ss_tot = accu(square(yTrue - mean_y));

    return (ss_tot == 0.0) ? 0.0 : 1.0 - (ss_res / ss_tot);
}

/* ============================================================
 *                Normalization Utilities
 * ============================================================ */

void FitMinMaxPerRow(const arma::mat& data,
                     arma::rowvec& mins,
                     arma::rowvec& maxs,
                     bool normalizeOutputs,
                     size_t inputSize,
                     size_t outputSize)
{
    const size_t nRows = data.n_rows;
    mins.set_size(nRows);
    maxs.set_size(nRows);

    for (size_t r = 0; r < nRows; ++r)
    {
        if (!normalizeOutputs && r >= inputSize)
        {
            mins[r] = 0.0;
            maxs[r] = 1.0;
            continue;
        }

        double minVal = data.row(r).min();
        double maxVal = data.row(r).max();

        if (fabs(maxVal - minVal) < 1e-12)
            maxVal = minVal + 1e-12;

        mins[r] = minVal;
        maxs[r] = maxVal;
    }
}

void TransformMinMaxPerRow(arma::mat& data,
                           const arma::rowvec& mins,
                           const arma::rowvec& maxs,
                           bool normalizeOutputs,
                           size_t inputSize,
                           size_t /*outputSize*/)
{
    const size_t nRows = data.n_rows;

    for (size_t r = 0; r < nRows; ++r)
    {
        if (!normalizeOutputs && r >= inputSize)
            continue;

        double range = maxs[r] - mins[r];
        if (fabs(range) < 1e-12)
            range = 1.0;

        data.row(r) = (data.row(r) - mins[r]) / range;
    }
}

static void InverseMinMaxPerRow(arma::mat& data,
                                const arma::rowvec& mins,
                                const arma::rowvec& maxs)
{
    const size_t nRows = data.n_rows;
    for (size_t r = 0; r < nRows; ++r)
        data.row(r) = data.row(r) * (maxs[r] - mins[r]) + mins[r];
}

/* ============================================================
 *                Extended Normalization Interface
 * ============================================================ */
/**
 * @brief General normalization dispatcher supporting multiple modes.
 *
 * @param mode     NormalizationType enum (PerVariable, MLpackMinMax, ZScore, None)
 * @param train    Training data matrix (modified in-place)
 * @param test     Test data matrix (modified in-place)
 * @param mins     Output vector for min values (for inverse scaling)
 * @param maxs     Output vector for max/scale values (for inverse scaling)
 * @param normalizeOutputs Whether to normalize outputs
 * @param inputSize Number of input features
 */
void ApplyNormalization(NormalizationType mode,
                        arma::mat& train,
                        arma::mat& test,
                        arma::rowvec& mins,
                        arma::rowvec& maxs,
                        bool normalizeOutputs,
                        size_t inputSize)
{
    switch (mode)
    {
        case NormalizationType::PerVariable:
        {
            cout << "[Normalization] Using per-variable min–max scaling\n";
            arma::mat fullData = arma::join_rows(train, test);
            FitMinMaxPerRow(fullData, mins, maxs, normalizeOutputs, inputSize, 0);
            TransformMinMaxPerRow(train, mins, maxs, normalizeOutputs, inputSize, 0);
            TransformMinMaxPerRow(test , mins, maxs, normalizeOutputs, inputSize, 0);
            break;
        }

        case NormalizationType::MLpackMinMax:
        {
            cout << "[Normalization] Using MLpack MinMaxScaler (0–1)\n";
            MinMaxScaler scaler;
            scaler.Fit(train);
            scaler.Transform(train, train);
            scaler.Transform(test, test);

            // Store synthetic min/max placeholders for consistency
            mins.set_size(train.n_rows);
            maxs.set_size(train.n_rows);
            mins.fill(0.0);
            maxs.fill(1.0);
            break;
        }

        case NormalizationType::ZScore:
        {
            cout << "[Normalization] Using Z-Score standardization (mean/std)\n";
            const size_t nRows = train.n_rows;
            mins.set_size(nRows);
            maxs.set_size(nRows);

            // Compute per-variable mean and std from training set only
            for (size_t r = 0; r < nRows; ++r)
            {
                if (!normalizeOutputs && r >= inputSize)
                {
                    mins[r] = 0.0;
                    maxs[r] = 1.0;
                    continue;
                }

                double meanVal = arma::mean(train.row(r));
                double stdVal  = arma::stddev(train.row(r));

                if (stdVal < 1e-12)
                    stdVal = 1.0;

                mins[r] = meanVal;  // store mean
                maxs[r] = stdVal;   // store std

                train.row(r) = (train.row(r) - meanVal) / stdVal;
                test.row(r)  = (test.row(r)  - meanVal) / stdVal;
            }
            break;
        }

        case NormalizationType::None:
        default:
        {
            cout << "[Normalization] No normalization applied\n";
            mins.reset();
            maxs.reset();
            break;
        }
    }
}
