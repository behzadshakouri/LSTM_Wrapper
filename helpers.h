#pragma once
#include <armadillo>
#include <string>

#include <pch.h>  // must come first
using namespace mlpack;
using namespace mlpack::data;

/* ============================================================
 *                Metric Evaluation
 * ============================================================ */
double ComputeMSE(arma::cube& pred, arma::cube& Y);
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Shape Validation
 * ============================================================ */
void ValidateShapes(const arma::mat& dataset,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    size_t rho);

/* ============================================================
 *                Time-Series Data Builder
 * ============================================================ */
// rho = sequence length (number of time steps per input window)
inline void CreateTimeSeriesData(const arma::mat& dataset,
                                 arma::cube& X,
                                 arma::cube& y,
                                 const size_t rho,
                                 const int inputSize,
                                 const int outputSize,
                                 const bool IO)
{
    if (!IO)
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputSize - 1),
                               arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize, inputSize + outputSize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
    else
    {
        for (size_t i = 0; i < dataset.n_cols - rho; ++i)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(0, inputSize - 1),
                               arma::span(i, i + rho - 1));
            y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize - outputSize, inputSize - 1),
                               arma::span(i + 1, i + rho));
        }
    }
}

/* ============================================================
 *                Result Saving / Post-Processing
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO);
