/**
 * @file helpers.cpp
 * @brief Implementation of utility functions for LSTM Wrapper.
 */

#include "pch.h"
#include "helpers.h"
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>

using namespace std;
using namespace arma;
using namespace mlpack::data;

/* ============================================================
 *                Metrics
 * ============================================================ */
double ComputeMSE(const arma::cube& pred, const arma::cube& Y)
{
    return arma::accu(arma::square(pred - Y)) / static_cast<double>(Y.n_elem);
}

double ComputeR2(const arma::cube& pred, const arma::cube& Y)
{
    // Manual size check (works for all Armadillo versions)
    if (pred.n_rows != Y.n_rows ||
        pred.n_cols != Y.n_cols ||
        pred.n_slices != Y.n_slices)
    {
        std::cerr << "[ComputeR2] Error: prediction and target cubes have different shapes.\n";
        return arma::datum::nan;
    }

    arma::vec yTrue = arma::vectorise(Y);
    arma::vec yPred = arma::vectorise(pred);

    double ss_res = arma::accu(arma::square(yTrue - yPred));
    double mean_y = arma::mean(yTrue);
    double ss_tot = arma::accu(arma::square(yTrue - mean_y));

    if (ss_tot < 1e-14) return 0.0;
    return 1.0 - (ss_res / ss_tot);
}

/* ============================================================
 *                Normalization
 * ============================================================ */
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
            std::cout << "[Normalization] Per-variable min–max scaling\n";
            arma::mat full = arma::join_rows(train, test);
            mins.set_size(full.n_rows);
            maxs.set_size(full.n_rows);
            for (size_t r = 0; r < full.n_rows; ++r)
            {
                if (!normalizeOutputs && r >= inputSize)
                {
                    mins[r] = 0; maxs[r] = 1; continue;
                }
                double minVal = full.row(r).min();
                double maxVal = full.row(r).max();
                if (fabs(maxVal - minVal) < 1e-12)
                    maxVal = minVal + 1e-12;
                mins[r] = minVal;
                maxs[r] = maxVal;
                train.row(r) = (train.row(r) - minVal) / (maxVal - minVal);
                test.row(r)  = (test.row(r) - minVal) / (maxVal - minVal);
            }
            break;
        }

        case NormalizationType::MLpackMinMax:
        {
            std::cout << "[Normalization] MLpack MinMaxScaler (0–1)\n";
            // Compute and store real mins/maxs manually
            arma::mat full = arma::join_rows(train, test);
            mins = arma::min(full, 1).t();
            maxs = arma::max(full, 1).t();

            for (size_t r = 0; r < train.n_rows; ++r)
            {
                if (!normalizeOutputs && r >= inputSize)
                    continue;
                double minVal = mins[r];
                double maxVal = maxs[r];
                train.row(r) = (train.row(r) - minVal) / (maxVal - minVal);
                test.row(r)  = (test.row(r) - minVal) / (maxVal - minVal);
            }
            break;
        }

        case NormalizationType::ZScore:
        {
            std::cout << "[Normalization] Z-Score normalization\n";
            mins.set_size(train.n_rows);
            maxs.set_size(train.n_rows);
            for (size_t r = 0; r < train.n_rows; ++r)
            {
                if (!normalizeOutputs && r >= inputSize)
                {
                    mins[r] = 0; maxs[r] = 1; continue;
                }
                double mu = mean(train.row(r));
                double sd = stddev(train.row(r));
                if (sd < 1e-12) sd = 1.0;
                mins[r] = mu;   // store mean
                maxs[r] = sd;   // store stddev
                train.row(r) = (train.row(r) - mu) / sd;
                test.row(r)  = (test.row(r)  - mu) / sd;
            }
            break;
        }

        default:
            std::cout << "[Normalization] None\n";
            mins.reset(); maxs.reset();
    }
}

/* ============================================================
 *                Time-Series Cube Creation
 * ============================================================ */
void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          int rho,
                          int inputSize,
                          int outputSize,
                          bool IO)
{
    const size_t nSamples = dataset.n_cols - rho;
    X.set_size(inputSize, nSamples, rho);
    Y.set_size(outputSize, nSamples, rho);

    for (size_t i = 0; i < nSamples; ++i)
    {
        // Fill cube in old mlpack RNN format
        X.subcube(span(), span(i), span()) =
            dataset.submat(0, i, inputSize - 1, i + rho - 1);

        if (!IO)
        {
            Y.subcube(span(), span(i), span()) =
                dataset.submat(inputSize, i + 1,
                               inputSize + outputSize - 1, i + rho);
        }
        else
        {
            Y.subcube(span(), span(i), span()) =
                dataset.submat(inputSize - outputSize, i + 1,
                               inputSize - 1, i + rho);
        }
    }
}

/* ============================================================
 *                Shape Validation
 * ============================================================ */
void ValidateShapes(const arma::mat& data,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    int rho)
{
    bool ok = true;

    if (X.n_rows != inputSize)
    {
        qWarning() << "X.n_rows (" << X.n_rows
                   << ") != inputSize (" << inputSize << ")";
        ok = false;
    }
    if (Y.n_rows != outputSize)
    {
        qWarning() << "Y.n_rows (" << Y.n_rows
                   << ") != outputSize (" << outputSize << ")";
        ok = false;
    }
    if (X.n_slices != (size_t)rho || Y.n_slices != (size_t)rho)
    {
        qWarning() << "rho mismatch → expected" << rho
                   << "got X.n_slices=" << X.n_slices
                   << ", Y.n_slices=" << Y.n_slices;
        ok = false;
    }
    if (ok)
        qInfo().noquote() << "✅ Shape check passed:"
                          << "X" << X.n_rows << "×" << X.n_cols << "×" << X.n_slices
                          << ", Y" << Y.n_rows << "×" << Y.n_cols << "×" << Y.n_slices;
}

/* ============================================================
 *                Save Results (CSV)
 * ============================================================ */
void SaveResults(const std::string& filename,
                 arma::cube predictions,   // scaled predictions
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 arma::cube X,              // scaled inputs
                 arma::cube Y,              // scaled observed
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs,
                 NormalizationType normType)
{
    using arma::uword;

    // Use the same slice index for all (aligned)
    uword sliceIndex = std::min({X.n_slices, Y.n_slices, predictions.n_slices}) - 1;

    arma::mat x = X.slice(sliceIndex).t();        // inputs (scaled)
    arma::mat y = Y.slice(sliceIndex).t();        // observed (scaled)
    arma::mat p = predictions.slice(sliceIndex).t(); // predicted (scaled)

    // === Inverse-scale all three matrices before combining ===
    if (normalizeOutputs && mins.n_elem == maxs.n_elem && maxs.n_elem > 0)
    {
        // Inputs
        for (int j = 0; j < inputSize && j < (int)mins.n_elem; ++j)
        {
            double a = mins[j], b = maxs[j];
            if (normType == NormalizationType::ZScore)
                x.col(j) = x.col(j) * b + a;
            else if (fabs(b - a) > 1e-12)
                x.col(j) = x.col(j) * (b - a) + a;
        }

        // Observed + Predicted outputs
        for (int j = 0; j < outputSize && inputSize + j < (int)mins.n_elem; ++j)
        {
            double a = mins[inputSize + j], b = maxs[inputSize + j];
            if (normType == NormalizationType::ZScore)
            {
                y.col(j) = y.col(j) * b + a;
                p.col(j) = p.col(j) * b + a;
            }
            else if (fabs(b - a) > 1e-12)
            {
                y.col(j) = y.col(j) * (b - a) + a;
                p.col(j) = p.col(j) * (b - a) + a;
            }
        }
    }

    // === Keep all rows directly aligned (no time-step shift) ===
    uword N = std::min({x.n_rows, y.n_rows, p.n_rows});
    x = x.rows(0, N - 1);
    y = y.rows(0, N - 1);
    p = p.rows(0, N - 1);

    // === Combine → [inputs | observed | predicted] ===
    arma::mat result = x;
    result.insert_cols(result.n_cols, y.cols(y.n_cols - outputSize, y.n_cols - 1));
    result.insert_cols(result.n_cols, p.cols(p.n_cols - outputSize, p.n_cols - 1));

    // === Save ===
    result.save(filename, arma::csv_ascii);

    std::cout << "✅ Saved → " << filename
              << " (" << result.n_rows << "×" << result.n_cols
              << " = " << inputSize << " inputs + "
              << outputSize << " observed + "
              << outputSize << " predicted, all unscaled & time-synced)\n";
}


/* ============================================================
 *                Run Configuration Logging
 * ============================================================ */
void PrintRunConfig(bool ASM, bool IO, bool bTrain, bool bLoadAndTrain,
                    size_t inputSize, size_t outputSize, int rho,
                    double stepSize, size_t epochs, size_t batchSize,
                    int H1, int H2, int H3,
                    int mode, int kfoldMode, int KFOLDS,
                    double trainRatio, double testHoldout,
                    const std::string& dataFile,
                    const std::string& modelFile,
                    const std::string& predFile_Test,
                    const std::string& predFile_Train)
{
    qInfo().noquote() << "================ LSTM Wrapper Config ================";
    qInfo().noquote() << "Inputs:" << inputSize << "Outputs:" << outputSize << "Rho:" << rho;
    qInfo().noquote() << "Mode:" << mode << "KFoldMode:" << kfoldMode << "KFOLDS:" << KFOLDS;
    qInfo().noquote() << "TrainRatio:" << trainRatio << "TestHoldout:" << testHoldout;
    qInfo().noquote() << "Learning rate:" << stepSize << "Epochs:" << epochs << "Batch:" << batchSize;
    qInfo().noquote() << "Hidden layers:" << H1 << H2 << H3;
    qInfo().noquote() << "ASM:" << ASM << "IO:" << IO << "Train:" << bTrain << "LoadTrain:" << bLoadAndTrain;
    qInfo().noquote() << "====================================================";
}

void ValidateConfigOrWarn(int mode, int kfoldMode, int KFOLDS,
                          double trainRatio, double testHoldout)
{
    if (KFOLDS < 2 && mode != 0)
        qWarning() << "KFold < 2 → switching to single mode!";
    if (trainRatio <= 0.0 || trainRatio >= 1.0)
        qWarning() << "Train ratio out of bounds!";
    if (testHoldout < 0.0 || testHoldout > 0.9)
        qWarning() << "Holdout ratio is suspiciously large!";
}
