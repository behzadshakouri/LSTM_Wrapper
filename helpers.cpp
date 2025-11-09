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
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return accu(square(pred - Y)) / Y.n_elem;
}

double ComputeR2(arma::cube& pred, arma::cube& Y)
{
    arma::vec yTrue = vectorise(Y);
    arma::vec yPred = vectorise(pred);

    double ss_res = accu(square(yTrue - yPred));
    double mean_y = mean(yTrue);
    double ss_tot = accu(square(yTrue - mean_y));

    return (ss_tot == 0.0) ? 0.0 : 1.0 - (ss_res / ss_tot);
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
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& X,
                 const arma::cube& Y,
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs,
                 NormalizationType normType)
{
    // Flatten last slices
    arma::mat flat = X.slice(X.n_slices - 1).t();  // Input columns
    arma::mat obs  = Y.slice(Y.n_slices - 1).t();  // Observed outputs
    arma::mat pred = predictions.slice(predictions.n_slices - 1).t(); // Predicted outputs

    // Combine: inputs | observed | predicted
    flat.insert_cols(flat.n_cols, obs.cols(obs.n_cols - outputSize, obs.n_cols - 1));
    flat.insert_cols(flat.n_cols, pred.cols(pred.n_cols - outputSize, pred.n_cols - 1));

    arma::mat result = flat;

    // Optional inverse scaling
    if (normalizeOutputs && mins.n_elem == maxs.n_elem && maxs.n_elem > 0)
    {
        switch (normType)
        {
            case NormalizationType::PerVariable:
            case NormalizationType::MLpackMinMax:
                std::cout << "[SaveResults] Inverse Min–Max scaling\n";
                for (size_t c = 0; c < result.n_cols && c < mins.n_elem; ++c)
                {
                    double minVal = mins[c];
                    double maxVal = maxs[c];
                    if (fabs(maxVal - minVal) < 1e-12) continue;
                    result.col(c) = result.col(c) * (maxVal - minVal) + minVal;
                }
                break;

            case NormalizationType::ZScore:
                std::cout << "[SaveResults] Inverse Z-Score scaling\n";
                for (size_t c = 0; c < result.n_cols && c < mins.n_elem; ++c)
                {
                    double meanVal = mins[c];
                    double stdVal  = maxs[c];
                    result.col(c) = result.col(c) * stdVal + meanVal;
                }
                break;

            default:
                std::cout << "[SaveResults] No inverse scaling\n";
                break;
        }
    }
    else
        std::cout << "[SaveResults] Skipped inverse scaling (none or disabled)\n";

    // Save CSV
    result.save(filename, arma::csv_ascii);
    std::cout << "✅ Saved predictions → " << filename
              << " (" << result.n_cols << " cols: inputs + observed + predicted)\n";
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
