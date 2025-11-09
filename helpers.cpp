/**
 * @file helpers.cpp
 * @brief Implementation of utility functions for LSTM Wrapper.
 */

#include "pch.h"
#include "helpers.h"
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>

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
            cout << "[Normalization] Per-variable min–max scaling\n";
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
            cout << "[Normalization] MLpack MinMaxScaler (0–1)\n";
            MinMaxScaler scaler;
            scaler.Fit(train);
            scaler.Transform(train, train);
            scaler.Transform(test, test);
            mins.zeros(train.n_rows);
            maxs.ones(train.n_rows);
            break;
        }

        case NormalizationType::ZScore:
        {
            cout << "[Normalization] Z-Score normalization\n";
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
                mins[r] = mu; maxs[r] = sd;
                train.row(r) = (train.row(r) - mu) / sd;
                test.row(r)  = (test.row(r)  - mu) / sd;
            }
            break;
        }

        default:
            cout << "[Normalization] None\n";
            mins.reset();
            maxs.reset();
    }
}

/* ============================================================
 *                Validation & Logging Stubs
 * ============================================================ */
void ValidateShapes(const arma::mat& data, const arma::cube& X,
                    const arma::cube& Y, size_t inputSize,
                    size_t outputSize, int rho)
{
    if (data.n_rows < inputSize + outputSize)
        qWarning() << "Data has fewer rows than expected!";
    if (X.n_slices != (size_t)rho)
        qWarning() << "Unexpected number of time slices in X";
}

void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          int rho,
                          int inputSize,
                          int outputSize,
                          bool IO)
{
    for (size_t i = 0; i < dataset.n_cols - rho; ++i)
    {
        X.slice(i) = dataset.submat(0, i, inputSize - 1, i + rho - 1);
        if (!IO)
            Y.slice(i) = dataset.submat(inputSize, i + 1,
                                        inputSize + outputSize - 1, i + rho);
        else
            Y.slice(i) = dataset.submat(inputSize - outputSize, i + 1,
                                        inputSize - 1, i + rho);
    }
}

void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& X,
                 int inputSize,
                 int outputSize,
                 bool IO,
                 bool normalizeOutputs)
{
    arma::mat out = predictions.slice(predictions.n_slices - 1);
    out.save(filename, arma::csv_ascii);
    cout << "✅ Saved predictions → " << filename << endl;
}

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
