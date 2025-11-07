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

using namespace std;
using namespace arma;

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
 *                Normalization (Per-Variable)
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
 *                Shape Validation
 * ============================================================ */
void ValidateShapes(const arma::mat& dataset,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    size_t rho)
{
    cout << "\n========== [ValidateShapes Debug Info] ==========\n";
    cout << "Dataset: " << dataset.n_rows << "x" << dataset.n_cols << endl;
    cout << "X cube : " << X.n_rows << "x" << X.n_cols << "x" << X.n_slices << endl;
    cout << "Y cube : " << Y.n_rows << "x" << Y.n_cols << "x" << Y.n_slices << endl;
    cout << "inputSize=" << inputSize << "  outputSize=" << outputSize
         << "  rho=" << rho << endl;
    cout << "==============================================\n";
}

/* ============================================================
 *                Time-Series Data Builder
 * ============================================================ */
void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          size_t rho,
                          int inputSize,
                          int outputSize,
                          bool IO)
{
    if (dataset.n_rows < (size_t)(inputSize + outputSize))
        throw invalid_argument("[CreateTimeSeriesData] Dataset rows < input+output.");

    const size_t nCols = dataset.n_cols;
    if (nCols <= rho)
        throw invalid_argument("[CreateTimeSeriesData] Dataset too short for rho.");

    X.set_size(inputSize, nCols - rho, rho);
    Y.set_size(outputSize, nCols - rho, rho);

    for (size_t i = 0; i < nCols - rho; ++i)
    {
        X.subcube(span(), span(i), span()) =
            dataset.submat(span(0, inputSize - 1),
                           span(i, i + rho - 1));

        if (!IO)
        {
            Y.subcube(span(), span(i), span()) =
                dataset.submat(span(inputSize, inputSize + outputSize - 1),
                               span(i + 1, i + rho));
        }
        else
        {
            Y.subcube(span(), span(i), span()) =
                dataset.submat(span(inputSize - outputSize, inputSize - 1),
                               span(i + 1, i + rho));
        }
    }

    if (rho == 1 && nCols > 2)
        cout << "[CreateTimeSeriesData] Alignment check: X(t)="
             << dataset(0, 0)
             << ", Y(t+1)=" << dataset(inputSize, 1) << endl;
}

/* ============================================================
 *                SaveResults (CSV Export)
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO,
                 const bool normalizeOutputs)
{
    cout << "\n========== [SaveResults Debug Info] ==========\n";
    arma::mat xSlice = IOData.slice(IOData.n_slices - 1);
    arma::mat yPred  = predictions.slice(predictions.n_slices - 1);

    arma::mat combined;
    if (!IO)
    {
        combined.set_size(inputSize + outputSize, xSlice.n_cols);
        combined.rows(0, inputSize - 1) = xSlice;
        combined.rows(inputSize, inputSize + outputSize - 1) = yPred;
    }
    else
    {
        combined.set_size(inputSize, xSlice.n_cols);
        combined.rows(0, inputSize - outputSize - 1) = xSlice.rows(0, inputSize - outputSize - 1);
        combined.rows(inputSize - outputSize, inputSize - 1) = yPred;
    }

    if (normalizeOutputs)
    {
        try
        {
            InverseMinMaxPerRow(combined, mins, maxs);
            cout << "[Scaler] Applied inverse-transform to outputs.\n";
        }
        catch (const exception& e)
        {
            cerr << "⚠️ Inverse failed: " << e.what() << "\n";
        }
    }
    else
    {
        cout << "[Scaler] Skipped inverse-transform (outputs unnormalized).\n";
    }

    if (!combined.save(filename, arma::csv_ascii))
        cerr << "❌ Failed to save " << filename << endl;
    else
        cout << "✅ Saved → " << filename << endl;

    cout << "==============================================\n";
}

/* ============================================================
 *                Config Utilities
 * ============================================================ */
string ModeName(int kfoldMode)
{
    switch (kfoldMode)
    {
        case 0: return "Random";
        case 1: return "TimeSeries";
        case 2: return "FixedRatio";
        default: return "Unknown";
    }
}

void ValidateConfigOrWarn(int mode, int kfoldMode, int& KFOLDS,
                          double& trainRatio, double& testHoldout)
{
    if (mode == 1)
    {
        if (KFOLDS < 2)
        {
            qWarning() << "[Config] KFOLDS < 2; forcing KFOLDS=2.";
            KFOLDS = 2;
        }

        if (kfoldMode == 2 && !(trainRatio > 0.0 && trainRatio < 1.0))
        {
            double suggested = static_cast<double>(KFOLDS - 1) / KFOLDS;
            qWarning() << "[Config] Invalid trainRatio; set to" << suggested;
            trainRatio = suggested;
        }
    }

    if (!(testHoldout > 0.0 && testHoldout < 1.0))
    {
        qWarning() << "[Config] Invalid testHoldout; forcing 0.3.";
        testHoldout = 0.3;
    }
}

void PrintRunConfig(bool ASM, bool IO, bool bTrain, bool bLoadAndTrain,
                    size_t inputSize, size_t outputSize, int rho,
                    double stepSize, size_t epochs, size_t batchSize,
                    int H1, int H2, int H3,
                    int mode, int kfoldMode, int KFOLDS,
                    double trainRatio, double testHoldout,
                    const string& dataFile,
                    const string& modelFile,
                    const string& predFile_Test,
                    const string& predFile_Train)
{
    ostringstream sig;
    sig << "LSTM[H1=" << H1 << ",H2=" << H2 << ",H3=" << H3 << "] "
        << "Inp=" << inputSize << " Out=" << outputSize
        << " rho=" << rho
        << " | LR=" << scientific << stepSize
        << " Ep=" << defaultfloat << epochs
        << " B=" << batchSize;

    qInfo() << "------------------- Run Configuration -------------------";
    qInfo() << "ASM=" << ASM << "| IO=" << IO
            << "| Train=" << bTrain << "| LoadAndTrain=" << bLoadAndTrain;
    qInfo() << "Signature:" << sig.str().c_str();
    qInfo() << "Mode=" << mode << "| KFoldMode=" << kfoldMode
            << "(" << ModeName(kfoldMode).c_str() << ") | KFOLDS=" << KFOLDS;
    qInfo() << "Ratios: train=" << trainRatio << "| holdout=" << testHoldout;
    qInfo() << "Files:";
    qInfo() << " dataFile=" << dataFile.c_str();
    qInfo() << " modelFile=" << modelFile.c_str();
    qInfo() << " predFile_Test=" << predFile_Test.c_str();
    qInfo() << " predFile_Train=" << predFile_Train.c_str();
    qInfo() << "----------------------------------------------------------";
}
