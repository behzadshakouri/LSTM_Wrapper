/**
 * @file helpers.cpp
 * @brief Implementation of LSTM Wrapper utility functions.
 *
 * Includes shape validation, metric evaluation, inverse scaling,
 * configuration logging, and K-Fold validation utilities.
 *
 * @authors
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include "helpers.h"
#include <iostream>
#include <sstream>
#include <QDebug>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;

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
/**
 * @brief Create LSTM-ready time-series input/output cubes.
 *
 * Generates sliding windows of length @p rho for each input feature
 * and aligns corresponding target output sequences.
 *
 * @param dataset Input dataset matrix (features × timesteps)
 * @param X Output cube for model input sequences
 * @param Y Output cube for model target sequences
 * @param rho Sequence length (number of time steps per window)
 * @param inputSize Number of input features
 * @param outputSize Number of output features
 * @param IO If true, outputs are part of input block layout
 */
void CreateTimeSeriesData(const arma::mat& dataset,
                                 arma::cube& X,
                                 arma::cube& Y,
                                 const size_t rho,
                                 const int inputSize,
                                 const int outputSize,
                                 const bool IO)
{
    if (dataset.n_rows < (size_t)(inputSize + outputSize))
        throw std::invalid_argument("[CreateTimeSeriesData] Dataset has fewer rows than input+output size.");

    const size_t nCols = dataset.n_cols;
    if (nCols <= rho)
        throw std::invalid_argument("[CreateTimeSeriesData] Dataset too short for given rho.");

    X.set_size(inputSize, nCols - rho, rho);
    Y.set_size(outputSize, nCols - rho, rho);

    for (size_t i = 0; i < nCols - rho; ++i)
    {
        // Input window
        X.subcube(arma::span(), arma::span(i), arma::span()) =
            dataset.submat(arma::span(0, inputSize - 1),
                           arma::span(i, i + rho - 1));

        if (!IO)
        {
            // ✅ One-step-ahead target (fix: -1 inside the argument)
            Y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize, inputSize + outputSize - 1),
                               arma::span(i + 1, i + rho));
        }
        else
        {
            Y.subcube(arma::span(), arma::span(i), arma::span()) =
                dataset.submat(arma::span(inputSize - outputSize, inputSize - 1),
                               arma::span(i + 1, i + rho));
        }
    }

    // Optional alignment check for rho=1
    if (rho == 1 && nCols > 2)
    {
        std::cout << "[CreateTimeSeriesData] Alignment check (first sample): "
                  << "X(t)=" << dataset(0, 0)
                  << ", Y(t+1)=" << dataset(inputSize, 1)
                  << std::endl;
    }
}

/* ============================================================
 *                Metric Computation
 * ============================================================ */
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return mlpack::metric::SquaredEuclideanDistance::Evaluate(pred, Y) / Y.n_elem;
}

double ComputeR2(arma::cube& pred, arma::cube& Y)
{
    arma::vec yTrue = arma::vectorise(Y);
    arma::vec yPred = arma::vectorise(pred);

    const double ss_res = arma::accu(arma::square(yTrue - yPred));
    const double mean_y = arma::mean(yTrue);
    const double ss_tot = arma::accu(arma::square(yTrue - mean_y));

    return (ss_tot == 0.0) ? 0.0 : 1.0 - (ss_res / ss_tot);
}

/* ============================================================
 *                SaveResults (CSV Export + Safe Scaling)
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO,
                 const bool normalizeOutputs)
{
    cout << "\n========== [SaveResults Debug Info] ==========\n";

    arma::mat xSlice = IOData.slice(IOData.n_slices - 1);
    arma::mat yPred  = predictions.slice(predictions.n_slices - 1);

    cout << "xSlice size: " << xSlice.n_rows << "x" << xSlice.n_cols << endl;
    cout << "yPred  size: " << yPred.n_rows  << "x" << yPred.n_cols  << endl;

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

    cout << "Combined pre-inverse size: " << combined.n_rows
         << "x" << combined.n_cols << endl;

    bool inverseOK = false;
    if (normalizeOutputs)
    {
        try
        {
            scale.InverseTransform(combined, combined);
            inverseOK = true;
        }
        catch (const std::exception& e)
        {
            cerr << "⚠️  InverseTransform failed (" << e.what()
                 << "). Saving scaled values instead.\n";
        }
    }
    else
    {
        cout << "[Scaler] Skipped inverse-transform (outputs unnormalized)." << endl;
    }

    // Always attempt to save the combined matrix
    if (!combined.save(filename, arma::csv_ascii))
        cerr << "❌ Error: Failed to save results to " << filename << endl;
    else
        cout << (inverseOK ? "✅ Saved predictions (inverse-transformed) → "
                           : "✅ Saved predictions (scaled/raw) → ")
             << filename << endl;

    cout << "==============================================\n";
}

/* ============================================================
 *                Mode & Config Utilities
 * ============================================================ */
std::string ModeName(int kfoldMode)
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
            const double suggested = static_cast<double>(KFOLDS - 1) / KFOLDS;
            qWarning() << "[Config] Invalid trainRatio. Setting to" << suggested;
            trainRatio = suggested;
        }
    }

    if (!(testHoldout > 0.0 && testHoldout < 1.0))
    {
        qWarning() << "[Config] testHoldout invalid. Forcing 0.3.";
        testHoldout = 0.3;
    }
}

/* ============================================================
 *                Run Configuration Logger
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
    std::ostringstream sig;
    sig << "LSTM[H1=" << H1 << ",H2=" << H2 << ",H3=" << H3 << "] "
        << "Inp=" << inputSize << " Out=" << outputSize
        << " rho=" << rho
        << " | LR=" << std::scientific << stepSize
        << " Ep=" << std::defaultfloat << epochs
        << " B=" << batchSize;

    qInfo() << "------------------- Run Configuration -------------------";
    qInfo() << "ASM=" << ASM << " | IO=" << IO
            << " | Train=" << bTrain << " | LoadAndTrain=" << bLoadAndTrain;
    qInfo() << "Signature:" << sig.str().c_str();
    qInfo() << "Mode=" << mode << " (0=Single,1=KFold)"
            << " | KFoldMode=" << kfoldMode << "(" << ModeName(kfoldMode).c_str() << ")"
            << " | KFOLDS=" << KFOLDS;
    qInfo() << "Ratios: trainRatio=" << trainRatio << " | testHoldout=" << testHoldout;
    qInfo() << "Files:";
    qInfo() << "  dataFile      =" << dataFile.c_str();
    qInfo() << "  modelFile     =" << modelFile.c_str();
    qInfo() << "  predFile_Test =" << predFile_Test.c_str();
    qInfo() << "  predFile_Train=" << predFile_Train.c_str();
    qInfo() << "----------------------------------------------------------";
}
