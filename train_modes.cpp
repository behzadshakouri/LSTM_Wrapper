/**
 * @file train_modes.cpp
 * @brief Implements training modes for LSTM Wrapper, including single run and
 *        K-Fold (Random, TimeSeries, FixedRatio) training with parametric optimizer
 *        and optional output normalization.
 *
 * Supports configurable LSTM architecture, optimizer parameters, and scaling
 * strategy for both input-only and full input–output normalization.
 *
 * Authors:
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include <pch.h>       // must come first
#include "helpers.h"   // NormalizationType, ApplyNormalization, helpers
#include "train_modes.h"

#include <ensmallen.hpp>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

/* ============================================================
 *                  Splitters (Random / TimeSeries / FixedRatio)
 * ============================================================ */

static std::pair<
    std::pair<arma::mat, arma::mat>,
    std::pair<arma::mat, arma::mat>
> KFoldSplit(const arma::mat& data,
             const arma::mat& labels,
             size_t k,
             size_t fold)
{
    if (k == 0 || fold >= k)
        throw std::invalid_argument("KFoldSplit: invalid fold or k.");

    const size_t n = data.n_cols;
    const size_t foldSize = n / k;
    const size_t start = fold * foldSize;
    const size_t end   = (fold == k - 1) ? n : start + foldSize;

    arma::uvec valIdx = arma::regspace<arma::uvec>(start, end - 1);
    arma::uvec mask   = arma::ones<arma::uvec>(n);
    mask(valIdx).zeros();
    arma::uvec trainIdx = arma::find(mask == 1);

    return std::make_pair(
        std::make_pair(data.cols(trainIdx).eval(), labels.cols(trainIdx).eval()),
        std::make_pair(data.cols(valIdx).eval(), labels.cols(valIdx).eval())
    );
}

static auto KFoldSplit_TimeSeries(const arma::mat& data,
                                  const arma::mat& labels,
                                  size_t k,
                                  size_t fold)
{
    if (k < 2) throw std::invalid_argument("KFoldSplit_TimeSeries: k must be >= 2.");
    if (fold >= k) throw std::invalid_argument("KFoldSplit_TimeSeries: fold out of range.");

    const size_t n = data.n_cols;
    const size_t foldSize = n / k;
    const size_t valStart = fold * foldSize;
    const size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;
    size_t trainEnd = (valStart == 0) ? foldSize : valStart;
    trainEnd = std::max<size_t>(2, trainEnd);

    return make_pair(
        make_pair(data.cols(0, trainEnd - 1), labels.cols(0, trainEnd - 1)),
        make_pair(data.cols(valStart, valEnd - 1), labels.cols(valStart, valEnd - 1))
    );
}

static auto KFoldSplit_FixedRatio(const arma::mat& data,
                                  const arma::mat& labels,
                                  size_t k,
                                  size_t fold,
                                  double trainRatio)
{
    if (k < 2) throw std::invalid_argument("KFoldSplit_FixedRatio: k must be >= 2.");
    if (fold >= k) throw std::invalid_argument("KFoldSplit_FixedRatio: fold out of range.");
    if (trainRatio <= 0.0 || trainRatio >= 1.0)
        throw std::invalid_argument("KFoldSplit_FixedRatio: invalid trainRatio.");

    const size_t n = data.n_cols;
    const size_t foldSize = n / k;
    const size_t valStart = fold * foldSize;
    const size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;
    const size_t trainEnd = std::max<size_t>(2, static_cast<size_t>(trainRatio * n));

    return make_pair(
        make_pair(data.cols(0, std::min(trainEnd, n) - 1),
                  labels.cols(0, std::min(trainEnd, n) - 1)),
        make_pair(data.cols(valStart, valEnd - 1),
                  labels.cols(valStart, valEnd - 1))
    );
}

/* ============================================================
 *                  Core Training Function (Extended)
 * ============================================================ */

void TrainCore(arma::mat& trainData,
                      arma::mat& testData,
                      const std::string& modelFile,
                      const std::string& predFile_Test,
                      const std::string& predFile_Train,
                      size_t inputSize, size_t outputSize,
                      int rho, double stepSize, size_t epochs,
                      size_t batchSize, bool IO,
                      bool bTrain, bool bLoadAndTrain,
                      int H1, int H2, int H3,
                      double beta1, double beta2,
                      double epsilon, double tolerance,
                      bool shuffle, bool normalizeOutputs,
                      NormalizationType normType)
{
    arma::rowvec mins, maxs; // used by SaveResults

    /* ------------------- Normalization ------------------- */
    ApplyNormalization(normType, trainData, testData, mins, maxs,
                       normalizeOutputs, inputSize);

    /* ------------------- Time-Series Cube Preparation ------------------- */
    arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
    arma::cube testX (inputSize, testData.n_cols  - rho, rho);
    arma::cube testY (outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData , testX , testY , rho, (int)inputSize, (int)outputSize, IO);

    ValidateShapes(trainData, trainX, trainY, inputSize, outputSize, rho);

    /* ------------------- Model Training ------------------- */
    if (bTrain || bLoadAndTrain)
    {
        RNN<MeanSquaredError, HeInitialization> model(rho);

        if (bLoadAndTrain)
        {
            cout << "Loading and further training model..." << endl;
            data::Load(modelFile, "LSTMMulti", model);
        }
        else
        {
            model.Add<Linear>(inputSize);
            model.Add<LSTM>(H1);
            model.Add<LSTM>(H2);
            model.Add<LSTM>(H3);
            model.Add<ReLU>();
            model.Add<Linear>(outputSize);
        }

        cout << "Adam optimizer settings: β1=" << beta1
             << ", β2=" << beta2
             << ", ε=" << epsilon
             << ", Tol=" << tolerance
             << ", Shuffle=" << std::boolalpha << shuffle << endl;

        const size_t maxIters = trainData.n_cols * epochs;
        Adam optimizer(stepSize, batchSize, beta1, beta2, epsilon,
                       maxIters, tolerance, shuffle);

        optimizer.Tolerance() = -1;

        cout << "Training ..." << endl;
        model.Train(trainX,
                    trainY,
                    optimizer,
                    PrintLoss(),
                    ProgressBar(),
                    EarlyStopAtMinLoss());

        cout << "Finished training.\nSaving Model" << endl;
        data::Save(modelFile, "LSTMMulti", model);
    }

    /* ------------------- Evaluation ------------------- */
    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predTest, predTrain;
    modelP.Predict(testX , predTest);
    modelP.Predict(trainX, predTrain);

    double mseTest  = ComputeMSE(predTest , testY );
    double mseTrain = ComputeMSE(predTrain, trainY);
    double r2Test   = ComputeR2(predTest , testY );
    double r2Train  = ComputeR2(predTrain, trainY);

    cout << "Test  MSE = " << mseTest  << ", R² = " << r2Test  << endl;
    cout << "Train MSE = " << mseTrain << ", R² = " << r2Train << endl;

    /* ------------------- Save Results ------------------- */
    SaveResults(predFile_Test , predTest , mins, maxs, testX ,
                (int)inputSize, (int)outputSize, IO, normalizeOutputs, normType);
    SaveResults(predFile_Train, predTrain, mins, maxs, trainX,
                (int)inputSize, (int)outputSize, IO, normalizeOutputs, normType);

}

/* ============================================================
 *                  Single Train/Test Mode
 * ============================================================ */

void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool /*ASM*/,
                 bool bTrain, bool bLoadAndTrain,
                 int H1, int H2, int H3,
                 double beta1, double beta2,
                 double epsilon, double tolerance,
                 bool shuffle, bool normalizeOutputs,
                 NormalizationType normType)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, ratio, false);

    TrainCore(trainData, testData, modelFile, predFile_Test, predFile_Train,
              inputSize, outputSize, rho, stepSize, epochs,
              batchSize, IO, bTrain, bLoadAndTrain,
              H1, H2, H3,
              beta1, beta2, epsilon, tolerance,
              shuffle, normalizeOutputs, normType);
}

/* ============================================================
 *                  K-Fold Mode (with selector)
 * ============================================================ */

static void TrainKFold_Impl(const std::string& dataFile,
                            const std::string& modelFile,
                            const std::string& predFile_Test,
                            const std::string& predFile_Train,
                            size_t inputSize, size_t outputSize,
                            int rho, int kfolds,
                            double stepSize, size_t epochs,
                            size_t batchSize, bool IO, bool /*ASM*/,
                            bool bTrain, bool bLoadAndTrain,
                            KFoldMode mode,
                            double trainRatioForFixed,
                            double holdoutRatioForTest,
                            int H1, int H2, int H3,
                            double beta1, double beta2,
                            double epsilon, double tolerance,
                            bool shuffle, bool normalizeOutputs,
                            NormalizationType normType)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    cout << "Loaded dataset: " << dataset.n_rows << "×" << dataset.n_cols << endl;

    arma::mat trainValData, testData;
    data::Split(dataset, trainValData, testData, holdoutRatioForTest, false);
    cout << "TrainVal: " << trainValData.n_cols << ", Test: " << testData.n_cols << endl;

    for (int fold = 0; fold < kfolds; ++fold)
    {
        cout << "\n===== Fold " << (fold + 1) << " / " << kfolds << " =====" << endl;

        pair<pair<arma::mat, arma::mat>, pair<arma::mat, arma::mat>> parts;
        if (mode == KFoldMode::Random)
            parts = KFoldSplit(trainValData, trainValData, kfolds, fold);
        else if (mode == KFoldMode::TimeSeries)
            parts = KFoldSplit_TimeSeries(trainValData, trainValData, kfolds, fold);
        else
            parts = KFoldSplit_FixedRatio(trainValData, trainValData, kfolds, fold, trainRatioForFixed);

        arma::mat trainData = parts.first.first;
        arma::mat valData   = parts.second.first;

        TrainCore(trainData, valData, modelFile, predFile_Test, predFile_Train,
                  inputSize, outputSize, rho, stepSize, epochs,
                  batchSize, IO, true, false,
                  H1, H2, H3,
                  beta1, beta2, epsilon, tolerance,
                  shuffle, normalizeOutputs, normType);
    }

    cout << "\n✅ TrainKFold completed successfully.\n";
}

/* ============================================================
 *                  Public Wrappers
 * ============================================================ */

void TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int kfolds,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain,
                int H1, int H2, int H3,
                double beta1, double beta2,
                double epsilon, double tolerance,
                bool shuffle, bool normalizeOutputs,
                NormalizationType normType)
{
    TrainKFold_Impl(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, kfolds,
                    stepSize, epochs, batchSize, IO, ASM,
                    bTrain, bLoadAndTrain,
                    KFoldMode::TimeSeries, 0.9, 0.3,
                    H1, H2, H3,
                    beta1, beta2, epsilon, tolerance,
                    shuffle, normalizeOutputs, normType);
}

void TrainKFold_WithMode(const std::string& dataFile,
                         const std::string& modelFile,
                         const std::string& predFile_Test,
                         const std::string& predFile_Train,
                         size_t inputSize, size_t outputSize,
                         int rho, int kfolds,
                         double stepSize, size_t epochs,
                         size_t batchSize, bool IO, bool ASM,
                         bool bTrain, bool bLoadAndTrain,
                         int modeInt, double trainRatio, double testHoldout,
                         int H1, int H2, int H3,
                         double beta1, double beta2,
                         double epsilon, double tolerance,
                         bool shuffle, bool normalizeOutputs,
                         NormalizationType normType)
{
    KFoldMode mode = (modeInt == 0) ? KFoldMode::Random :
                     (modeInt == 1) ? KFoldMode::TimeSeries :
                                       KFoldMode::FixedRatio;

    TrainKFold_Impl(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, kfolds,
                    stepSize, epochs, batchSize, IO, ASM,
                    bTrain, bLoadAndTrain,
                    mode, trainRatio, testHoldout,
                    H1, H2, H3,
                    beta1, beta2, epsilon, tolerance,
                    shuffle, normalizeOutputs, normType);
}
