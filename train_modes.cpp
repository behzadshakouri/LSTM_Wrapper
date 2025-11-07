/**
 * @file train_modes.cpp
 * @brief Implements training modes for LSTM Wrapper, including single run and
 *        K-Fold (Random, TimeSeries, FixedRatio) training with parametric  optimizer.
 *
 * Provides reusable core routines for model setup, normalization, time-series
 * cube generation, training, and validation. Supports configurable number of
 * folds and modes for robust performance analysis on ASM-type datasets.
 *
 * @authors
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include <pch.h>       // must come first
#include "helpers.h"
#include "train_modes.h"

#include <ensmallen.hpp>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
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
 *                  Core Training Function
 * ============================================================ */

static void TrainCore(arma::mat& trainData,
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
                      double epsilon, double tolerance, bool shuffle)
{
    MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
    arma::cube testX (inputSize, testData.n_cols  - rho, rho);
    arma::cube testY (outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData , testX , testY , rho, (int)inputSize, (int)outputSize, IO);

    ValidateShapes(trainData, trainX, trainY, inputSize, outputSize, rho);

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

        Adam optimizer(stepSize, batchSize, beta1, beta2, epsilon,
                       trainData.n_cols * epochs, tolerance, shuffle);
        optimizer.Tolerance() = tolerance;

        cout << "Training ..." << endl;
        model.Train(trainX, trainY, optimizer,
                    PrintLoss(), ProgressBar(), EarlyStopAtMinLoss());
        cout << "Finished training.\nSaving Model" << endl;
        data::Save(modelFile, "LSTMMulti", model);
    }

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

    SaveResults(predFile_Test , predTest , scale, testX , (int)inputSize, (int)outputSize, IO);
    SaveResults(predFile_Train, predTrain, scale, trainX, (int)inputSize, (int)outputSize, IO);
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
                 double epsilon, double tolerance, bool shuffle)
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
              beta1, beta2, epsilon, tolerance, shuffle);
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
                            double epsilon, double tolerance, bool shuffle)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    cout << "Loaded dataset: " << dataset.n_rows << "×" << dataset.n_cols << endl;

    arma::mat trainValData, testData;
    data::Split(dataset, trainValData, testData, holdoutRatioForTest, false);
    cout << "TrainVal: " << trainValData.n_cols << ", Test: " << testData.n_cols << endl;

    vector<double> mseValList, r2ValList;

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

        MinMaxScaler scale;
        scale.Fit(trainData);
        scale.Transform(trainData, trainData);
        scale.Transform(valData, valData);

        arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
        arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
        arma::cube valX(inputSize, valData.n_cols - rho, rho);
        arma::cube valY(outputSize, valData.n_cols - rho, rho);

        CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
        CreateTimeSeriesData(valData, valX, valY, rho, (int)inputSize, (int)outputSize, IO);

        RNN<MeanSquaredError, HeInitialization> model(rho);
        model.Add<Linear>(inputSize);
        model.Add<LSTM>(H1);
        model.Add<LSTM>(H2);
        model.Add<LSTM>(H3);
        model.Add<ReLU>();
        model.Add<Linear>(outputSize);

        Adam optimizer(stepSize, batchSize, beta1, beta2, epsilon,
                       trainData.n_cols * epochs, tolerance, shuffle);
        optimizer.Tolerance() = tolerance;

        model.Train(trainX, trainY, optimizer,
                    PrintLoss(), ProgressBar(), EarlyStopAtMinLoss());

        arma::cube predVal;
        model.Predict(valX, predVal);

        double mseVal = ComputeMSE(predVal, valY);
        double r2Val  = ComputeR2(predVal, valY);
        mseValList.push_back(mseVal);
        r2ValList.push_back(r2Val);

        cout << "Fold " << (fold + 1)
             << " | Val MSE=" << mseVal
             << ", R²=" << r2Val << endl;
    }

    cout << "\n===== Average Validation Results =====" << endl;
    cout << "Val MSE = " << arma::mean(arma::vec(mseValList))
         << ", R² = " << arma::mean(arma::vec(r2ValList)) << endl;

    cout << "\nRetraining final model on full (train+val) dataset...\n";

    MinMaxScaler fullScale;
    fullScale.Fit(trainValData);
    fullScale.Transform(trainValData, trainValData);
    fullScale.Transform(testData, testData);

    arma::cube trainX(inputSize, trainValData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainValData.n_cols - rho, rho);
    arma::cube testX(inputSize, testData.n_cols - rho, rho);
    arma::cube testY(outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainValData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData, testX, testY, rho, (int)inputSize, (int)outputSize, IO);

    RNN<MeanSquaredError, HeInitialization> modelFinal(rho);
    modelFinal.Add<Linear>(inputSize);
    modelFinal.Add<LSTM>(H1);
    modelFinal.Add<LSTM>(H2);
    modelFinal.Add<LSTM>(H3);
    modelFinal.Add<ReLU>();
    modelFinal.Add<Linear>(outputSize);

    Adam optimizerFinal(stepSize, batchSize, beta1, beta2, epsilon,
                        trainValData.n_cols * epochs, tolerance, shuffle);
    optimizerFinal.Tolerance() = tolerance;

    modelFinal.Train(trainX, trainY, optimizerFinal,
                     PrintLoss(), ProgressBar(), EarlyStopAtMinLoss());

    arma::cube predTrainFull, predTestFull;
    modelFinal.Predict(trainX, predTrainFull);
    modelFinal.Predict(testX, predTestFull);

    cout << "\n===== Final Full-Data Results =====" << endl;
    cout << "Train MSE = " << ComputeMSE(predTrainFull, trainY)
         << ", R² = " << ComputeR2(predTrainFull, trainY) << endl;
    cout << "Test  MSE = " << ComputeMSE(predTestFull, testY)
         << ", R² = " << ComputeR2(predTestFull, testY) << endl;

    SaveResults(predFile_Train, predTrainFull, fullScale, trainX, (int)inputSize, (int)outputSize, IO);
    SaveResults(predFile_Test, predTestFull, fullScale, testX, (int)inputSize, (int)outputSize, IO);

    data::Save(modelFile, "LSTMMulti", modelFinal);
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
                double epsilon, double tolerance, bool shuffle)
{
    TrainKFold_Impl(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, kfolds,
                    stepSize, epochs, batchSize, IO, ASM,
                    bTrain, bLoadAndTrain,
                    KFoldMode::TimeSeries, 0.9, 0.3,
                    H1, H2, H3,
                    beta1, beta2, epsilon, tolerance, shuffle);
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
                         double epsilon, double tolerance, bool shuffle)
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
                    beta1, beta2, epsilon, tolerance, shuffle);
}
