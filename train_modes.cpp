#include <pch.h>       // must come first
#include "helpers.h"   // includes CreateTimeSeriesData, ComputeMSE, ComputeR2, SaveResults
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


// ============================================================================================
// Splitters (FFN-style API; here labels are not used by LSTM, but we keep the signature)
// ============================================================================================

static
std::pair<std::pair<arma::mat, arma::mat>,
          std::pair<arma::mat, arma::mat>>
KFoldSplit(const arma::mat& data,
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

    arma::uvec valIdx  = arma::regspace<arma::uvec>(start, end - 1);
    arma::uvec mask    = arma::ones<arma::uvec>(n);
    mask(valIdx).zeros();
    arma::uvec trainIdx = arma::find(mask == 1);

    arma::mat trainData   = data.cols(trainIdx);
    arma::mat trainLabels = labels.cols(trainIdx);
    arma::mat validData   = data.cols(valIdx);
    arma::mat validLabels = labels.cols(valIdx);

    return {{trainData, trainLabels}, {validData, validLabels}};
}

static
std::pair<std::pair<arma::mat, arma::mat>,
          std::pair<arma::mat, arma::mat>>
KFoldSplit_TimeSeries(const arma::mat& data,
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

    // Train = prefix up to valStart (ensure at least 2 cols)
    size_t trainEnd = (valStart == 0) ? foldSize : valStart;
    if (trainEnd < 2) trainEnd = 2;

    arma::mat trainData   = data.cols(0, trainEnd - 1);
    arma::mat trainLabels = labels.cols(0, trainEnd - 1);
    arma::mat validData   = data.cols(valStart, valEnd - 1);
    arma::mat validLabels = labels.cols(valStart, valEnd - 1);

    return {{trainData, trainLabels}, {validData, validLabels}};
}

static
std::pair<std::pair<arma::mat, arma::mat>,
          std::pair<arma::mat, arma::mat>>
KFoldSplit_FixedRatio(const arma::mat& data,
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

    arma::mat trainData   = data.cols(0, std::min(trainEnd, n) - 1);
    arma::mat trainLabels = labels.cols(0, std::min(trainEnd, n) - 1);
    arma::mat validData   = data.cols(valStart, valEnd - 1);
    arma::mat validLabels = labels.cols(valStart, valEnd - 1);

    return {{trainData, trainLabels}, {validData, validLabels}};
}


// ============================================================================================
// CORE TRAINING FUNCTION (shared by TrainSingle and TrainKFold final retrain)
// ============================================================================================

static void TrainCore(arma::mat& trainData,
                      arma::mat& testData,
                      const std::string& modelFile,
                      const std::string& predFile_Test,
                      const std::string& predFile_Train,
                      size_t inputSize, size_t outputSize,
                      int rho, double stepSize, size_t epochs,
                      size_t batchSize, bool IO,
                      bool bTrain, bool bLoadAndTrain)
{
    // === Scaling ===
    MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    // === Build time-series tensors ===
    arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
    arma::cube testX (inputSize, testData.n_cols  - rho, rho);
    arma::cube testY (outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData , testX , testY , rho, (int)inputSize, (int)outputSize, IO);

    if (bTrain || bLoadAndTrain)
    {
        RNN<MeanSquaredError, HeInitialization> model(rho);
        const int H1 = 20, H2 = 16, H3 = 14;

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

        Adam optimizer(
            stepSize, batchSize, 0.9, 0.999, 1e-8,
            trainData.n_cols * epochs, 1e-8, true);

        optimizer.Tolerance() = -1;

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

    arma::cube testIO  = testX;
    arma::cube trainIO = trainX;
    if (!IO)
    {
        testIO.insert_rows (testX.n_rows , testY );
        trainIO.insert_rows(trainX.n_rows, trainY);
    }

    SaveResults(predFile_Test , predTest , scale, testIO , (int)inputSize, (int)outputSize, IO);
    SaveResults(predFile_Train, predTrain, scale, trainIO, (int)inputSize, (int)outputSize, IO);
}


// ============================================================================================
// TrainSingle (identical to old working version)
// ============================================================================================

void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool /*ASM*/,
                 bool bTrain, bool bLoadAndTrain)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, ratio, false);

    TrainCore(trainData, testData, modelFile, predFile_Test, predFile_Train,
              inputSize, outputSize, rho, stepSize, epochs,
              batchSize, IO, bTrain, bLoadAndTrain);
}


// ============================================================================================
// TrainKFold OVERLOAD with mode + trainRatio (for FixedRatio)
// ============================================================================================

static
void TrainKFold_Impl(const std::string& dataFile,
                     const std::string& modelFile,
                     const std::string& predFile_Test,
                     const std::string& predFile_Train,
                     size_t inputSize, size_t outputSize,
                     int rho, int kfolds,
                     double stepSize, size_t epochs,
                     size_t batchSize, bool IO, bool /*ASM*/,
                     bool bTrain, bool bLoadAndTrain,
                     KFoldMode mode,
                     double trainRatioForFixed = 0.9,   // used only in FixedRatio
                     double holdoutRatioForTest = 0.3)  // same test split as TrainSingle
{
    // ====================== LOAD DATA ======================
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    cout << "Loaded dataset: " << dataset.n_rows << "×" << dataset.n_cols << endl;

    // ====================== INITIAL SPLIT (TrainVal + Test) ======================
    arma::mat trainValData, testData;
    data::Split(dataset, trainValData, testData, holdoutRatioForTest, false);
    cout << "TrainVal: " << trainValData.n_cols << ", Test: " << testData.n_cols << endl;

    // ====================== K-FOLD TRAINING ======================
    const size_t n = trainValData.n_cols;
    if (kfolds < 2) throw std::invalid_argument("TrainKFold: kfolds must be >= 2.");
    const size_t foldSize = std::max<size_t>(1, n / (size_t)kfolds);

    std::vector<double> mseValList, r2ValList;

    for (int fold = 0; fold < kfolds; ++fold)
    {
        cout << "\n===== Fold " << (fold + 1) << " / " << kfolds << " =====" << endl;

        // We pass trainValData as both data and labels (labels unused for LSTM splitting).
        pair<pair<arma::mat, arma::mat>, pair<arma::mat, arma::mat>> parts;

        switch (mode)
        {
            case KFoldMode::Random:
                parts = KFoldSplit(trainValData, trainValData, (size_t)kfolds, (size_t)fold);
                break;

            case KFoldMode::TimeSeries:
                parts = KFoldSplit_TimeSeries(trainValData, trainValData, (size_t)kfolds, (size_t)fold);
                break;

            case KFoldMode::FixedRatio:
                parts = KFoldSplit_FixedRatio(trainValData, trainValData, (size_t)kfolds, (size_t)fold, trainRatioForFixed);
                break;
        }

        arma::mat trainData = parts.first.first;   // (data portion)
        arma::mat valData   = parts.second.first;  // (data portion)

        cout << "Train cols: " << trainData.n_cols
             << " | Val cols: " << valData.n_cols << endl;

        // Ensure we have enough columns to build sequences
        if (trainData.n_cols <= (size_t)rho || valData.n_cols <= (size_t)rho)
        {
            cout << "[Skip] Fold " << (fold + 1)
                 << " too small for rho=" << rho << ".\n";
            continue;
        }

        // Scale train/val
        data::MinMaxScaler scale;
        scale.Fit(trainData);
        scale.Transform(trainData, trainData);
        scale.Transform(valData, valData);

        // Prepare cubes
        arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
        arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
        arma::cube valX(inputSize, valData.n_cols - rho, rho);
        arma::cube valY(outputSize, valData.n_cols - rho, rho);

        CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
        CreateTimeSeriesData(valData, valX, valY, rho, (int)inputSize, (int)outputSize, IO);

        // Model
        RNN<MeanSquaredError, HeInitialization> model(rho);
        const int H1 = 20, H2 = 16, H3 = 14;
        model.Add<Linear>(inputSize);
        model.Add<LSTM>(H1);
        model.Add<LSTM>(H2);
        model.Add<LSTM>(H3);
        model.Add<ReLU>();
        model.Add<Linear>(outputSize);

        ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, 1e-8,
                            trainData.n_cols * epochs, 1e-8, true);
        optimizer.Tolerance() = -1;

        model.Train(trainX, trainY, optimizer,
                    ens::PrintLoss(), ens::ProgressBar(), ens::EarlyStopAtMinLoss());

        // Predict & metrics (validation)
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

    // Average validation metrics
    const double avgMseVal = (mseValList.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                 : arma::mean(arma::vec(mseValList)));
    const double avgR2Val  = (r2ValList.empty()  ? std::numeric_limits<double>::quiet_NaN()
                                                 : arma::mean(arma::vec(r2ValList)));
    cout << "\n===== Average Validation Results =====" << endl;
    cout << "Val MSE = " << avgMseVal << ", R² = " << avgR2Val << endl;


    // ====================== FINAL FULL RETRAIN =========================
    cout << "\nRetraining final model on full (train+val) dataset...\n";

    data::MinMaxScaler fullScale;
    fullScale.Fit(trainValData);
    fullScale.Transform(trainValData, trainValData);
    fullScale.Transform(testData, testData);

    arma::cube trainX(inputSize, trainValData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainValData.n_cols - rho, rho);
    arma::cube testX(inputSize, testData.n_cols - rho, rho);
    arma::cube testY(outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainValData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData,     testX,  testY,  rho, (int)inputSize, (int)outputSize, IO);

    RNN<MeanSquaredError, HeInitialization> modelFinal(rho);
    {
        const int H1 = 20, H2 = 16, H3 = 14;
        modelFinal.Add<Linear>(inputSize);
        modelFinal.Add<LSTM>(H1);
        modelFinal.Add<LSTM>(H2);
        modelFinal.Add<LSTM>(H3);
        modelFinal.Add<ReLU>();
        modelFinal.Add<Linear>(outputSize);
    }

    ens::Adam optimizerFinal(stepSize, batchSize, 0.9, 0.999, 1e-8,
                             trainValData.n_cols * epochs, 1e-8, true);
    optimizerFinal.Tolerance() = -1;

    modelFinal.Train(trainX, trainY, optimizerFinal,
                     ens::PrintLoss(), ens::ProgressBar(), ens::EarlyStopAtMinLoss());

    // Predict on final sets
    arma::cube predTrainFull, predTestFull;
    modelFinal.Predict(trainX, predTrainFull);
    modelFinal.Predict(testX,  predTestFull);

    double mseTrainFinal = ComputeMSE(predTrainFull, trainY);
    double mseTestFinal  = ComputeMSE(predTestFull,  testY);
    double r2TrainFinal  = ComputeR2(predTrainFull,  trainY);
    double r2TestFinal   = ComputeR2(predTestFull,   testY);

    cout << "\n===== Final Full-Data Results =====" << endl;
    cout << "Train MSE = " << mseTrainFinal << ", R² = " << r2TrainFinal << endl;
    cout << "Test  MSE = " << mseTestFinal  << ", R² = " << r2TestFinal  << endl;

    // Save predictions via SaveResults (keeps scaling/columns aligned)
    arma::cube trainIO = trainX, testIO = testX;
    if (!IO)
    {
        trainIO.insert_rows(trainX.n_rows, trainY);
        testIO.insert_rows(testX.n_rows,  testY);
    }

    SaveResults(predFile_Train, predTrainFull, fullScale, trainIO,
                (int)inputSize, (int)outputSize, IO);
    SaveResults(predFile_Test,  predTestFull,  fullScale, testIO,
                (int)inputSize, (int)outputSize, IO);

    // Save model
    data::Save(modelFile, "LSTMMulti", modelFinal);
    cout << "[Saved] Final model: " << modelFile << endl;

    cout << "\n✅ TrainKFold completed successfully.\n";
}


// ============================================================================================
// Public TrainKFold: legacy signature (defaults to TimeSeries mode)
// ============================================================================================

void TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int kfolds,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain)
{
    // Default behavior = TimeSeries mode, 30% test holdout (same as TrainSingle)
    TrainKFold_Impl(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, kfolds,
                    stepSize, epochs, batchSize, IO, ASM,
                    bTrain, bLoadAndTrain,
                    KFoldMode::TimeSeries, /*trainRatioForFixed*/0.9, /*testHoldout*/0.3);
}


// ============================================================================================
// Optional: extended TrainKFold with explicit mode/ratio (use from your caller if needed)
// ============================================================================================

void TrainKFold_WithMode(const std::string& dataFile,
                         const std::string& modelFile,
                         const std::string& predFile_Test,
                         const std::string& predFile_Train,
                         size_t inputSize, size_t outputSize,
                         int rho, int kfolds,
                         double stepSize, size_t epochs,
                         size_t batchSize, bool IO, bool ASM,
                         bool bTrain, bool bLoadAndTrain,
                         int modeInt,          // 0=Random, 1=TimeSeries, 2=FixedRatio
                         double trainRatio,    // used only for FixedRatio
                         double testHoldout)   // e.g. 0.3 to mirror TrainSingle
{
    KFoldMode mode = KFoldMode::TimeSeries;
    if (modeInt == 0) mode = KFoldMode::Random;
    else if (modeInt == 1) mode = KFoldMode::TimeSeries;
    else if (modeInt == 2) mode = KFoldMode::FixedRatio;

    TrainKFold_Impl(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, kfolds,
                    stepSize, epochs, batchSize, IO, ASM,
                    bTrain, bLoadAndTrain,
                    mode, trainRatio, testHoldout);
}
