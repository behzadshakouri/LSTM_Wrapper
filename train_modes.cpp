#include <pch.h>       // must come first
#include "helpers.h"   // utilities: CreateTimeSeriesData, ComputeMSE, ComputeR2, SaveResults
#include "train_modes.h"

#include <ensmallen.hpp>
#include <iostream>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;


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
// TrainKFold (safe index handling + final full retrain + test evaluation)
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
    // ====================== LOAD DATA ======================
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    cout << "Loaded dataset: " << dataset.n_rows << "×" << dataset.n_cols << endl;

    // ====================== INITIAL SPLIT (TrainVal + Test) ======================
    const double RATIO = 0.3;
    arma::mat trainValData, testData;
    data::Split(dataset, trainValData, testData, RATIO, false);
    cout << "TrainVal: " << trainValData.n_cols << ", Test: " << testData.n_cols << endl;

    // Safe range lambda
    auto safe_range = [](arma::uword a, arma::uword b) -> arma::uvec {
        if (b < a) return arma::uvec();
        return arma::regspace<arma::uvec>(a, b);
    };

    // ====================== K-FOLD TRAINING ======================
    size_t n = trainValData.n_cols;
    size_t foldSize = n / kfolds;
    vector<double> mseValList, r2ValList;

    for (int fold = 0; fold < kfolds; ++fold)
    {
        cout << "\n===== Fold " << (fold + 1) << " / " << kfolds << " =====" << endl;

        size_t start = fold * foldSize;
        size_t end   = (fold == kfolds - 1) ? n : start + foldSize;

        arma::uvec valIdx   = safe_range((arma::uword)start, (arma::uword)(end - 1));
        arma::uvec leftIdx  = (start == 0) ? arma::uvec()
                                           : safe_range(0, (arma::uword)(start - 1));
        arma::uvec rightIdx = (end >= n) ? arma::uvec()
                                         : safe_range((arma::uword)end, (arma::uword)(n - 1));

        arma::uvec trainIdx;
        if (leftIdx.n_elem && rightIdx.n_elem)
            trainIdx = arma::join_cols(leftIdx, rightIdx);
        else if (leftIdx.n_elem)
            trainIdx = leftIdx;
        else
            trainIdx = rightIdx;

        if (trainIdx.n_elem <= (size_t)rho || valIdx.n_elem <= (size_t)rho)
        {
            cout << "[Skip] Fold " << (fold + 1)
                 << " too small after split (train=" << trainIdx.n_elem
                 << ", val=" << valIdx.n_elem << ", rho=" << rho << ").\n";
            continue;
        }

        arma::mat trainData = trainValData.cols(trainIdx);
        arma::mat valData   = trainValData.cols(valIdx);
        cout << "Train cols: " << trainData.n_cols
             << " | Val cols: " << valData.n_cols << endl;

        // Scale data
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

        // Model setup
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

        // Predict and evaluate validation performance
        arma::cube predVal;
        model.Predict(valX, predVal);

        double mseVal = ComputeMSE(predVal, valY);
        double r2Val = ComputeR2(predVal, valY);
        mseValList.push_back(mseVal);
        r2ValList.push_back(r2Val);

        cout << "Fold " << (fold + 1)
             << " | Val MSE=" << mseVal
             << ", R²=" << r2Val << endl;
    }

    // Average validation metrics
    double avgMseVal = arma::mean(arma::vec(mseValList));
    double avgR2Val  = arma::mean(arma::vec(r2ValList));
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
    CreateTimeSeriesData(testData, testX, testY, rho, (int)inputSize, (int)outputSize, IO);

    RNN<MeanSquaredError, HeInitialization> modelFinal(rho);
    const int H1 = 20, H2 = 16, H3 = 14;
    modelFinal.Add<Linear>(inputSize);
    modelFinal.Add<LSTM>(H1);
    modelFinal.Add<LSTM>(H2);
    modelFinal.Add<LSTM>(H3);
    modelFinal.Add<ReLU>();
    modelFinal.Add<Linear>(outputSize);

    ens::Adam optimizerFinal(stepSize, batchSize, 0.9, 0.999, 1e-8,
                             trainValData.n_cols * epochs, 1e-8, true);
    optimizerFinal.Tolerance() = -1;

    modelFinal.Train(trainX, trainY, optimizerFinal,
                     ens::PrintLoss(), ens::ProgressBar(), ens::EarlyStopAtMinLoss());

    // Predict on final sets
    arma::cube predTrainFull, predTestFull;
    modelFinal.Predict(trainX, predTrainFull);
    modelFinal.Predict(testX, predTestFull);

    double mseTrainFinal = ComputeMSE(predTrainFull, trainY);
    double mseTestFinal  = ComputeMSE(predTestFull, testY);
    double r2TrainFinal  = ComputeR2(predTrainFull, trainY);
    double r2TestFinal   = ComputeR2(predTestFull, testY);

    cout << "\n===== Final Full-Data Results =====" << endl;
    cout << "Train MSE = " << mseTrainFinal << ", R² = " << r2TrainFinal << endl;
    cout << "Test  MSE = " << mseTestFinal  << ", R² = " << r2TestFinal  << endl;

    // Save predictions
    predTrainFull.slice(predTrainFull.n_slices - 1)
        .save(predFile_Train, arma::csv_ascii);
    predTestFull.slice(predTestFull.n_slices - 1)
        .save(predFile_Test, arma::csv_ascii);
    cout << "[Saved] Final train/test predictions to CSV.\n";

    // Save model
    data::Save(modelFile, "LSTMMulti", modelFinal);
    cout << "[Saved] Final model: " << modelFile << endl;

    cout << "\n✅ TrainKFold completed successfully.\n";
}
