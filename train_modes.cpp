#include <pch.h>       // ✅ must come first
#include "helpers.h"   // now all mlpack headers use the defined macro
#include "train_modes.h"

#include <ensmallen.hpp>
#include <iostream>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;


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

// ---------------------------------------------------------
// TrainSingle (same as before)
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// TrainKFold (new addition)
// ---------------------------------------------------------
/**
 * TrainKFold: Perform K-fold cross-validation, then retrain on full dataset
 * and evaluate on test set.
 *
 * @param dataFile Path to dataset file (e.g., observedoutput_t10&11_NO.txt)
 * @param modelFile Path to model save file
 * @param predFile_Test Path for test predictions CSV
 * @param predFile_Train Path for train predictions CSV
 * @param inputSize Number of input features
 * @param outputSize Number of output targets
 * @param rho Time steps (sequence length)
 * @param kfolds Number of folds for cross-validation
 * @param stepSize Learning rate
 * @param epochs Number of epochs
 * @param batchSize Batch size
 * @param IO Whether outputs are used as inputs
 * @param ASM Whether ASM mode is used (for logging)
 * @param bTrain Whether to train
 * @param bLoadAndTrain Whether to load a pre-trained model and continue training
 */
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

    // ====================== K-FOLD TRAINING ======================
    size_t n = trainValData.n_cols;
    size_t foldSize = n / kfolds;
    vector<double> mseValList, r2ValList;

    for (int fold = 0; fold < kfolds; ++fold)
    {
        cout << "\n===== Fold " << (fold + 1) << " / " << kfolds << " =====" << endl;

        // Define validation range
        size_t start = fold * foldSize;
        size_t end = (fold == kfolds - 1) ? n : start + foldSize;

        arma::uvec valIdx = arma::regspace<arma::uvec>(start, end - 1);
        arma::uvec trainIdx = arma::join_cols(
            arma::regspace<arma::uvec>(0, start - 1),
            arma::regspace<arma::uvec>(end, n - 1)
        );

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

    cout << "[Debug] Pre-retrain full data dims: "
         << trainX.n_rows << "×" << trainX.n_cols
         << " | outputs: " << trainY.n_rows << "×"
         << trainY.n_cols << endl;

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
