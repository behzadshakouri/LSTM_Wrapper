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
void TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int kfolds,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool /*ASM*/,
                bool bTrain, bool bLoadAndTrain)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    int n = dataset.n_cols;
    int foldSize = n / kfolds;

    for (int fold = 0; fold < kfolds; ++fold)
    {
        cout << "\n===== Fold " << (fold + 1) << " / " << kfolds << " =====" << endl;
        int start = fold * foldSize;
        int end = (fold == kfolds - 1) ? n : start + foldSize;

        arma::uvec testIdx = arma::regspace<arma::uvec>(start, end - 1);
        arma::uvec trainIdx = arma::find(arma::ones<arma::uvec>(n) - arma::conv_to<arma::uvec>::from(arma::linspace<arma::vec>(start, end - 1, end - start)));

        arma::mat trainData = dataset.cols(trainIdx);
        arma::mat testData  = dataset.cols(testIdx);

        std::string foldModel = modelFile.substr(0, modelFile.find_last_of('.')) + "_fold" + std::to_string(fold + 1) + ".bin";
        std::string foldPredT = predFile_Test.substr(0, predFile_Test.find_last_of('.')) + "_fold" + std::to_string(fold + 1) + ".csv";
        std::string foldPredR = predFile_Train.substr(0, predFile_Train.find_last_of('.')) + "_fold" + std::to_string(fold + 1) + ".csv";

        TrainCore(trainData, testData, foldModel, foldPredT, foldPredR,
                  inputSize, outputSize, rho, stepSize, epochs,
                  batchSize, IO, bTrain, bLoadAndTrain);
    }
}


/*
bool TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int k,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain);

void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool ASM,
                 bool bTrain, bool bLoadAndTrain);
*/
