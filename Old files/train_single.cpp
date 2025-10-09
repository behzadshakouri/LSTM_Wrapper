#include <armadillo>
#include <iostream>
#include <string>
#include <vector>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;
using namespace std;

#include "train_modes.h"

#include <pch.h>       // ✅ must come first
#include "helpers.h"   // now all mlpack headers use the defined macro

#include <ensmallen.hpp>

void TrainSingle_(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool /*ASM*/,
                 bool bTrain, bool bLoadAndTrain)
{
    // === Load & preprocess (identical logic) ===
    arma::mat dataset;
    mlpack::data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    mlpack::data::Split(dataset, trainData, testData, ratio, false);

    mlpack::data::MinMaxScaler scale;
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

    // === Train (exact architecture & optimizer) ===
    if (bTrain || bLoadAndTrain)
    {
        RNN<MeanSquaredError, HeInitialization> model(rho);

        if (bLoadAndTrain)
        {
            cout << "Loading and further training model..." << endl;
            mlpack::data::Load(modelFile, "LSTMMulti", model);
        }
        else
        {
            // H1=20, H2=16, H3=14 exactly as old code (10*2, 8*2, 7*2).
            const int H1 = 20, H2 = 16, H3 = 14;

            model.Add<Linear>(inputSize);
            model.Add<LSTM>(H1);
            model.Add<LSTM>(H2);
            model.Add<LSTM>(H3);
            model.Add<ReLU>();
            model.Add<Linear>(outputSize);
        }

        ens::Adam optimizer(
            stepSize,              // 5e-5
            batchSize,             // 16
            0.9, 0.999, 1e-8,
            trainData.n_cols * epochs, // max iters = cols * 1000
            1e-8,
            true
        );
        optimizer.Tolerance() = -1; // Use EarlyStopAtMinLoss

        cout << "Training ..." << endl;
        model.Train(trainX, trainY, optimizer,
                    ens::PrintLoss(),
                    ens::ProgressBar(),
                    ens::EarlyStopAtMinLoss());

        cout << "Finished training. \nSaving Model" << endl;
        mlpack::data::Save(modelFile, "LSTMMulti", model);
        cout << "Model saved in " << modelFile << endl;
    }

    // === Predict and report (unchanged) ===
    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    mlpack::data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predTest, predTrain;
    modelP.Predict(testX , predTest);
    modelP.Predict(trainX, predTrain);

    double mseTrain = ComputeMSE(predTrain, trainY);
    double mseTest  = ComputeMSE(predTest , testY );

    double r2Train  = ComputeR2(predTrain, trainY);
    double r2Test   = ComputeR2(predTest , testY );

    cout << "Train MSE = " << mseTrain << ", R² = " << r2Train << endl;
    cout << "Test  MSE = " << mseTest  << ", R² = " << r2Test  << endl;

    // === Save outputs (identical layout) ===
    arma::cube testIO  = testX;
    arma::cube trainIO = trainX;
    if (!IO)
    {
        testIO.insert_rows (testX.n_rows , testY );
        trainIO.insert_rows(trainX.n_rows, trainY);
    }

    SaveResults(predFile_Test , predTest , scale, testIO , (int)inputSize, (int)outputSize, IO);
    SaveResults(predFile_Train, predTrain, scale, trainIO, (int)inputSize, (int)outputSize, IO);

    cout << "✅ Done (TrainSingle mirrored to old version)." << endl;
}

/*
bool TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool ASM,
                 bool bTrain, bool bLoadAndTrain)
{
    cout << "Training on single split..." << endl;

    arma::mat dataset;
    data::Load(dataFile, dataset, true);

    // Remove header/date column
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;

    // ✅ Chronological split (old behavior)
    data::Split(dataset, trainData, testData, ratio, true);

    // ✅ Scale based on training only
    data::MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    // Prevent out-of-range values due to unseen test distribution
    testData = arma::clamp(testData, 0.0, 1.0);

    // ✅ Optional debug scaling check
    cout << std::fixed << setprecision(3);
    cout << "Train range: [" << trainData.min() << ", " << trainData.max() << "]" << endl;
    cout << "Test  range: [" << testData.min()  << ", " << testData.max()  << "]" << endl;

    arma::cube trainX(inputSize, trainData.n_cols - rho, rho);
    arma::cube trainY(outputSize, trainData.n_cols - rho, rho);
    arma::cube testX(inputSize, testData.n_cols - rho, rho);
    arma::cube testY(outputSize, testData.n_cols - rho, rho);

    CreateTimeSeriesData(trainData, trainX, trainY, rho, inputSize, outputSize, IO);
    CreateTimeSeriesData(testData, testX, testY, rho, inputSize, outputSize, IO);

    if (bTrain || bLoadAndTrain)
    {
        RNN<MeanSquaredError, HeInitialization> model(rho);

        if (bLoadAndTrain)
        {
            cout << "Loading and continuing training..." << endl;
            data::Load(modelFile, "LSTMMulti", model);
        }
        else
        {
            // ✅ Restore old architecture (stronger)
            model.Add<Linear>(inputSize);
            model.Add<LSTM>(40);
            model.Add<LSTM>(32);
            model.Add<LSTM>(28);
            model.Add<ReLU>();
            model.Add<Linear>(outputSize);
        }

        // ✅ Restore old epochs & training intensity
        ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, 1e-8,
                            trainData.n_cols * epochs, 1e-8, true);
        optimizer.Tolerance() = -1;

        cout << "Epochs: " << epochs << ", Step size: " << stepSize << endl;

        model.Train(trainX, trainY, optimizer,
                    ens::PrintLoss(), ens::ProgressBar(),
                    ens::EarlyStopAtMinLoss());

        data::Save(modelFile, "LSTMMulti", model);
        cout << "Model saved to: " << modelFile << endl;
    }

    // Reload and predict
    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predTrain, predTest;
    modelP.Predict(trainX, predTrain);
    modelP.Predict(testX, predTest);

    double mseTrain = ComputeMSE(predTrain, trainY);
    double mseTest  = ComputeMSE(predTest, testY);
    double r2Train  = ComputeR2(predTrain, trainY);
    double r2Test   = ComputeR2(predTest, testY);

    cout << "Test  MSE = " << mseTest  << ", R² = " << r2Test  << endl;
    cout << "Train MSE = " << mseTrain << ", R² = " << r2Train << endl;

    cout << "Saving results..." << endl;
    SaveResults(predFile_Test,  predTest,  scale, testX,  inputSize, outputSize, IO);
    SaveResults(predFile_Train, predTrain, scale, trainX, inputSize, outputSize, IO);

    cout << "✅ Done (TrainSingle mode)." << endl;
    return true;
}
*/
