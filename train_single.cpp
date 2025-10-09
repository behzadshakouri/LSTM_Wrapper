#include <armadillo>
#include <iostream>
#include <string>
#include <vector>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;
using namespace std;

#include <pch.h>       // ✅ must come first
#include "helpers.h"   // now all mlpack headers use the defined macro

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
