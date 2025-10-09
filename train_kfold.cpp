#include <mlpack.hpp>
#include <armadillo>
#include <iostream>
#include <string>
#include <vector>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::data;
using namespace ens;
using namespace std;

#include "helpers.h"

/* ============================================================
 *                K-Fold Training Function
 * ============================================================ */

bool TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int k,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, 0.3, false);

    data::MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    size_t nCols = trainData.n_cols;
    size_t foldSize = nCols / k;

    vector<double> mseTrainFold, r2TrainFold, mseValFold, r2ValFold;

    cout << "Starting " << k << "-fold cross-validation..." << endl;

    for (int fold = 0; fold < k; ++fold)
    {
        size_t start = fold * foldSize;
        size_t end = (fold == k - 1) ? nCols : (fold + 1) * foldSize;

        arma::mat valData = trainData.cols(start, end - 1);
        arma::mat subTrain;

        if (start == 0)
            subTrain = trainData.cols(end, nCols - 1);
        else if (end >= nCols)
            subTrain = trainData.cols(0, start - 1);
        else
            subTrain = arma::join_rows(trainData.cols(0, start - 1),
                                       trainData.cols(end, nCols - 1));

        arma::cube trainX(inputSize, subTrain.n_cols - rho, rho);
        arma::cube trainY(outputSize, subTrain.n_cols - rho, rho);
        arma::cube valX(inputSize, valData.n_cols - rho, rho);
        arma::cube valY(outputSize, valData.n_cols - rho, rho);

        CreateTimeSeriesData(subTrain, trainX, trainY, rho, inputSize, outputSize, IO);
        CreateTimeSeriesData(valData, valX, valY, rho, inputSize, outputSize, IO);

        RNN<MeanSquaredError, HeInitialization> model(rho);
        model.Add<Linear>(inputSize);
        model.Add<LSTM>(20);
        model.Add<LSTM>(16);
        model.Add<LSTM>(14);
        model.Add<ReLU>();
        model.Add<Linear>(outputSize);

        ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, 1e-8,
                            subTrain.n_cols * epochs, 1e-8, true);
        optimizer.Tolerance() = -1;

        cout << "\nFold " << (fold + 1) << "/" << k << endl;

        model.Train(trainX, trainY, optimizer,
                    ens::PrintLoss(), ens::ProgressBar(),
                    ens::EarlyStopAtMinLoss());

        arma::cube predTrain, predVal;
        model.Predict(trainX, predTrain);
        model.Predict(valX, predVal);

        double mseTrain = ComputeMSE(predTrain, trainY);
        double mseVal = ComputeMSE(predVal, valY);
        double r2Train = ComputeR2(predTrain, trainY);
        double r2Val = ComputeR2(predVal, valY);

        mseTrainFold.push_back(mseTrain);
        mseValFold.push_back(mseVal);
        r2TrainFold.push_back(r2Train);
        r2ValFold.push_back(r2Val);

        cout << "  Train MSE: " << mseTrain << " | R²: " << r2Train
             << " | Val MSE: " << mseVal << " | R²: " << r2Val << endl;
    }

    cout << "\nAverage training   MSE: " << arma::mean(arma::vec(mseTrainFold))
         << " | R²: " << arma::mean(arma::vec(r2TrainFold)) << endl;
    cout << "Average validation MSE: " << arma::mean(arma::vec(mseValFold))
         << " | R²: " << arma::mean(arma::vec(r2ValFold)) << endl;

    cout << "\nRetraining final model on full dataset..." << endl;

    arma::cube fullX(inputSize, trainData.n_cols - rho, rho);
    arma::cube fullY(outputSize, trainData.n_cols - rho, rho);
    CreateTimeSeriesData(trainData, fullX, fullY, rho, inputSize, outputSize, IO);

    RNN<MeanSquaredError, HeInitialization> finalModel(rho);
    finalModel.Add<Linear>(inputSize);
    finalModel.Add<LSTM>(20);
    finalModel.Add<LSTM>(16);
    finalModel.Add<LSTM>(14);
    finalModel.Add<ReLU>();
    finalModel.Add<Linear>(outputSize);

    ens::Adam finalOpt(stepSize, batchSize, 0.9, 0.999, 1e-8,
                       trainData.n_cols * epochs, 1e-8, true);
    finalOpt.Tolerance() = -1;

    finalModel.Train(fullX, fullY, finalOpt,
                     ens::PrintLoss(), ens::ProgressBar(),
                     ens::EarlyStopAtMinLoss());

    arma::cube predFull, predTest;
    finalModel.Predict(fullX, predFull);

    arma::cube testX(inputSize, testData.n_cols - rho, rho);
    arma::cube testY(outputSize, testData.n_cols - rho, rho);
    CreateTimeSeriesData(testData, testX, testY, rho, inputSize, outputSize, IO);

    finalModel.Predict(testX, predTest);

    double mseFull = ComputeMSE(predFull, fullY);
    double r2Full  = ComputeR2(predFull, fullY);
    double mseTest = ComputeMSE(predTest, testY);
    double r2Test  = ComputeR2(predTest, testY);

    cout << "Final full-data MSE: " << mseFull << " | R²: " << r2Full << endl;
    cout << "Test  MSE = " << mseTest << ", R² = " << r2Test << endl;

    cout << "✅ Done (TrainKFold mode)." << endl;

    data::Save(modelFile, "LSTMMulti", finalModel);
    return true;
}
