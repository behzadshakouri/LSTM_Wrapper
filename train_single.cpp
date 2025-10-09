#include "helpers.h"

void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool ASM,
                 bool bTrain, bool bLoadAndTrain)
{
    arma::mat dataset;
    data::Load(dataFile, dataset, true);

    // Remove header/date columns if present
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, ratio, false);

    data::MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

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
            std::cout << "Loading existing model for further training..." << std::endl;
            data::Load(modelFile, "LSTMMulti", model);
        }
        else
        {
            model.Add<Linear>(inputSize);
            model.Add<LSTM>(20);
            model.Add<LSTM>(16);
            model.Add<LSTM>(14);
            model.Add<ReLU>();
            model.Add<Linear>(outputSize);
        }

        ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, 1e-8,
                            trainData.n_cols * epochs, 1e-8, true);
        optimizer.Tolerance() = -1;

        std::cout << "Training on single split..." << std::endl;

        model.Train(trainX, trainY, optimizer,
                    ens::PrintLoss(), ens::ProgressBar(), ens::EarlyStopAtMinLoss());

        data::Save(modelFile, "LSTMMulti", model);
        std::cout << "Model saved to: " << modelFile << std::endl;
    }

    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predTrain, predTest;
    modelP.Predict(trainX, predTrain);
    modelP.Predict(testX, predTest);

    double mseTrain = ComputeMSE(predTrain, trainY);
    double r2Train  = ComputeR2(predTrain, trainY);
    double mseTest  = ComputeMSE(predTest, testY);
    double r2Test   = ComputeR2(predTest, testY);

    std::cout << "Test  MSE = " << mseTest  << ", R² = " << r2Test  << std::endl;
    std::cout << "Train MSE = " << mseTrain << ", R² = " << r2Train << std::endl;

    arma::cube trainIO = trainX;
    arma::cube testIO  = testX;
    if (!IO)
    {
        trainIO.insert_rows(trainX.n_rows, trainY);
        testIO.insert_rows(testX.n_rows, testY);
    }

    std::cout << "Saving results..." << std::endl;
    SaveResults(predFile_Test, predTest, scale, testIO, inputSize, outputSize, IO);
    SaveResults(predFile_Train, predTrain, scale, trainIO, inputSize, outputSize, IO);

    std::cout << "✅ Done (TrainSingle mode)." << std::endl;
}
