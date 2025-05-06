/**
 * An example of using Recurrent Neural Network (RNN)
 * to make forecasts on a time series of ASM.
 * which we aim to solve using a simple LSTM neural network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file LSTMTimeSeriesMultivariate.cpp
 * @author Mehul Kumar Nirala.
 * @author Zoltan Somogyi
 * Modified by Behzad Shakouri & Arash Massoudieh
 */

/*
NOTE: the data need to be sorted by date in ascending order! The RNN learns from
oldest to newest!

Date  Inputs Outputs
27-06-16  ... ...
28-06-16  ... ...
...
*/

#include <QDebug>
#include "lstmtimeseriesset.h"


#define MLPACK_ENABLE_ANN_SERIALIZATION
//#include <mlpack.hpp>
#include <pch.h>

using namespace std;
using namespace mlpack;
using namespace ens;

/*
 * Function to calculate MSE for arma::cube.
 */
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

/**
 *
 * NOTE: We do not use the last input data point in the training because there
 * is no target (next day (high, low)) for that point.
 */
template<typename InputDataType = arma::mat,
         typename DataType = arma::cube,
         typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset,
                          DataType& X,
                          LabelType& y,
                          const size_t rho,
                          const int inputsize,
                          const int outputsize,
                          const bool IO)
{
    if (!IO) {
        for (size_t i = 0; i < dataset.n_cols - rho; i++)
            {
            //arma::cube LHS = X.subcube(arma::span(), arma::span(i), arma::span());
            //qDebug()<<LHS.n_rows<<","<<LHS.n_cols<<","<<LHS.n_slices;
            //arma::mat RHS = dataset.submat(arma::span(0), arma::span(i, i + rho - 1));
            //qDebug()<<RHS.n_rows<<","<<RHS.n_cols;
            X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(0, inputsize - 1), arma::span(i, i + rho - 1)); //col 0 input
            y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(inputsize, inputsize + outputsize - 1), arma::span(i + 1, i + rho)); //col 1 output , dataset.submat(arma::span(3, 4), arma::span(i + 1, i + rho)), dataset.submat(arma::span(9, 10), arma::span(i + 1, i + rho))
            }
    }
    else if (IO) {
        for (size_t i = 0; i < dataset.n_cols - rho; i++)
        {
            X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(0, inputsize - 1), arma::span(i, i + rho - 1)); //col 0 input
            y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(inputsize - outputsize, inputsize - 1), arma::span(i + 1, i + rho)); //col 1 output , dataset.submat(arma::span(3, 4), arma::span(i + 1, i + rho)), dataset.submat(arma::span(9, 10), arma::span(i + 1, i + rho))
        }
    }
}

/**
 * This function saves the input data for prediction and the prediction results
 * in CSV format. The prediction results are the (high, low) for the next day
 * and come from the last slice of the prediction. The last 2 columns are the
 * predictions; the preceding columns are the data used to generate those
 * predictions.
 */
void SaveResults(const string filename,
                 const arma::cube& predictions,
                 data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputsize,
                 const int outputsize,
                 const bool IO)
{
    arma::mat flatDataAndPreds = IOData.slice(IOData.n_slices - 1);
    scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);

    // The prediction results are the (high, low) for the next day and come from
    // the last slice from the prediction.
    arma::mat temp = predictions.slice(predictions.n_slices - 1);
    temp.save("/home/behzad/Projects/LSTM_TRY/Results/temp.txt", arma::arma_ascii);

    // NOTE: We add 3 extra rows here in order to recreate the input data
    // structure used to transform the data. This is needed in order to be able
    // to use the right scaling parameters for the specific column stock
    // (high, low).
    if (!IO)
        temp.insert_rows(0, inputsize, 0); // temp.insert_rows(0, 3, 0); // inputsize = 9 for ASM
    else if (IO)
        temp.insert_rows(0, inputsize - outputsize, 0); // temp.insert_rows(0, 3, 0); // inputsize = 9 for ASM

    scale.InverseTransform(temp, temp);

    // We shift the predictions such that the true values are synchronized with
    // the predictions, and we also add one more record to the input. Please note
    // that this means the last input record is zero and the first prediction record
    // is also zero.
    temp.insert_cols(0, 1, true);
    flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);

    // We add the prediction as the last columns.

    flatDataAndPreds.insert_rows(flatDataAndPreds.n_rows, temp.rows(temp.n_rows - outputsize, temp.n_rows - 1));

    // Save the data to file. The last columns are the predictions; the preceding
    // columns are the data used to generate those predictions.
    data::Save(filename, flatDataAndPreds);

    // Print the output to screen.
    cout << "The predicted output (last one) is: " << endl;
    int counter=outputsize;
    while (counter > 0) {
        cout << " (" << flatDataAndPreds(flatDataAndPreds.n_rows - counter, flatDataAndPreds.n_cols - 1) << ")" << endl;
    counter--;
    }

}

int main()
{

    bool ASM = 1;

    bool IO = 0; // IO = 0 if Targets (Outputs) will NOT be considered as Inputs (ASM), IO = 1 if Targets (Outputs) will be considered as Inputs (Stock Market).

    if (ASM)
        IO = 0;

    // We have 9 input data columns and 1/2/3/... output columns (targets).
    size_t inputSize = 9, outputSize = 1; // ASM

    // We have 5 input data columns and 2 output columns (target).
    if (!ASM)
        inputSize = 5, outputSize = 2; // Stock market : 5,2

    // Change the names of these files as necessary. They should be correct
    // already, if your program's working directory contains the data and/or
    // model.
    string dataFile;

    if (ASM)
        dataFile = "/home/behzad/Projects/LSTM_TRY/observedoutput_t10&11_sCOD.txt";
    else if (!ASM)
        dataFile = "/home/behzad/Projects/LSTM_TRY/Google2016-2019.csv";

    // example: const string dataFile =
    //              "C:/mlpack-model-app/Google2016-2019.csv";
    // example: const string dataFile =
    //              "/home/user/mlpack-model-app/Google2016-2019.csv";

    const string modelFile = "/home/behzad/Projects/LSTM_TRY/Results/lstm_multi.bin";
    // example: const string modelFile =
    //              "C:/mlpack-model-app/lstm_multi.bin";
    // example: const string modelFile =
    //              "/home/user/mlpack-model-app/lstm_multi.bin";

    const string predFile_Test = "/home/behzad/Projects/LSTM_TRY/Results/lstm_multi_predictions_test.csv";
    const string predFile_Train = "/home/behzad/Projects/LSTM_TRY/Results/lstm_multi_predictions_train.csv";

    // If true, the model will be trained; if false, the saved model will be
    // read and used for prediction
    // NOTE: Training the model may take a long time, therefore once it is
    // trained you can set this to false and use the model for prediction.
    // NOTE: There is no error checking in this example to see if the trained
    // model exists!
    const bool bTrain = true;
    // You can load and further train a model by setting this to true.
    const bool bLoadAndTrain = false;

    // Testing data is taken from the dataset in this ratio.
    const double RATIO = 0.5; //0.1

    // Step size of an optimizer.
    const double STEP_SIZE = 5e-5;

    // Number of epochs for training.
    const int EPOCHS = 1000; // 150

    // Number of cells in the LSTM (hidden layers in standard terms).
    // NOTE: you may play with this variable in order to further optimize the
    // model (as more cells are added, accuracy is likely to go up, but training
    // time may take longer).
    const int H1 = 10*2; //15
    const int H2 = 8*2; //15
    const int H3 = 7*2; //15

    // Number of data points in each iteration of SGD.
    const size_t BATCH_SIZE = 16;

    // Number of timesteps to look backward for in the RNN.
    const int rho = 1; //25 ~ Lag

    arma::mat dataset;

    // In Armadillo rows represent features, columns represent data points.
    cout << "Reading data ..." << endl;
    data::Load(dataFile, dataset, true);

    // The CSV file has a header, so it is necessary to remove it. In Armadillo's
    // representation it is the first column.
    // The first column in the CSV is the date which is not required, therefore
    // we remove it also (first row in in arma::mat).

    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1); // Eliminate the first column

    // Split the dataset into training and validation sets.
    arma::mat trainData;
    arma::mat testData;
    data::Split(dataset, trainData, testData, RATIO, false);

    trainData.save("/home/behzad/Projects/LSTM_TRY/Results/trainData_bs.csv", arma::csv_ascii);
    testData.save("/home/behzad/Projects/LSTM_TRY/Results/testData_bs.csv", arma::csv_ascii);

    // Scale all data into the range (0, 1) for increased numerical stability.
    data::MinMaxScaler scale;
    // Fit scaler only on training data.
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    // We need to represent the input data for RNN in an arma::cube (3D matrix).
    // The 3rd dimension is the rho number of past data records the RNN uses for
    // learning.
    arma::cube trainX, trainY, testX, testY;
    trainX.set_size(inputSize, trainData.n_cols - rho, rho);
    trainY.set_size(outputSize, trainData.n_cols - rho, rho);
    testX.set_size(inputSize, testData.n_cols - rho, rho);
    testY.set_size(outputSize, testData.n_cols - rho, rho);

    // Create training sets for one-step-ahead regression.
    CreateTimeSeriesData(trainData, trainX, trainY, rho, inputSize, outputSize, IO);
    // Create test sets for one-step-ahead regression.
    CreateTimeSeriesData(testData, testX, testY, rho, inputSize, outputSize, IO);

    trainData.save("/home/behzad/Projects/LSTM_TRY/Results/trainData.csv", arma::csv_ascii);
    testData.save("/home/behzad/Projects/LSTM_TRY/Results/testData.csv", arma::csv_ascii);

    trainX.save("/home/behzad/Projects/LSTM_TRY/Results/trainX.txt", arma::arma_ascii);
    trainX.save("/home/behzad/Projects/LSTM_TRY/Results/trainY.txt", arma::arma_ascii);
    testX.save("/home/behzad/Projects/LSTM_TRY/Results/testX.txt", arma::arma_ascii);
    testX.save("/home/behzad/Projects/LSTM_TRY/Results/testY.txt", arma::arma_ascii);

    // Only train the model if required.
    if (bTrain || bLoadAndTrain)
    {
        // RNN regression model.
        RNN<MeanSquaredError, HeInitialization> model(rho);

        if (bLoadAndTrain)
        {
            // The model will be trained further.
            cout << "Loading and further training model..." << endl;
            data::Load(modelFile, "LSTMMulti", model);
        }
        else
        {
            // Model building.
            model.Add<Linear>(inputSize);
            //model.Add<Dropout>(0.1); //model.Add<LeakyReLU>(); //model.Add<ReLU>(); //model.Add<PReLU>(); //model.Add<Sigmoid>();
            model.Add<LSTM>(H1);
            //model.Add<Dropout>(0.1); //model.Add<LeakyReLU>(); //model.Add<ReLU>(); //model.Add<PReLU>(); //model.Add<Sigmoid>();
            model.Add<LSTM>(H2);
            //model.Add<Dropout>(0.1); //model.Add<LeakyReLU>(); //model.Add<ReLU>(); //model.Add<PReLU>(); //model.Add<Sigmoid>();
            model.Add<LSTM>(H3);
            model.Add<ReLU>();
            //model.Add<Dropout>(0.1); //model.Add<LeakyReLU>(); //model.Add<ReLU>(); //model.Add<PReLU>(); //model.Add<Sigmoid>();
            model.Add<Linear>(outputSize);
        }

        // Set parameters for the Adam optimizer.
        ens::Adam optimizer(
            STEP_SIZE,  // Step size of the optimizer.
            BATCH_SIZE, // Batch size. Number of data points that are used in each
            // iteration.
            0.9,        // Exponential decay rate for the first moment estimates.
            0.999,      // Exponential decay rate for the weighted infinity norm
            // estimates.
            1e-8, // Value used to initialise the mean squared gradient parameter.
            trainData.n_cols * EPOCHS, // Max number of iterations.
            1e-8,                      // Tolerance.
            true);

        // Instead of terminating based on the tolerance of the objective function,
        // we'll depend on the maximum number of iterations, and terminate early
        // using the EarlyStopAtMinLoss callback.
        optimizer.Tolerance() = -1;

        cout << "Training ..." << endl;

        model.Train(trainX,
                    trainY,
                    optimizer,
                    // PrintLoss Callback prints loss for each epoch.
                    ens::PrintLoss(),
                    // Progressbar Callback prints progress bar for each epoch.
                    ens::ProgressBar(),
                    // Stops the optimization process if the loss stops decreasing
                    // or no improvement has been made. This will terminate the
                    // optimization once we obtain a minima on training set.
                    ens::EarlyStopAtMinLoss());

        cout << "Finished training. \n Saving Model" << endl;
        data::Save(modelFile, "LSTMMulti", model);
        cout << "Model saved in " << modelFile << endl;
    }

    // NOTE: the code below is added in order to show how in a real application
    // the model would be saved, loaded and then used for prediction. Please note
    // that we do not have the last data point in testX because we did not use it
    // for the training, therefore the prediction result will be for the day
    // before.  In your own application you may of course load any dataset.

    // Load RNN model and use it for prediction.
    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    cout << "Loading model ..." << endl;
    data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predOutP_Test;
    arma::cube predOutP_Train;

    // Get predictions on test data points.
    modelP.Predict(testX, predOutP_Test);
    modelP.Predict(trainX, predOutP_Train);

    predOutP_Test.save("/home/behzad/Projects/LSTM_TRY/Results/predOutP_Test.txt", arma::arma_ascii);
    predOutP_Train.save("/home/behzad/Projects/LSTM_TRY/Results/predOutP_Train.txt", arma::arma_ascii);

    // Calculate MSE on prediction.
    double testMSEP = ComputeMSE(predOutP_Test, testY);
    cout << "Mean Squared Error on Prediction data points for test Data= " << testMSEP << endl;

    double trainMSEP = ComputeMSE(predOutP_Train, trainY);
    cout << "Mean Squared Error on Prediction data points for train Data= " << trainMSEP << endl;

    // Save the output predictions and show the results.    
    arma::cube trainIO, testIO;

    testIO = testX;
    if (!IO)
    testIO.insert_rows(testX.n_rows, testY);

    trainIO = trainX;
    if (!IO)
    trainIO.insert_rows(trainX.n_rows, trainY);

    testIO.save("testIO.txt", arma::arma_ascii);
    trainIO.save("trainIO.txt", arma::arma_ascii);

    cout << "test Data:" << endl;
    SaveResults(predFile_Test, predOutP_Test, scale, testIO, inputSize, outputSize, IO);
    cout << "train Data:" << endl;
    SaveResults(predFile_Train, predOutP_Train, scale, trainIO, inputSize, outputSize, IO);

    // Use this on Windows in order to keep the console window open.
    // cout << "Ready!" << endl;
    // getchar();
}
