/**
 * LSTM Wrapper with K-fold (Random / Expanding / Fixed-ratio) Training
 * for ASM-type time-series prediction using mlpack::RNN.
 *
 * @authors
 *   Behzad Shakouri & Arash Massoudieh
 */

#include <QDebug>
#include "lstmtimeseriesset.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <pch.h>

using namespace std;
using namespace mlpack;
using namespace ens;

#include <ensmallen.hpp>

#ifdef PowerEdge
string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif Behzad
string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif Arash
string path = "/home/arash/Projects/LSTM_Wrapper/";
#endif


/* --------------------------------------------------------------- */
/* --------------------- Utility Functions ----------------------- */
/* --------------------------------------------------------------- */

double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

double ComputeR2(arma::cube& pred, arma::cube& Y)
{
    arma::vec pred_flat = arma::vectorise(pred);
    arma::vec Y_flat = arma::vectorise(Y);
    const double ssRes = arma::accu(arma::square(Y_flat - pred_flat));
    const double ssTot = arma::accu(arma::square(Y_flat - arma::mean(Y_flat)));
    return 1.0 - ssRes / ssTot;
}

/* Extract selected slices (indices) from a cube (samples are in slices). */
arma::cube SelectSlices(const arma::cube& cubeIn, const arma::uvec& indices)
{
    arma::cube out(cubeIn.n_rows, cubeIn.n_cols, indices.n_elem);
    for (size_t i = 0; i < indices.n_elem; ++i)
        out.slice(i) = cubeIn.slice(indices(i));
    return out;
}

/* --------------------------------------------------------------- */
/* --------------------- Split Helpers (cube) -------------------- */
/* --------------------------------------------------------------- */

/* Random (traditional) k-fold split over samples (slices). */
auto KFoldSplit_Cube(const arma::cube& data, const arma::cube& labels, size_t k, size_t fold)
{
    if (k == 0 || fold >= k)
        throw std::invalid_argument("KFoldSplit_Cube: invalid fold index or k.");

    const size_t n = data.n_slices;
    const size_t foldSize = std::max<size_t>(1, n / k);
    const size_t start = fold * foldSize;
    const size_t end   = (fold == k - 1) ? n : std::min(n, start + foldSize);

    arma::uvec valIdx = arma::regspace<arma::uvec>(start, end - 1);

    arma::uvec mask(n, arma::fill::ones);
    mask(valIdx).zeros();
    arma::uvec trainIdx = arma::find(mask == 1);

    arma::cube trainData   = SelectSlices(data, trainIdx);
    arma::cube trainLabels = SelectSlices(labels, trainIdx);
    arma::cube validData   = SelectSlices(data, valIdx);
    arma::cube validLabels = SelectSlices(labels, valIdx);

    return std::make_pair(std::make_pair(trainData, trainLabels),
                          std::make_pair(validData, validLabels));
}

/*
 * Time-series expanding-window split:
 * Fold f (0-based):
 *   Train: slices [0 .. trainEnd-1]
 *   Val:   slices [trainEnd .. valEnd-1]
 * where trainEnd = min((f+1)*foldSize, n-1)
 * and valEnd = (last fold ? n : min(trainEnd + foldSize, n))
 */
auto KFoldSplit_TimeSeries_Cube(const arma::cube& data, const arma::cube& labels, size_t k, size_t fold)
{
    const size_t n = data.n_slices;
    const size_t foldSize = std::max<size_t>(1, n / k);

    if (n < 2)
        throw std::invalid_argument("KFoldSplit_TimeSeries_Cube: not enough samples.");

    size_t trainEnd = std::min((fold + 1) * foldSize, (n > 0 ? n - 1 : 0));
    trainEnd = std::max<size_t>(1, trainEnd); // ensure at least one training sample

    const size_t valStart = trainEnd;
    const size_t valEnd   = (fold == k - 1) ? n : std::min(valStart + foldSize, n);

    if (valEnd <= valStart)
        throw std::invalid_argument("KFoldSplit_TimeSeries_Cube: empty validation split.");

    arma::cube trainData   = data.slices(0, trainEnd - 1);
    arma::cube trainLabels = labels.slices(0, trainEnd - 1);
    arma::cube validData   = data.slices(valStart, valEnd - 1);
    arma::cube validLabels = labels.slices(valStart, valEnd - 1);

    return std::make_pair(std::make_pair(trainData, trainLabels),
                          std::make_pair(validData, validLabels));
}

/* --------------------------------------------------------------- */
/* -------------------- K-Fold Trainer (cube) --------------------- */
/* --------------------------------------------------------------- */

bool TrainWithKFold_Cube(
    const size_t inputSize,
    const size_t outputSize,
    const size_t rho,
    arma::cube& X,                 // features × rho × samples
    arma::cube& Y,                 // outputs  × rho × samples
    double stepSize,
    size_t batchSize,
    int epochs,
    int nFolds,
    int splitMode,                 // 0: random, 1: expanding time-series
    const std::string& outputPath = "./Results/")
{
    const int H1 = 20, H2 = 16, H3 = 14;

    const size_t nSamples = X.n_slices;
    if (nSamples < 2)
    {
        std::cerr << "[Error] Not enough samples to train (" << nSamples << ")\n";
        return false;
    }
    if (nSamples < static_cast<size_t>(nFolds))
    {
        std::cerr << "[Warning] Reducing folds to " << nSamples << ".\n";
        nFolds = static_cast<int>(nSamples);
    }

    const double trainRatio = 1.0 - 1.0 / static_cast<double>(nFolds);
    std::cout << "Starting " << nFolds << "-fold cross-validation (mode "
              << splitMode << ", train ratio ≈ " << trainRatio * 100 << "%)...\n";

    std::vector<double> trainMSEs, valMSEs, trainR2s, valR2s;
    double totalTrainMSE = 0, totalValMSE = 0, totalTrainR2 = 0, totalValR2 = 0;

    // =============================================================
    //                   CROSS-VALIDATION
    // =============================================================
    for (int fold = 0; fold < nFolds; ++fold)
    {
        arma::cube Xtrain, Ytrain, Xval, Yval;
        if (splitMode == 1)
        {
            auto s = KFoldSplit_TimeSeries_Cube(X, Y, nFolds, fold);
            Xtrain = s.first.first;  Ytrain = s.first.second;
            Xval   = s.second.first; Yval   = s.second.second;
        }
        else
        {
            auto s = KFoldSplit_Cube(X, Y, nFolds, fold);
            Xtrain = s.first.first;  Ytrain = s.first.second;
            Xval   = s.second.first; Yval   = s.second.second;
        }

        std::cout << "\nFold " << (fold + 1) << "/" << nFolds
                  << " | Train: " << Xtrain.n_slices
                  << " | Validation: " << Xval.n_slices
                  << " | Shape: " << Xtrain.n_rows << "×" << Xtrain.n_cols
                  << std::endl;

        // ------------------ Model ------------------
        RNN<MeanSquaredError, HeInitialization> model(rho);
        model.Add<Linear>(inputSize);
        model.Add<LSTM>(H1);
        model.Add<LSTM>(H2);
        model.Add<LSTM>(H3);
        model.Add<ReLU>();
        model.Add<Linear>(outputSize);

        const size_t maxIterations =
            std::min<size_t>(1000, static_cast<size_t>(epochs) * Xtrain.n_slices);

        ens::Adam opt(stepSize, batchSize, 0.9, 0.999, 1e-8,
                      maxIterations, 1e-8, true);
        opt.Tolerance() = -1;

        // ------------------ Training ------------------
        model.Train(Xtrain, Ytrain, opt,
                    ens::ProgressBar(),
                    ens::PrintLoss(),
                    ens::EarlyStopAtMinLoss(20));   // patience = 20 epochs

        // ------------------ Evaluation ------------------
        arma::cube predTrain, predVal;
        model.Predict(Xtrain, predTrain);
        model.Predict(Xval,   predVal);

        double mseTrain = ComputeMSE(predTrain, Ytrain);
        double r2Train  = ComputeR2(predTrain,  Ytrain);
        double mseVal   = ComputeMSE(predVal,   Yval);
        double r2Val    = ComputeR2(predVal,    Yval);

        trainMSEs.push_back(mseTrain);
        valMSEs.push_back(mseVal);
        trainR2s.push_back(r2Train);
        valR2s.push_back(r2Val);

        totalTrainMSE += mseTrain; totalValMSE += mseVal;
        totalTrainR2  += r2Train;  totalValR2  += r2Val;

        std::cout << "  Train MSE: " << mseTrain << " | R²: " << r2Train
                  << " | Val MSE: " << mseVal << " | R²: " << r2Val << "\n";
    }

    // ------------------ Averages ------------------
    const double avgTrainMSE = totalTrainMSE / trainMSEs.size();
    const double avgValMSE   = totalValMSE   / valMSEs.size();
    const double avgTrainR2  = totalTrainR2  / trainR2s.size();
    const double avgValR2    = totalValR2    / valR2s.size();

    std::cout << "\nAverage training   MSE: " << avgTrainMSE
              << " | R²: " << avgTrainR2 << std::endl;
    std::cout << "Average validation MSE: " << avgValMSE
              << " | R²: " << avgValR2 << std::endl;

    std::ofstream csv(outputPath + "kfold_results.csv");
    csv << "Fold,TrainMSE,TrainR2,ValMSE,ValR2\n";
    for (size_t i = 0; i < valMSEs.size(); ++i)
        csv << (i + 1) << "," << trainMSEs[i] << "," << trainR2s[i]
            << "," << valMSEs[i] << "," << valR2s[i] << "\n";
    csv << "Average," << avgTrainMSE << "," << avgTrainR2
        << "," << avgValMSE << "," << avgValR2 << "\n";
    csv.close();

    // =============================================================
    //               FINAL RETRAIN ON ALL DATA
    // =============================================================
    std::cout << "\nRetraining final model on full dataset...\n";

    RNN<MeanSquaredError, HeInitialization> finalModel(rho);
    finalModel.Add<Linear>(inputSize);
    finalModel.Add<LSTM>(H1);
    finalModel.Add<LSTM>(H2);
    finalModel.Add<LSTM>(H3);
    finalModel.Add<ReLU>();
    finalModel.Add<Linear>(outputSize);

    const size_t maxIterationsFinal =
        std::min<size_t>(1000, static_cast<size_t>(epochs) * X.n_slices);

    ens::Adam finalOpt(stepSize, batchSize, 0.9, 0.999, 1e-8,
                       maxIterationsFinal, 1e-8, true);
    finalOpt.Tolerance() = -1;

    finalModel.Train(X, Y, finalOpt,
                     ens::ProgressBar(),
                     ens::PrintLoss(),
                     ens::EarlyStopAtMinLoss(20));

    arma::cube fullPred;
    finalModel.Predict(X, fullPred);

    double mseFull = ComputeMSE(fullPred, Y);
    double r2Full  = ComputeR2(fullPred,  Y);

    std::cout << "Final full-data MSE: " << mseFull << " | R²: " << r2Full << std::endl;

    data::Save(outputPath + "lstm_final.bin", "LSTMMulti", finalModel);
    return true;
}



/* --------------------------------------------------------------- */
/* ---------------- Time-Series Data Pre-Processor ---------------- */
/* --------------------------------------------------------------- */
/*
 * Build cubes as: features × timesteps (rho) × samples
 * This matches mlpack’s RNN API (slices = batch).
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
    const size_t nSamples = dataset.n_cols - rho;

    // features × rho × samples
    X.set_size(inputsize, rho, nSamples);
    y.set_size(outputsize, rho, nSamples);

    if (!IO)
    {
        for (size_t i = 0; i < nSamples; ++i)
        {
            // Inputs: rows [0 .. inputsize-1], cols [i .. i+rho-1]
            X.slice(i) = dataset.submat(0, i, inputsize - 1, i + rho - 1);
            // Targets (shifted by +1 in time): rows [inputsize .. inputsize+outputsize-1],
            //                                   cols [i+1 .. i+rho]
            y.slice(i) = dataset.submat(inputsize, i + 1,
                                        inputsize + outputsize - 1, i + rho);
        }
    }
    else
    {
        for (size_t i = 0; i < nSamples; ++i)
        {
            X.slice(i) = dataset.submat(0, i, inputsize - 1, i + rho - 1);
            y.slice(i) = dataset.submat(inputsize - outputsize, i + 1,
                                        inputsize - 1, i + rho);
        }
    }
}

/* --------------------------------------------------------------- */
/* ---------------------------- main ------------------------------ */
/* --------------------------------------------------------------- */

int main()
{
    bool ASM = true, IO = false;
    size_t inputSize = 9, outputSize = 1;
    if (!ASM) inputSize = 5, outputSize = 2;

    const string data_name = "NO";
    const string dataFile = ASM
        ? path + "Data/observedoutput_t10&11_" + data_name + ".txt"
        : path + "Data/Google2016-2019.csv";

    // ----------------- Hyperparameters -----------------
    const double RATIO      = 0.30;      // train/test split for *reporting* only
    const double STEP_SIZE  = 1e-3;      // Adam step size
    const int    EPOCHS     = 1000;      // effective passes budgeted via maxIterations
    const size_t BATCH_SIZE = 16;
    const int    rho        = 1;        // temporal window length (set >1 for sequence learning)
    const int    KFOLDS     = 10;         // K-Fold
    const int    SPLIT_MODE = 0;         // 0 = random K-fold, 1 = expanding window, 2 = fixed
    // ---------------------------------------------------

    arma::mat dataset;
    cout << "Reading data ..." << endl;
    data::Load(dataFile, dataset, true);

    // Drop header row & first column (date).
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    // Split for *reporting* (model is trained via K-Fold on the training partition here).
    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, RATIO, false);

    // Scale with parameters fit on train only.
    data::MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    // Build cubes (features × rho × samples).
    arma::cube trainX, trainY, testX, testY;
    CreateTimeSeriesData(trainData, trainX, trainY, rho, (int)inputSize, (int)outputSize, IO);
    CreateTimeSeriesData(testData,  testX,  testY,  rho, (int)inputSize, (int)outputSize, IO);

    cout << "trainX dims: " << trainX.n_rows << "×" << trainX.n_cols
         << "×" << trainX.n_slices << endl;
    cout << "testX dims: "  << testX.n_rows  << "×" << testX.n_cols
         << "×" << testX.n_slices  << endl;

    // K-Fold CV + final model training on the full training cube.
    if (!TrainWithKFold_Cube(inputSize, outputSize, rho, trainX, trainY,
                             STEP_SIZE, BATCH_SIZE, EPOCHS, KFOLDS, SPLIT_MODE,
                             path + "Results/"))
    {
        std::cerr << "[Error] Training failed.\n";
        return 255;
    }

    // Load final model and evaluate on both train & test cubes.
    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    cout << "Loading final model ..." << endl;
    if (!data::Load(path + "Results/lstm_final.bin", "LSTMMulti", modelP))
    {
        cerr << "[Error] Final model not found or training failed.\n";
        return -1;
    }

    arma::cube predOutP_Test, predOutP_Train;
    modelP.Predict(testX,  predOutP_Test);
    modelP.Predict(trainX, predOutP_Train);

    double testMSE  = ComputeMSE(predOutP_Test,  testY);
    double trainMSE = ComputeMSE(predOutP_Train, trainY);
    double testR2   = ComputeR2(predOutP_Test,   testY);
    double trainR2  = ComputeR2(predOutP_Train,  trainY);

    cout << "Test  MSE = " << testMSE  << ", R² = " << testR2
         << " | Train MSE = " << trainMSE << ", R² = " << trainR2 << endl;

    cout << "✅ Done.\n";
    return 0;
}


//--------------------------------------------------------------------------------------------------
//---------------------------------------OLD version------------------------------------------------
//--------------------------------------------------------------------------------------------------

/**
 * LSTM Wrapper
 * for ASM-type time-series prediction using mlpack::RNN.
 *
 * @authors
 *   Behzad Shakouri & Arash Massoudieh
 */

/*
NOTE: the data need to be sorted by date in ascending order! The RNN learns from
oldest to newest!

Date  Inputs Outputs
27-06-16  ... ...
28-06-16  ... ...
...
*/


/*
 * Function to calculate MSE for arma::cube.
 */
double ComputeMSE_old(arma::cube& pred, arma::cube& Y)
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
void CreateTimeSeriesData_old(InputDataType dataset,
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
void SaveResults_old(const string filename,
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
    temp.save(path + "Results/temp.txt", arma::arma_ascii);

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

int main_()
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

    string data_name = "NO"; // NO, NH, ND, sCOD, VSS, TKN

    if (ASM)
        dataFile = path + "Data/observedoutput_t10&11_" + data_name + ".txt";
    else if (!ASM)
        dataFile = path + "Data/Google2016-2019.csv";

    // example: const string dataFile =
    //              "C:/mlpack-model-app/Google2016-2019.csv";
    // example: const string dataFile =
    //              "/home/user/mlpack-model-app/Google2016-2019.csv";

    const string modelFile = path + "Results/lstm_multi.bin";
    // example: const string modelFile =
    //              "C:/mlpack-model-app/lstm_multi.bin";
    // example: const string modelFile =
    //              "/home/user/mlpack-model-app/lstm_multi.bin";

    const string predFile_Test = path + "Results/lstm_multi_predictions_test.csv";
    const string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

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
    const double RATIO = 0.3; // 0.2~0.3

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

    trainData.save(path + "Results/trainData_bs.csv", arma::csv_ascii);
    testData.save(path + "Results/testData_bs.csv", arma::csv_ascii);

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
    CreateTimeSeriesData_old(trainData, trainX, trainY, rho, inputSize, outputSize, IO);
    // Create test sets for one-step-ahead regression.
    CreateTimeSeriesData_old(testData, testX, testY, rho, inputSize, outputSize, IO);

    trainData.save(path + "Results/trainData.csv", arma::csv_ascii);
    testData.save(path + "Results/testData.csv", arma::csv_ascii);

    trainX.save(path + "Results/trainX.txt", arma::arma_ascii);
    trainX.save(path + "Results/trainY.txt", arma::arma_ascii);
    testX.save(path + "Results/testX.txt", arma::arma_ascii);
    testX.save(path + "Results/testY.txt", arma::arma_ascii);

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

    predOutP_Test.save(path + "Results/predOutP_Test.txt", arma::arma_ascii);
    predOutP_Train.save(path + "Results/predOutP_Train.txt", arma::arma_ascii);

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

    testIO.save(path + "Results/testIO.txt", arma::arma_ascii);
    trainIO.save(path + "Results/trainIO.txt", arma::arma_ascii);

    cout << "test Data:" << endl;
    SaveResults_old(predFile_Test, predOutP_Test, scale, testIO, inputSize, outputSize, IO);
    cout << "train Data:" << endl;
    SaveResults_old(predFile_Train, predOutP_Train, scale, trainIO, inputSize, outputSize, IO);


    // Use this on Windows in order to keep the console window open.
    // cout << "Ready!" << endl;
    // getchar();
}
