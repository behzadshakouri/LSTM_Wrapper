/**
 * LSTM Wrapper with K-fold (Random / Expanding / Fixed-ratio) Training
 * for ASM-type time-series prediction using mlpack::RNN.
 *
 * @authors
 *   Mehul Kumar Nirala,
 *   Zoltan Somogyi,
 *   Modified by Behzad Shakouri & Arash Massoudieh
 */

#include <QDebug>
#include "lstmtimeseriesset.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <pch.h>

using namespace std;
using namespace mlpack;
using namespace ens;

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
    arma::mat pred_flat = arma::vectorise(pred);
    arma::mat Y_flat = arma::vectorise(Y);

    double ssRes = arma::accu(arma::square(Y_flat - pred_flat));
    double ssTot = arma::accu(arma::square(Y_flat - arma::mean(Y_flat)));
    return 1.0 - ssRes / ssTot;
}

/* Extract selected slices (indices) from a cube */
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

// --- Random ---
auto KFoldSplit_Cube(const arma::cube& data, const arma::cube& labels, size_t k, size_t fold)
{
    if (k == 0 || fold >= k)
        throw std::invalid_argument("KFoldSplit_Cube: invalid fold index or k.");

    size_t n = data.n_slices;
    size_t foldSize = n / k;
    size_t start = fold * foldSize;
    size_t end   = (fold == k - 1) ? n : start + foldSize;

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

// --- Time series (expanding window) ---
auto KFoldSplit_TimeSeries_Cube(const arma::cube& data, const arma::cube& labels, size_t k, size_t fold)
{
    size_t n = data.n_slices;
    size_t foldSize = n / k;
    size_t valStart = fold * foldSize;
    size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;

    size_t trainEnd = (fold == 0) ? foldSize : valStart;
    if (trainEnd < 2) trainEnd = foldSize;

    arma::cube trainData   = data.slices(0, trainEnd - 1);
    arma::cube trainLabels = labels.slices(0, trainEnd - 1);
    arma::cube validData   = data.slices(valStart, valEnd - 1);
    arma::cube validLabels = labels.slices(valStart, valEnd - 1);

    if (trainData.n_slices == 0) trainData = data.slices(0, foldSize - 1);
    if (validData.n_slices == 0) validData = data.slices(valStart, std::min(valStart + foldSize - 1, n - 1));

    return std::make_pair(std::make_pair(trainData, trainLabels),
                          std::make_pair(validData, validLabels));
}

// --- Fixed ratio ---
auto KFoldSplit_FixedRatio_Cube(const arma::cube& data, const arma::cube& labels,
                                size_t k, size_t fold, double trainRatio)
{
    size_t n = data.n_slices;
    size_t foldSize = n / k;
    size_t valStart = fold * foldSize;
    size_t valEnd   = (fold == k - 1) ? n : (fold + 1) * foldSize;
    size_t trainEnd = static_cast<size_t>(trainRatio * n);
    if (trainEnd < 2) trainEnd = foldSize;

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

bool TrainWithKFold_Cube(mlpack::ann::RNN<MeanSquaredError, mlpack::ann::HeInitialization>& model,
                         arma::cube& trainX,
                         arma::cube& trainY,
                         ens::Adam& optimizer,
                         int nFolds,
                         int splitMode)
{
    if (nFolds < 2)
    {
        std::cerr << "Error: nFolds must be >= 2.\n";
        return false;
    }

    const size_t nSamples = trainX.n_slices;
    if (nSamples < static_cast<size_t>(nFolds))
    {
        std::cerr << "Error: not enough samples for " << nFolds << " folds.\n";
        return false;
    }

    arma::cube X = trainX, Y = trainY;
    if (splitMode == 0)
    {
        arma::uvec indices = arma::randperm(nSamples);
        X = SelectSlices(X, indices);
        Y = SelectSlices(Y, indices);
    }

    const double trainRatio = 1.0 - (1.0 / static_cast<double>(nFolds));
    std::cout << "Starting " << nFolds << "-fold cross-validation (mode "
              << splitMode << ", train ratio ≈ " << trainRatio * 100 << "%)...\n";

    double totalValLoss = 0.0, totalR2 = 0.0;
    std::vector<double> foldLoss, foldR2;

    for (int fold = 0; fold < nFolds; ++fold)
    {
        arma::cube Xtrain, Ytrain, Xval, Yval;

        if (splitMode == 0)
        {
            auto split = KFoldSplit_Cube(X, Y, nFolds, fold);
            Xtrain = split.first.first;
            Ytrain = split.first.second;
            Xval   = split.second.first;
            Yval   = split.second.second;
        }
        else if (splitMode == 1)
        {
            auto split = KFoldSplit_TimeSeries_Cube(X, Y, nFolds, fold);
            Xtrain = split.first.first;
            Ytrain = split.first.second;
            Xval   = split.second.first;
            Yval   = split.second.second;
        }
        else
        {
            auto split = KFoldSplit_FixedRatio_Cube(X, Y, nFolds, fold, trainRatio);
            Xtrain = split.first.first;
            Ytrain = split.first.second;
            Xval   = split.second.first;
            Yval   = split.second.second;
        }

        if (Xtrain.n_slices == 0 || Xval.n_slices == 0)
        {
            std::cout << "Skipping fold " << fold + 1 << " (empty split)\n";
            continue;
        }

        std::cout << "\nFold " << (fold + 1) << " / " << nFolds
                  << " | Train: " << Xtrain.n_slices
                  << " | Validation: " << Xval.n_slices << std::endl;

        model.Reset();
        model.Train(Xtrain, Ytrain, optimizer);

        arma::cube preds;
        model.Predict(Xval, preds);

        double valMSE = ComputeMSE(preds, Yval);
        double valR2  = ComputeR2(preds, Yval);

        foldLoss.push_back(valMSE);
        foldR2.push_back(valR2);
        totalValLoss += valMSE;
        totalR2 += valR2;

        std::cout << "  Validation MSE: " << valMSE
                  << " | R²: " << valR2 << std::endl;
    }

    if (foldLoss.empty())
    {
        std::cerr << "No valid folds processed.\n";
        return false;
    }

    double avgLoss = totalValLoss / foldLoss.size();
    double avgR2   = totalR2 / foldR2.size();

    std::cout << "\nAverage validation MSE: " << avgLoss
              << " | Average R²: " << avgR2 << std::endl;

    std::cout << "Retraining final model on all data...\n";
    model.Reset();
    model.Train(trainX, trainY, optimizer);
    std::cout << "Final training completed.\n";
    return true;
}

/* --------------------------------------------------------------- */
/* ---------------- Time-Series Data Pre-Processor ---------------- */
/* --------------------------------------------------------------- */

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
    X.set_size(inputsize, rho, nSamples);
    y.set_size(outputsize, rho, nSamples);

    if (!IO)
    {
        for (size_t i = 0; i < nSamples; ++i)
        {
            X.slice(i) = dataset.submat(0, i, inputsize - 1, i + rho - 1);
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
/* --------------------------- Save ------------------------------- */
/* --------------------------------------------------------------- */

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

    arma::mat temp = predictions.slice(predictions.n_slices - 1);
    temp.save(path + "temp.txt", arma::arma_ascii);

    if (!IO) temp.insert_rows(0, inputsize, 0);
    else     temp.insert_rows(0, inputsize - outputsize, 0);

    scale.InverseTransform(temp, temp);

    temp.insert_cols(0, 1, true);
    flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);
    flatDataAndPreds.insert_rows(flatDataAndPreds.n_rows,
                                 temp.rows(temp.n_rows - outputsize, temp.n_rows - 1));

    data::Save(filename, flatDataAndPreds);

    cout << "Predicted output (last " << outputsize << "): ";
    for (int i = 0; i < outputsize; ++i)
        cout << flatDataAndPreds(flatDataAndPreds.n_rows - outputsize + i,
                                 flatDataAndPreds.n_cols - 1) << " ";
    cout << endl;
}

/* --------------------------------------------------------------- */
/* ---------------------------- main ------------------------------ */
/* --------------------------------------------------------------- */

int main()
{
    bool ASM = true, IO = false;
    size_t inputSize = 9, outputSize = 1;
    if (!ASM) inputSize = 5, outputSize = 2;

    string data_name = "NO";
    string dataFile = ASM ?
        path + "Data/observedoutput_t10&11_" + data_name + ".txt" :
        path + "Data/Google2016-2019.csv";

    const string modelFile = path + "Results/lstm_multi.bin";
    const string predFile_Test = path + "Results/lstm_multi_predictions_test.csv";
    const string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    const bool bTrain = true;
    const bool bLoadAndTrain = false;
    const double RATIO = 0.3;
    const double STEP_SIZE = 5e-5;
    const int EPOCHS = 1000;
    const int H1 = 20, H2 = 16, H3 = 14;
    const size_t BATCH_SIZE = 16;
    const int rho = 1;

    arma::mat dataset;
    cout << "Reading data ..." << endl;
    data::Load(dataFile, dataset, true);
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, RATIO, false);

    data::MinMaxScaler scale;
    scale.Fit(trainData);
    scale.Transform(trainData, trainData);
    scale.Transform(testData, testData);

    arma::cube trainX, trainY, testX, testY;
    CreateTimeSeriesData(trainData, trainX, trainY, rho, inputSize, outputSize, IO);
    CreateTimeSeriesData(testData, testX, testY, rho, inputSize, outputSize, IO);

    if (bTrain || bLoadAndTrain)
    {
        RNN<MeanSquaredError, HeInitialization> model(rho);
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

        ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-8,
                            trainData.n_cols * EPOCHS, 1e-8, true);
        optimizer.Tolerance() = -1;

        cout << "Training with 5-fold expanding window CV...\n";
        TrainWithKFold_Cube(model, trainX, trainY, optimizer, 5, 1);

        cout << "Saving model...\n";
        data::Save(modelFile, "LSTMMulti", model);
    }

    RNN<MeanSquaredError, HeInitialization> modelP(rho);
    cout << "Loading model ..." << endl;
    data::Load(modelFile, "LSTMMulti", modelP);

    arma::cube predOutP_Test, predOutP_Train;
    modelP.Predict(testX, predOutP_Test);
    modelP.Predict(trainX, predOutP_Train);

    double testMSEP = ComputeMSE(predOutP_Test, testY);
    double trainMSEP = ComputeMSE(predOutP_Train, trainY);
    double testR2 = ComputeR2(predOutP_Test, testY);
    double trainR2 = ComputeR2(predOutP_Train, trainY);

    cout << "Test MSE = " << testMSEP << ", R² = " << testR2
         << " | Train MSE = " << trainMSEP << ", R² = " << trainR2 << endl;

    arma::cube trainIO = trainX, testIO = testX;
    if (!IO)
    {
        testIO.insert_rows(testX.n_rows, testY);
        trainIO.insert_rows(trainX.n_rows, trainY);
    }

    cout << "Saving results..." << endl;
    SaveResults(predFile_Test, predOutP_Test, scale, testIO, inputSize, outputSize, IO);
    SaveResults(predFile_Train, predOutP_Train, scale, trainIO, inputSize, outputSize, IO);

    cout << "Ready." << endl;
    return 0;
}
