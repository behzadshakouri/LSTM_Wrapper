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

#include "helpers.h"
#include "train_modes.h"

#ifdef PowerEdge
string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif Behzad
string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif Arash
string path = "/home/arash/Projects/LSTM_Wrapper/";
#endif

// -------------------------------------------------------------------------
//                              New pipeline
// -------------------------------------------------------------------------



int main()
{
    bool ASM = true;
    bool IO  = false;
    const size_t inputSize = 9;
    const size_t outputSize = 1;
    const int rho = 1;
    const double RATIO = 0.3;
    const double STEP_SIZE = 5e-5;
    const size_t EPOCHS = 10;
    const size_t BATCH_SIZE = 16;
    const bool bTrain = true;
    const bool bLoadAndTrain = false;
    int mode = 1; // 0=single, 1=kfold
    const int KFOLDS = 3;

    std::string data_name = "NO";
    std::string dataFile = path + "Data/observedoutput_t10&11_" + data_name + ".txt";
    std::string modelFile = path + "Results/lstm_multi.bin";
    std::string predFile_Test = path + "Results/lstm_multi_predictions_test.csv";
    std::string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    if (mode == 0)
        TrainSingle(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, RATIO,
                    STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                    bTrain, bLoadAndTrain);
    else
        TrainKFold(dataFile, modelFile, predFile_Test, predFile_Train,
                   inputSize, outputSize, rho, KFOLDS,
                   STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                   bTrain, bLoadAndTrain);

    return 0;
}



/*
int main()
{
    bool ASM = true;
    bool IO = ASM ? false : true;

    const size_t inputSize = ASM ? 9 : 5;
    const size_t outputSize = ASM ? 1 : 2;
    const int rho = 1;
    const double RATIO = 0.3;
    const double STEP_SIZE = 5e-5;
    const size_t EPOCHS = 100;
    const size_t BATCH_SIZE = 16;
    const bool bTrain = true;
    const bool bLoadAndTrain = false;

    // 0 = single (train/test only), 1 = k-fold
    int mode = 0;
    int kfold_num = 3; // k = number of folds

    std::string data_name = "NO";
    std::string dataFile = ASM ?
        path + "Data/observedoutput_t10&11_" + data_name + ".txt" :
        path + "Data/Google2016-2019.csv";

    std::string modelFile = path + "Results/lstm_multi.bin";
    std::string predFile_Test = path + "Results/lstm_multi_predictions_test.csv";
    std::string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    if (mode == 0)
    {
        TrainSingle(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, RATIO,
                    STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                    bTrain, bLoadAndTrain);
    }
    else if (mode == 1)
    {
        TrainKFold(dataFile, modelFile, predFile_Test, predFile_Train,
                   inputSize, outputSize, rho, kfold_num,
                   STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                   bTrain, bLoadAndTrain);
    }

    return 0;
}
*/
