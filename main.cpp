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
    const size_t EPOCHS = 1000; // best: 1000 good: 150
    const size_t BATCH_SIZE = 16;
    const bool bTrain = true;
    const bool bLoadAndTrain = false;
    int mode = 0; // 0=single, 1=kfold
    const int KFOLDS = 10;

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

