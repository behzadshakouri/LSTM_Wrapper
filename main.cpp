/**
 * LSTM Wrapper with K-fold (Random / TimeSeries / Fixed-Ratio) Training
 * for ASM-type time-series prediction using mlpack::RNN.
 *
 * @authors
 *   Behzad Shakouri & Arash Massoudieh
 */

#include <QDebug>
#include "lstmtimeseriesset.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <pch.h>

#include <ensmallen.hpp>
#include "helpers.h"
#include "train_modes.h"

using namespace std;
using namespace mlpack;
using namespace ens;

#ifdef PowerEdge
std::string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif Behzad
std::string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif Arash
std::string path = "/home/arash/Projects/LSTM_Wrapper/";
#endif


// -------------------------------------------------------------------------
//                              Main pipeline
// -------------------------------------------------------------------------

int main()
{
    // ------------------- Configuration -------------------
    const bool ASM  = true;
    const bool IO   = false;
    const bool bTrain = true;
    const bool bLoadAndTrain = false;

    const size_t inputSize  = 9;
    const size_t outputSize = 1;
    const int rho           = 1;
    const double RATIO      = 0.3;
    const double STEP_SIZE  = 5e-5;
    const size_t EPOCHS     = 1000;   // best: 1000
    const size_t BATCH_SIZE = 16;
    const int KFOLDS        = 10;

    // --- Mode selection ---
    int mode = 1;        // 0 = single, 1 = k-fold
    int kfoldMode = 1;   // 0=Random, 1=TimeSeries, 2=FixedRatio
    double trainRatio = 0.9;   // used only in FixedRatio
    double testHoldout = 0.3;  // test split fraction

    std::string data_name = "NO";
    std::string dataFile       = path + "Data/observedoutput_t10&11_" + data_name + ".txt";
    std::string modelFile      = path + "Results/lstm_multi.bin";
    std::string predFile_Test  = path + "Results/lstm_multi_predictions_test.csv";
    std::string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    // ------------------- Execution -------------------
    if (mode == 0)
    {
        qInfo() << "Running Single-Train/Test Mode...";
        TrainSingle(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, RATIO,
                    STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                    bTrain, bLoadAndTrain);
    }
    else
    {
        qInfo() << "Running K-Fold Cross-Validation...";
        qInfo() << "  KFolds =" << KFOLDS
                << "| Mode =" << kfoldMode
                << "| TrainRatio =" << trainRatio
                << "| TestHoldout =" << testHoldout;

        TrainKFold_WithMode(dataFile, modelFile, predFile_Test, predFile_Train,
                            inputSize, outputSize, rho, KFOLDS,
                            STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                            bTrain, bLoadAndTrain,
                            kfoldMode, trainRatio, testHoldout);
    }

    qInfo() << "✅ Training process completed successfully.";
    return 0;
}
