/**
 * LSTM Wrapper with K-fold (Random / TimeSeries / Fixed-Ratio) Training
 * for ASM-type time-series prediction using mlpack::RNN.
 *
 * Authors:
 *   Behzad Shakouri & Arash Massoudieh
 */

#include <QDebug>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <pch.h>

#include <ensmallen.hpp>
#include <iomanip>
#include <sstream>
#include <string>

#include "helpers.h"
#include "train_modes.h"

using namespace std;
using namespace mlpack;
using namespace ens;

// -------------------------------------------------------------------------
// Paths (select via compile definitions; provide a fallback)
// -------------------------------------------------------------------------
#ifdef PowerEdge
static std::string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif defined(Behzad)
static std::string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif defined(Arash)
static std::string path = "/home/arash/Projects/LSTM_Wrapper/";
#else
static std::string path = "./";  // fallback to current directory
#endif

// -------------------------------------------------------------------------
// Small utilities for logging config clearly
// -------------------------------------------------------------------------
static std::string ModeName(int kfoldMode)
{
    switch (kfoldMode)
    {
        case 0: return "Random";
        case 1: return "TimeSeries";
        case 2: return "FixedRatio";
        default: return "Unknown";
    }
}

static void PrintRunConfig(bool ASM, bool IO, bool bTrain, bool bLoadAndTrain,
                           size_t inputSize, size_t outputSize, int rho,
                           double stepSize, size_t epochs, size_t batchSize,
                           int H1, int H2, int H3,
                           int mode, int kfoldMode, int KFOLDS,
                           double trainRatio, double testHoldout,
                           const std::string& dataFile,
                           const std::string& modelFile,
                           const std::string& predFile_Test,
                           const std::string& predFile_Train)
{
    std::ostringstream sig;
    sig << "LSTM[H1=" << H1 << ",H2=" << H2 << ",H3=" << H3 << "] "
        << "Inp=" << inputSize << " Out=" << outputSize
        << " rho=" << rho
        << " | LR=" << std::scientific << stepSize
        << " Ep=" << std::defaultfloat << epochs
        << " B=" << batchSize;

    qInfo() << "------------------- Run Configuration -------------------";
    qInfo() << "ASM=" << ASM << " | IO=" << IO
            << " | Train=" << bTrain << " | LoadAndTrain=" << bLoadAndTrain;
    qInfo() << "Signature:" << sig.str().c_str();
    qInfo() << "Mode=" << mode << " (0=Single,1=KFold)"
            << " | KFoldMode=" << kfoldMode << "(" << ModeName(kfoldMode).c_str() << ")"
            << " | KFOLDS=" << KFOLDS;
    qInfo() << "Ratios: trainRatio=" << trainRatio << " | testHoldout=" << testHoldout;
    qInfo() << "Files:";
    qInfo() << "  dataFile      =" << dataFile.c_str();
    qInfo() << "  modelFile     =" << modelFile.c_str();
    qInfo() << "  predFile_Test =" << predFile_Test.c_str();
    qInfo() << "  predFile_Train=" << predFile_Train.c_str();
    qInfo() << "----------------------------------------------------------";
}

static void ValidateConfigOrWarn(int mode, int kfoldMode, int& KFOLDS,
                                 double& trainRatio, double& testHoldout)
{
    if (mode == 1) {
        if (KFOLDS < 2) {
            qWarning() << "[Config] KFOLDS < 2; forcing KFOLDS=2.";
            KFOLDS = 2;
        }
        if (kfoldMode == 2) { // FixedRatio
            if (!(trainRatio > 0.0 && trainRatio < 1.0)) {
                const double suggested = static_cast<double>(KFOLDS - 1) / KFOLDS;
                qWarning() << "[Config] trainRatio out of (0,1). Setting trainRatio=" << suggested;
                trainRatio = suggested;
            }
        }
    }
    if (!(testHoldout > 0.0 && testHoldout < 1.0)) {
        qWarning() << "[Config] testHoldout out of (0,1). Forcing 0.3.";
        testHoldout = 0.3;
    }
}

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
    const double STEP_SIZE  = 5e-5;
    const size_t EPOCHS     = 1000;   // tune as needed, 1000 is a good option
    const size_t BATCH_SIZE = 16;

    // --- LSTM architecture ---
    const int H1 = 20;
    const int H2 = 16;
    const int H3 = 14;

    // --- Mode selection ---
    int mode = 0;        // 0 = single, 1 = k-fold
    int kfoldMode = 2;   // 0 = Random, 1 = TimeSeries, 2 = FixedRatio
    int KFOLDS = 10;     // >= 2 for k-fold

    // --- Ratios ---
    // For single run, you might only need testHoldout (or RATIO_SINGLE passed to TrainSingle).
    const double RATIO_SINGLE = 0.3; // used in single train/test
    double trainRatio  = static_cast<double>(KFOLDS - 1) / KFOLDS; // used only when kfoldMode=FixedRatio
    double testHoldout = 0.3;                                       // used in KFold path

    // --- Files ---
    std::string data_name = "NO";
    std::string dataFile       = path + "Data/observedoutput_t10&11_" + data_name + ".txt";
    std::string modelFile      = path + "Results/lstm_multi.bin";
    std::string predFile_Test  = path + "Results/lstm_multi_predictions_test.csv";
    std::string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    // ------------------- Log + validate -------------------
    PrintRunConfig(ASM, IO, bTrain, bLoadAndTrain,
                   inputSize, outputSize, rho,
                   STEP_SIZE, EPOCHS, BATCH_SIZE,
                   H1, H2, H3,
                   mode, kfoldMode, KFOLDS,
                   trainRatio, testHoldout,
                   dataFile, modelFile, predFile_Test, predFile_Train);

    ValidateConfigOrWarn(mode, kfoldMode, KFOLDS, trainRatio, testHoldout);

    // ------------------- Execution -------------------
    if (mode == 0)
    {
        qInfo() << "Running Single-Train/Test Mode...";
        TrainSingle(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, RATIO_SINGLE,
                    STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                    bTrain, bLoadAndTrain,
                    H1, H2, H3);
    }
    else
    {
        qInfo() << "Running K-Fold Cross-Validation...";
        qInfo() << "  KFOLDS =" << KFOLDS
                << "| Mode =" << kfoldMode << "(" << ModeName(kfoldMode).c_str() << ")"
                << "| TrainRatio =" << trainRatio
                << "| TestHoldout =" << testHoldout;

        TrainKFold_WithMode(dataFile, modelFile, predFile_Test, predFile_Train,
                            inputSize, outputSize, rho, static_cast<size_t>(KFOLDS),
                            STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                            bTrain, bLoadAndTrain,
                            kfoldMode, trainRatio, testHoldout,
                            H1, H2, H3);
    }

    qInfo() << "✅ Training process completed successfully.";
    return 0;
}
