/**
 * @file main.cpp
 * @brief Entry point for the LSTM Wrapper with single or K-Fold training modes.
 *
 * This program orchestrates LSTM-based time-series training and evaluation for
 * ASM-type environmental modeling. The configuration is logged clearly and
 * supports three K-Fold strategies: Random, TimeSeries, and FixedRatio.
 *
 * Authors:
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include <QDebug>
#include <pch.h>
#include <ensmallen.hpp>
#include <string>
#include "helpers.h"
#include "train_modes.h"

using namespace std;
using namespace mlpack;
using namespace ens;

/**
 * @brief Main entry point of the LSTM Wrapper program.
 *
 * Defines network architecture, training parameters, and dataset paths,
 * then calls either `TrainSingle()` or `TrainKFold_WithMode()` depending on
 * the selected mode.
 *
 * @return 0 on successful execution.
 */
int main()
{
    // ------------------- Core Configuration -------------------
    const bool ASM  = true;           ///< Enable ASM data
    const bool IO   = false;          ///< Enable Input/Output overlap mode
    const bool bTrain = true;         ///< Enable training
    const bool bLoadAndTrain = false; ///< Load model and continue training

    // --- Data configuration ---
    const size_t inputSize  = 9;
    const size_t outputSize = 1;
    const int rho           = 1;
    const double STEP_SIZE  = 5e-5;
    const size_t EPOCHS     = 1000;
    const size_t BATCH_SIZE = 16;

    // --- LSTM architecture ---
    const int H1 = 40, H2 = 40, H3 = 40;

    // --- Mode and ratios ---
    int mode = 1;        // 0 = single train/test, 1 = KFold cross-validation
    int kfoldMode = 2;   // 0 = Random, 1 = TimeSeries, 2 = FixedRatio
    int KFOLDS = 10;
    const double RATIO_SINGLE = 0.3;
    double trainRatio  = static_cast<double>(KFOLDS - 1) / KFOLDS;
    double testHoldout = 0.3;

    // --- File paths ---
#ifdef PowerEdge
    static std::string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif defined(Behzad)
    static std::string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif defined(Arash)
    static std::string path = "/home/arash/Projects/LSTM_Wrapper/";
#else
    static std::string path = "./"; ///< Fallback: current working directory
#endif

    std::string data_name = "sCOD"; ///< Target variable (e.g., NO, NH, sCOD, TKN, VSS, ND)
    std::string dataFile       = path + "Data/observedoutput_t10&11_" + data_name + ".txt";
    std::string modelFile      = path + "Results/lstm_multi.bin";
    std::string predFile_Test  = path + "Results/lstm_multi_predictions_test.csv";
    std::string predFile_Train = path + "Results/lstm_multi_predictions_train.csv";

    // ------------------- Logging and Validation -------------------
    PrintRunConfig(ASM, IO, bTrain, bLoadAndTrain,
                   inputSize, outputSize, rho,
                   STEP_SIZE, EPOCHS, BATCH_SIZE,
                   H1, H2, H3,
                   mode, kfoldMode, KFOLDS,
                   trainRatio, testHoldout,
                   dataFile, modelFile, predFile_Test, predFile_Train);

    ValidateConfigOrWarn(mode, kfoldMode, KFOLDS, trainRatio, testHoldout);

    // ------------------- Execute -------------------
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
