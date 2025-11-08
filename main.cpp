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
#include <armadillo>
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
 * Defines network architecture, optimizer, and dataset configuration,
 * then calls either `TrainSingle()` or `TrainKFold_WithMode()` depending on
 * the selected mode.
 *
 * @return 0 on successful execution.
 */
int main()
{
    // ------------------- Core Configuration -------------------
    const bool ASM  = true;              ///< Enable ASM dataset format
    const bool IO   = false;             ///< Enable Input/Output overlap mode
    const bool bTrain = true;            ///< Train a new model
    const bool bLoadAndTrain = false;    ///< Continue training existing model
    const bool NORMALIZE_OUTPUTS = true; ///< Normalize both inputs & outputs (true) or only inputs (false)

    // ------------------- Data Configuration -------------------
    const size_t inputSize  = 9;         ///< Number of input features
    const size_t outputSize = 1;         ///< Number of output variables
    const int rho           = 1;         ///< Sequence length (lag)
    const double STEP_SIZE  = 5e-5;      ///< Learning rate
    const size_t EPOCHS     = 1000;      ///< Training epochs
    const size_t BATCH_SIZE = 16;        ///< Mini-batch size

    // ------------------- LSTM Architecture -------------------
    const int H1 = 10 * 4;
    const int H2 = 10 * 4;
    const int H3 = 10 * 4;

    // ------------------- Adam Optimizer Parameters -------------------
    const double BETA1      = 0.9;     ///< First moment decay rate
    const double BETA2      = 0.999;   ///< Second moment decay rate
    const double EPSILON    = 1e-8;    ///< Numerical stability term
    const double TOLERANCE  = 1e-8;    ///< Early-stop / convergence tolerance
    const bool   SHUFFLE    = true;   ///< Keep order for time-series data

    // ------------------- Mode and Ratios -------------------
    int mode = 0;          ///< 0 = single train/test, 1 = KFold cross-validation
    int kfoldMode = 2;     ///< 0 = Random, 1 = TimeSeries, 2 = FixedRatio
    int KFOLDS = 10;
    const double RATIO_SINGLE = 0.3;   ///< 70% train / 30% test split
    double trainRatio  = static_cast<double>(KFOLDS - 1) / KFOLDS;
    double testHoldout = 0.3;

    // ------------------- File Paths -------------------
#ifdef PowerEdge
    static std::string path = "/mnt/3rd900/Projects/LSTM_Wrapper/";
#elif defined(Behzad)
    static std::string path = "/home/behzad/Projects/LSTM_Wrapper/";
#elif defined(Arash)
    static std::string path = "/home/arash/Projects/LSTM_Wrapper/";
#else
    static std::string path = "./"; ///< Fallback: current working directory
#endif

    std::string data_name = "NO"; ///< Target variable (e.g., NO, NH, sCOD, TKN, VSS, ND)
    std::string dataFile       = path + "Data/observedoutput_" + data_name + ".txt";
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
        qInfo().noquote() << "Running Single-Train/Test Mode...";
        TrainSingle(dataFile, modelFile, predFile_Test, predFile_Train,
                    inputSize, outputSize, rho, RATIO_SINGLE,
                    STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                    bTrain, bLoadAndTrain,
                    H1, H2, H3,
                    BETA1, BETA2, EPSILON, TOLERANCE,
                    SHUFFLE, NORMALIZE_OUTPUTS);
    }
    else
    {
        qInfo().noquote() << "Running K-Fold Cross-Validation...";
        TrainKFold_WithMode(dataFile, modelFile, predFile_Test, predFile_Train,
                            inputSize, outputSize, rho, static_cast<size_t>(KFOLDS),
                            STEP_SIZE, EPOCHS, BATCH_SIZE, IO, ASM,
                            bTrain, bLoadAndTrain,
                            kfoldMode, trainRatio, testHoldout,
                            H1, H2, H3,
                            BETA1, BETA2, EPSILON, TOLERANCE,
                            SHUFFLE, NORMALIZE_OUTPUTS);
    }

    qInfo().noquote() << "✅ Training process completed successfully.";
    return 0;
}
