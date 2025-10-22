#pragma once
#include <string>
#include <cstddef>

#include <pch.h>        // must come first
#include "helpers.h"    // includes CreateTimeSeriesData, ComputeMSE, ComputeR2, SaveResults

/* ============================================================
 *                 K-Fold Mode Selector
 * ============================================================ */
// Mirrors FFN wrapper; defines available cross-validation strategies
enum class KFoldMode
{
    Random = 0,       // classic shuffled k-fold
    TimeSeries = 1,   // forward-chaining (chronological split, no lookahead)
    FixedRatio = 2    // fixed training prefix + moving validation fold
};

/* ============================================================
 *                 Training Function Declarations
 * ============================================================ */

/**
 * @brief Single train/test split (no cross-validation)
 *
 * @param dataFile        Path to input data file
 * @param modelFile       Path to save/load serialized model
 * @param predFile_Test   CSV output for test predictions
 * @param predFile_Train  CSV output for training predictions
 * @param inputSize       Number of input features
 * @param outputSize      Number of target outputs
 * @param rho             Sequence length (time window)
 * @param ratio           Train/test ratio (e.g. 0.7 means 70% train, 30% test)
 * @param stepSize        Learning rate for optimizer
 * @param epochs          Number of training epochs
 * @param batchSize       Batch size for optimizer
 * @param IO              Whether to use IO-type input layout
 * @param ASM             Reserved for ASM-type models (currently unused)
 * @param bTrain          If true, trains model from scratch
 * @param bLoadAndTrain   If true, loads existing model and continues training
 * @param H1,H2,H3        Hidden layer sizes for stacked LSTM layers
 */
void TrainSingle(const std::string& dataFile,
                 const std::string& modelFile,
                 const std::string& predFile_Test,
                 const std::string& predFile_Train,
                 size_t inputSize, size_t outputSize,
                 int rho, double ratio,
                 double stepSize, size_t epochs,
                 size_t batchSize, bool IO, bool ASM,
                 bool bTrain, bool bLoadAndTrain,
                 int H1, int H2, int H3);

/**
 * @brief Default K-Fold training (TimeSeries mode)
 *
 * Performs k-fold cross-validation using forward-chaining mode,
 * retrains on full dataset, and saves final model + predictions.
 */
void TrainKFold(const std::string& dataFile,
                const std::string& modelFile,
                const std::string& predFile_Test,
                const std::string& predFile_Train,
                size_t inputSize, size_t outputSize,
                int rho, int kfolds,
                double stepSize, size_t epochs,
                size_t batchSize, bool IO, bool ASM,
                bool bTrain, bool bLoadAndTrain,
                int H1, int H2, int H3);

/**
 * @brief Extended K-Fold training with selectable mode and ratios
 *
 * @param modeInt       0 = Random, 1 = TimeSeries, 2 = FixedRatio
 * @param trainRatio    Training portion (only used for FixedRatio mode)
 * @param testHoldout   Fraction of dataset reserved as final holdout test set
 */
void TrainKFold_WithMode(const std::string& dataFile,
                         const std::string& modelFile,
                         const std::string& predFile_Test,
                         const std::string& predFile_Train,
                         size_t inputSize, size_t outputSize,
                         int rho, int kfolds,
                         double stepSize, size_t epochs,
                         size_t batchSize, bool IO, bool ASM,
                         bool bTrain, bool bLoadAndTrain,
                         int modeInt,          // 0=Random, 1=TimeSeries, 2=FixedRatio
                         double trainRatio,    // used only for FixedRatio
                         double testHoldout,   // e.g. 0.3 to mirror TrainSingle
                         int H1, int H2, int H3);
