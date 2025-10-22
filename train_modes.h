/**
 * @file train_modes.h
 * @brief Declarations of training routines and K-Fold configurations
 *        for the LSTM Wrapper.
 *
 * Provides function interfaces for single-run and cross-validation
 * (K-Fold) training of LSTM-based surrogate models, used primarily in
 * ASM-type time-series simulation and environmental model emulation.
 *
 * This header defines all available training entry points and their
 * configuration parameters, supporting reproducible model development
 * with variable depth (H1–H3), sequence length, and training modes.
 *
 * @see train_modes.cpp
 *
 * @authors
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#pragma once
#include <string>
#include <cstddef>

#include <pch.h>        ///< must come first to ensure mlpack linkage
#include "helpers.h"    ///< includes CreateTimeSeriesData, ComputeMSE, ComputeR2, SaveResults

/* ============================================================
 *                 K-Fold Mode Selector
 * ============================================================ */
/**
 * @enum KFoldMode
 * @brief Enumeration of supported cross-validation strategies.
 *
 * Defines splitting approaches used for sequential (time-series)
 * and randomized model evaluation.
 *
 * - **Random**: Traditional shuffled K-Fold
 * - **TimeSeries**: Forward-chaining K-Fold (chronological order, no lookahead)
 * - **FixedRatio**: Fixed training prefix with rolling validation window
 */
enum class KFoldMode
{
    Random = 0,       ///< Classic randomized K-Fold
    TimeSeries = 1,   ///< Chronological forward-chaining split
    FixedRatio = 2    ///< Fixed training prefix + moving validation fold
};

/* ============================================================
 *                 Training Function Declarations
 * ============================================================ */

/**
 * @brief Single train/test split (no cross-validation).
 *
 * Splits data once into training and testing sets according to
 * the provided ratio, trains (or continues training) an LSTM model,
 * and exports predictions for both partitions.
 *
 * @param dataFile        Path to input data file.
 * @param modelFile       Path to save/load serialized model (.bin).
 * @param predFile_Test   CSV file for test predictions.
 * @param predFile_Train  CSV file for training predictions.
 * @param inputSize       Number of input features.
 * @param outputSize      Number of target outputs.
 * @param rho             Sequence length (time window).
 * @param ratio           Train/test ratio (e.g., 0.7 → 70 % train, 30 % test).
 * @param stepSize        Learning rate for optimizer.
 * @param epochs          Number of training epochs.
 * @param batchSize       Batch size for optimizer.
 * @param IO              Whether to use IO-type (input/output) layout.
 * @param ASM             Reserved for ASM-type model structure (currently unused).
 * @param bTrain          If true, trains model from scratch.
 * @param bLoadAndTrain   If true, loads an existing model and continues training.
 * @param H1,H2,H3        Hidden layer sizes for stacked LSTM layers.
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
 * @brief Default K-Fold training using forward-chaining (TimeSeries) mode.
 *
 * Performs K-Fold cross-validation by preserving the temporal order
 * of the data (no shuffling), then retrains the model on the combined
 * dataset and saves final weights and predictions.
 *
 * @param dataFile        Path to dataset file.
 * @param modelFile       Path to save trained model.
 * @param predFile_Test   Output file for test predictions.
 * @param predFile_Train  Output file for training predictions.
 * @param inputSize       Number of input features.
 * @param outputSize      Number of output features.
 * @param rho             Sequence length (time window).
 * @param kfolds          Number of folds (≥ 2).
 * @param stepSize        Learning rate.
 * @param epochs          Number of training epochs.
 * @param batchSize       Batch size.
 * @param IO              Use IO layout.
 * @param ASM             Reserved for ASM model compatibility.
 * @param bTrain          Whether to train the model.
 * @param bLoadAndTrain   Whether to resume training from saved weights.
 * @param H1,H2,H3        Hidden layer sizes for stacked LSTM layers.
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
 * @brief Extended K-Fold training interface with selectable mode and ratios.
 *
 * Enables explicit selection of cross-validation mode (`Random`,
 * `TimeSeries`, or `FixedRatio`) and flexible control of the train/test
 * ratios, including a separate holdout fraction.
 *
 * @param dataFile        Path to input data file.
 * @param modelFile       Path to save serialized model.
 * @param predFile_Test   CSV file for test predictions.
 * @param predFile_Train  CSV file for training predictions.
 * @param inputSize       Number of input variables.
 * @param outputSize      Number of output variables.
 * @param rho             Sequence length (number of time steps).
 * @param kfolds          Number of K-Fold partitions.
 * @param stepSize        Optimizer learning rate.
 * @param epochs          Total training epochs.
 * @param batchSize       Mini-batch size.
 * @param IO              Whether to use IO-layout dataset.
 * @param ASM             Reserved for ASM-style configuration.
 * @param bTrain          Whether to train new model.
 * @param bLoadAndTrain   Whether to continue training existing model.
 * @param modeInt         0 = Random, 1 = TimeSeries, 2 = FixedRatio.
 * @param trainRatio      Training portion (used only for FixedRatio mode).
 * @param testHoldout     Fraction reserved for final holdout test set (e.g. 0.3).
 * @param H1,H2,H3        Hidden layer neuron counts for LSTM stack.
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
                         int modeInt,          ///< 0 = Random, 1 = TimeSeries, 2 = FixedRatio
                         double trainRatio,    ///< Training portion (only for FixedRatio)
                         double testHoldout,   ///< Holdout fraction (e.g. 0.3)
                         int H1, int H2, int H3);
