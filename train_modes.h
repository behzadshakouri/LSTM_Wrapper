/**
 * @file train_modes.h
 * @brief Declarations of training routines and K-Fold configurations
 *        for the LSTM Wrapper (parametric Adam optimizer version).
 *
 * Provides interfaces for single-run and cross-validation (K-Fold)
 * training of LSTM-based surrogate models, supporting flexible optimizer
 * settings (Adam hyperparameters), optional output normalization, and
 * consistent time-series splitting modes for ASM-type applications.
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

#include <pch.h>        ///< must come first for mlpack linkage
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
 * - **TimeSeries**: Forward-chaining K-Fold (chronological order)
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
 * @param dataFile          Path to input data file.
 * @param modelFile         Path to save/load serialized model (.bin).
 * @param predFile_Test     CSV file for test predictions.
 * @param predFile_Train    CSV file for training predictions.
 * @param inputSize         Number of input features.
 * @param outputSize        Number of target outputs.
 * @param rho               Sequence length (time window).
 * @param ratio             Train/test ratio (e.g., 0.7 → 70% train, 30% test).
 * @param stepSize          Learning rate for Adam optimizer.
 * @param epochs            Number of training epochs.
 * @param batchSize         Batch size for optimizer.
 * @param IO                Whether to use IO-type (input/output) layout.
 * @param ASM               Reserved for ASM-type model structure (currently unused).
 * @param bTrain            If true, trains model from scratch.
 * @param bLoadAndTrain     If true, loads an existing model and continues training.
 * @param H1,H2,H3          Hidden layer sizes for stacked LSTM layers.
 * @param beta1             Adam β₁ parameter (first moment decay, default 0.9).
 * @param beta2             Adam β₂ parameter (second moment decay, default 0.999).
 * @param epsilon           Adam ε parameter (numerical stability, default 1e-8).
 * @param tolerance         Early stopping tolerance (-1 disables tolerance-based stop).
 * @param shuffle           Whether to shuffle mini-batches each epoch.
 * @param normalizeOutputs  If true, scales both inputs and outputs; if false, scales inputs only.
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
                 int H1, int H2, int H3,
                 double beta1, double beta2,
                 double epsilon, double tolerance,
                 bool shuffle, bool normalizeOutputs);

/**
 * @brief Default K-Fold training using forward-chaining (TimeSeries) mode.
 *
 * Performs K-Fold cross-validation by preserving the temporal order
 * of the data (no shuffling), then retrains the model on the combined
 * dataset and saves final weights and predictions.
 *
 * @param dataFile          Path to dataset file.
 * @param modelFile         Path to save trained model.
 * @param predFile_Test     Output file for test predictions.
 * @param predFile_Train    Output file for training predictions.
 * @param inputSize         Number of input features.
 * @param outputSize        Number of output features.
 * @param rho               Sequence length (time window).
 * @param kfolds            Number of folds (≥ 2).
 * @param stepSize          Learning rate for Adam optimizer.
 * @param epochs            Number of training epochs.
 * @param batchSize         Batch size for optimizer.
 * @param IO                Whether to use IO layout.
 * @param ASM               Reserved for ASM model compatibility.
 * @param bTrain            Whether to train the model.
 * @param bLoadAndTrain     Whether to resume training from saved weights.
 * @param H1,H2,H3          Hidden layer sizes for stacked LSTM layers.
 * @param beta1             Adam β₁ parameter (first moment decay).
 * @param beta2             Adam β₂ parameter (second moment decay).
 * @param epsilon           Adam ε parameter (numerical stability).
 * @param tolerance         Early stopping tolerance (-1 disables).
 * @param shuffle           Whether to shuffle batches each epoch.
 * @param normalizeOutputs  If true, scales both inputs and outputs; if false, scales inputs only.
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
                int H1, int H2, int H3,
                double beta1, double beta2,
                double epsilon, double tolerance,
                bool shuffle, bool normalizeOutputs);

/**
 * @brief Extended K-Fold training interface with selectable mode and ratios.
 *
 * Enables explicit selection of cross-validation mode (`Random`,
 * `TimeSeries`, or `FixedRatio`) and flexible control of the train/test
 * ratios, including a separate holdout fraction. Includes parametric
 * Adam optimizer configuration and optional output normalization.
 *
 * @param dataFile          Path to input data file.
 * @param modelFile         Path to save serialized model.
 * @param predFile_Test     CSV file for test predictions.
 * @param predFile_Train    CSV file for training predictions.
 * @param inputSize         Number of input variables.
 * @param outputSize        Number of output variables.
 * @param rho               Sequence length (number of time steps).
 * @param kfolds            Number of K-Fold partitions.
 * @param stepSize          Adam learning rate.
 * @param epochs            Total training epochs.
 * @param batchSize         Mini-batch size.
 * @param IO                Whether to use IO-layout dataset.
 * @param ASM               Reserved for ASM-style configuration.
 * @param bTrain            Whether to train a new model.
 * @param bLoadAndTrain     Whether to continue training existing model.
 * @param modeInt           0 = Random, 1 = TimeSeries, 2 = FixedRatio.
 * @param trainRatio        Training portion (used only for FixedRatio mode).
 * @param testHoldout       Fraction reserved for final holdout test set.
 * @param H1,H2,H3          Hidden layer neuron counts for LSTM stack.
 * @param beta1             Adam β₁ parameter (first moment decay).
 * @param beta2             Adam β₂ parameter (second moment decay).
 * @param epsilon           Adam ε parameter (numerical stability).
 * @param tolerance         Early stopping tolerance (-1 disables).
 * @param shuffle           Whether to shuffle batches each epoch.
 * @param normalizeOutputs  If true, scales both inputs and outputs; if false, scales inputs only.
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
                         int modeInt,
                         double trainRatio,
                         double testHoldout,
                         int H1, int H2, int H3,
                         double beta1, double beta2,
                         double epsilon, double tolerance,
                         bool shuffle, bool normalizeOutputs);
