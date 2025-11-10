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
                 bool shuffle, bool normalizeOutputs = true,
                 NormalizationType normType = NormalizationType::PerVariable,
                 bool normalize_only_Outputs = false);

/**
 * @brief Default K-Fold training using forward-chaining (TimeSeries) mode.
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
                bool shuffle, bool normalizeOutputs = true,
                NormalizationType normType = NormalizationType::PerVariable,
                bool normalize_only_Outputs = false);

/**
 * @brief Extended K-Fold training interface with selectable mode and ratios.
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
                         bool shuffle, bool normalizeOutputs = true,
                         NormalizationType normType = NormalizationType::PerVariable,
                         bool normalize_only_Outputs = false);

/* ============================================================
 *                 GridSearch_LSTM
 * ============================================================ */
void GridSearch_LSTM(const std::string& dataFile,
                     const std::string& resultsCSV,
                     const std::string& modelFolder,
                     size_t inputSize,
                     size_t outputSize,
                     bool IO,
                     bool normalizeOutputs = true,
                     NormalizationType normType = NormalizationType::PerVariable,
                     bool normalize_only_Outputs = false);
