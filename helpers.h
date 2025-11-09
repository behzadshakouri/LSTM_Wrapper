/**
 * @file helpers.h
 * @brief Utility declarations for the LSTM Wrapper: normalization, metrics,
 *        validation, data builders, file I/O, and configuration logging.
 *
 * Authors:
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#pragma once

#include <armadillo>
#include <string>
#include <pch.h>  // must come first for mlpack linkage

/* ============================================================
 *                Normalization Type Enum
 * ============================================================ */
/**
 * @enum NormalizationType
 * @brief Enumeration for selecting the normalization strategy.
 *
 * 0 = PerVariable   → Custom per-row min–max scaling
 * 1 = MLpackMinMax  → mlpack::data::MinMaxScaler (0–1)
 * 2 = ZScore        → Standardization using mean/std
 * 3 = None          → No normalization
 */
enum class NormalizationType
{
    PerVariable = 0,
    MLpackMinMax = 1,
    ZScore = 2,
    None = 3
};

/* ============================================================
 *                Metric Evaluation
 * ============================================================ */

/** @brief Compute Mean Squared Error (MSE) between predicted and observed cubes. */
double ComputeMSE(arma::cube& pred, arma::cube& Y);

/** @brief Compute coefficient of determination (R²) between predicted and observed cubes. */
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Normalization (Per-Variable Min–Max)
 * ============================================================ */

void FitMinMaxPerRow(const arma::mat& data,
                     arma::rowvec& mins,
                     arma::rowvec& maxs,
                     bool normalizeOutputs,
                     size_t inputSize,
                     size_t outputSize);

void TransformMinMaxPerRow(arma::mat& data,
                           const arma::rowvec& mins,
                           const arma::rowvec& maxs,
                           bool normalizeOutputs,
                           size_t inputSize,
                           size_t outputSize);

/* ============================================================
 *                Extended Normalization Dispatcher
 * ============================================================ */
void ApplyNormalization(NormalizationType mode,
                        arma::mat& train,
                        arma::mat& test,
                        arma::rowvec& mins,
                        arma::rowvec& maxs,
                        bool normalizeOutputs,
                        size_t inputSize);

/* ============================================================
 *                Shape Validation
 * ============================================================ */
void ValidateShapes(const arma::mat& dataset,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    size_t rho);

/* ============================================================
 *                Time-Series Data Builder
 * ============================================================ */
void CreateTimeSeriesData(const arma::mat& dataset,
                          arma::cube& X,
                          arma::cube& Y,
                          size_t rho,
                          int inputSize,
                          int outputSize,
                          bool IO);

/* ============================================================
 *                Result Saving / Post-Processing
 * ============================================================ */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 const arma::rowvec& mins,
                 const arma::rowvec& maxs,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO,
                 const bool normalizeOutputs);

/* ============================================================
 *                Configuration Logging & Validation
 * ============================================================ */
std::string ModeName(int kfoldMode);

void ValidateConfigOrWarn(int mode, int kfoldMode, int& KFOLDS,
                          double& trainRatio, double& testHoldout);

void PrintRunConfig(bool ASM, bool IO, bool bTrain, bool bLoadAndTrain,
                    size_t inputSize, size_t outputSize, int rho,
                    double stepSize, size_t epochs, size_t batchSize,
                    int H1, int H2, int H3,
                    int mode, int kfoldMode, int KFOLDS,
                    double trainRatio, double testHoldout,
                    const std::string& dataFile,
                    const std::string& modelFile,
                    const std::string& predFile_Test,
                    const std::string& predFile_Train);
