/**
 * @file helpers.h
 * @brief Utility declarations for the LSTM Wrapper: normalization, metrics,
 *        validation, data builders, file I/O, and configuration logging.
 *
 * These helpers support the LSTM time-series training framework for ASM-type
 * environmental and hydrological modeling applications. Functions handle
 * per-variable normalization, cube generation, metric computation, and
 * structured run configuration printing.
 *
 * @authors
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#pragma once
#include <armadillo>
#include <string>
#include <pch.h>  // must come first

/* ============================================================
 *                Metric Evaluation
 * ============================================================ */
double ComputeMSE(arma::cube& pred, arma::cube& Y);
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Normalization (Per-Variable Min–Max)
 * ============================================================ */
/**
 * @brief Compute per-variable (row-wise) min/max statistics on the training set.
 *
 * Each variable is normalized independently, preventing cross-variable scale
 * coupling. If @p normalizeOutputs is false, output rows remain unscaled.
 */
void FitMinMaxPerRow(const arma::mat& data,
                     arma::rowvec& mins,
                     arma::rowvec& maxs,
                     bool normalizeOutputs,
                     size_t inputSize,
                     size_t outputSize);

/**
 * @brief Apply previously fitted per-variable min/max normalization.
 *
 * @param data Matrix to normalize
 * @param mins Stored minima per variable
 * @param maxs Stored maxima per variable
 * @param normalizeOutputs Whether to apply normalization to outputs
 * @param inputSize Number of input features
 * @param outputSize Number of output features
 */
void TransformMinMaxPerRow(arma::mat& data,
                           const arma::rowvec& mins,
                           const arma::rowvec& maxs,
                           bool normalizeOutputs,
                           size_t inputSize,
                           size_t outputSize);

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
/**
 * @brief Save predictions and input data to CSV, optionally applying
 * inverse per-variable scaling.
 *
 * When `normalizeOutputs = false`, only input rows are inverse-transformed;
 * outputs are assumed to already be in physical units.
 */
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
 *                Configuration Logging and Validation
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
