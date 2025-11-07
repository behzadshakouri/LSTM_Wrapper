/**
 * @file helpers.h
 * @brief Declarations for utility functions used in the LSTM Wrapper,
 *        including metrics, validation, data builders, file I/O, and configuration logging.
 *
 * These helper utilities support the LSTM time-series training framework for ASM-type
 * environmental and hydrological modeling applications. The functions handle data
 * preprocessing, inverse-scaling, K-Fold configuration validation, and structured
 * configuration printing.
 *
 * @authors
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#pragma once
#include <armadillo>
#include <string>
#include <pch.h>  // must come first

using namespace mlpack;
using namespace mlpack::data;

/* ============================================================
 *                Metric Evaluation
 * ============================================================ */
/**
 * @brief Compute Mean Squared Error (MSE) between predicted and observed data cubes.
 * @param pred Predicted values (arma::cube)
 * @param Y Observed/true values (arma::cube)
 * @return Mean squared error value
 */
double ComputeMSE(arma::cube& pred, arma::cube& Y);

/**
 * @brief Compute the coefficient of determination (R²) between predicted and observed cubes.
 * @param pred Predicted values (arma::cube)
 * @param Y Observed/true values (arma::cube)
 * @return R² score (1 - SSR/SST)
 */
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Shape Validation
 * ============================================================ */
/**
 * @brief Print debug information about dataset and cube dimensions to verify consistency.
 *
 * Used for checking compatibility between dataset matrices and generated input/output cubes.
 *
 * @param dataset Original dataset (features × timesteps)
 * @param X Input cube (inputSize × nCols × rho)
 * @param Y Output cube (outputSize × nCols × rho)
 * @param inputSize Number of input variables
 * @param outputSize Number of output variables
 * @param rho Temporal window size (sequence length)
 */
void ValidateShapes(const arma::mat& dataset,
                    const arma::cube& X,
                    const arma::cube& Y,
                    size_t inputSize,
                    size_t outputSize,
                    size_t rho);

/* ============================================================
 *                Time-Series Data Builder
 * ============================================================ */
/**
 * @brief Create LSTM-ready time-series input/output cubes.
 *
 * Generates sliding windows of length @p rho for each input feature
 * and aligns corresponding target output sequences.
 *
 * @param dataset Input dataset matrix (features × timesteps)
 * @param X Output cube for model input sequences
 * @param Y Output cube for model target sequences
 * @param rho Sequence length (number of time steps per window)
 * @param inputSize Number of input features
 * @param outputSize Number of output features
 * @param IO If true, outputs are part of input block layout
 */
void CreateTimeSeriesData(const arma::mat& dataset,
                                 arma::cube& X,
                                 arma::cube& Y,
                                 const size_t rho,
                                 const int inputSize,
                                 const int outputSize,
                                 const bool IO);

/* ============================================================
 *                Result Saving / Post-Processing
 * ============================================================ */
/**
 * @brief Save predictions and input features to CSV.
 *
 * Robust to whether outputs were normalized or not. When
 * `normalizeOutputs = false`, the function skips inverse
 * transformation and directly saves scaled/raw data.
 *
 * @param filename          Output CSV file path
 * @param predictions       Predicted outputs (outputSize × nCols × rho)
 * @param scale             Fitted MinMaxScaler (used if normalization was applied)
 * @param IOData            Input cube (inputSize × nCols × rho)
 * @param inputSize         Number of input variables
 * @param outputSize        Number of output variables
 * @param IO                Whether IO layout is active
 * @param normalizeOutputs  Whether outputs were normalized with inputs
 */
void SaveResults(const std::string& filename,
                 const arma::cube& predictions,
                 mlpack::data::MinMaxScaler& scale,
                 const arma::cube& IOData,
                 const int inputSize,
                 const int outputSize,
                 const bool IO,
                 const bool normalizeOutputs);

/* ============================================================
 *                Configuration Logging and Validation
 * ============================================================ */
/**
 * @brief Convert integer K-Fold mode code to readable name.
 * @param kfoldMode Integer code (0=Random, 1=TimeSeries, 2=FixedRatio)
 * @return Mode name string
 */
std::string ModeName(int kfoldMode);

/**
 * @brief Validate K-Fold and ratio configurations, correcting invalid values.
 * @param mode Run mode (0=Single, 1=KFold)
 * @param kfoldMode Type of KFold splitting (0=Random, 1=TimeSeries, 2=FixedRatio)
 * @param KFOLDS Number of folds
 * @param trainRatio Fraction of training data (0–1)
 * @param testHoldout Fraction of data reserved for testing
 */
void ValidateConfigOrWarn(int mode, int kfoldMode, int& KFOLDS,
                          double& trainRatio, double& testHoldout);

/**
 * @brief Print structured summary of all run configuration parameters and file paths.
 */
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
