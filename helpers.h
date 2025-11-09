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
#include <pch.h>  // must come first for mlpack linkage

/* ============================================================
 *                Normalization Type Selector
 * ============================================================ */
/**
 * @enum NormalizationType
 * @brief Enumeration of supported normalization/scaling modes.
 *
 * Defines which scaling strategy is applied to input/output data.
 *
 * - **PerVariable** : Custom per-variable min–max (manual)
 * - **MLpackMinMax**: MLpack’s built-in MinMaxScaler (0–1 range)
 * - **ZScore**      : Standardization using mean and standard deviation
 * - **None**        : No normalization applied
 */
enum class NormalizationType
{
    PerVariable = 0,   ///< Row-wise min–max scaling (default)
    MLpackMinMax = 1,  ///< MLpack MinMaxScaler (0–1)
    ZScore = 2,        ///< Standardization (mean/std)
    None = 3           ///< No normalization
};

/* ============================================================
 *                Metric Evaluation
 * ============================================================ */

/**
 * @brief Compute Mean Squared Error (MSE) between predicted and observed cubes.
 */
double ComputeMSE(arma::cube& pred, arma::cube& Y);

/**
 * @brief Compute coefficient of determination (R²) between predicted and observed cubes.
 */
double ComputeR2(arma::cube& pred, arma::cube& Y);

/* ============================================================
 *                Normalization (Per-Variable Min–Max)
 * ============================================================ */

/**
 * @brief Compute per-variable (row-wise) min/max from the entire dataset.
 *
 * Each variable (row) is scaled independently to [0,1] based on its own
 * global range across all timesteps. If @p normalizeOutputs is false,
 * output variables (after inputSize) are left unchanged.
 *
 * @param data             Input dataset (rows = variables, cols = timesteps)
 * @param mins             Output: per-variable minima
 * @param maxs             Output: per-variable maxima
 * @param normalizeOutputs Whether to include output variables in scaling
 * @param inputSize        Number of input features
 * @param outputSize       Number of output features
 */
void FitMinMaxPerRow(const arma::mat& data,
                     arma::rowvec& mins,
                     arma::rowvec& maxs,
                     bool normalizeOutputs,
                     size_t inputSize,
                     size_t outputSize);

/**
 * @brief Normalize dataset using stored per-variable min/max values.
 *
 * @param data             Matrix to normalize
 * @param mins             Stored minima per variable
 * @param maxs             Stored maxima per variable
 * @param normalizeOutputs Whether to normalize outputs
 * @param inputSize        Number of input features
 * @param outputSize       Number of output features
 */
void TransformMinMaxPerRow(arma::mat& data,
                           const arma::rowvec& mins,
                           const arma::rowvec& maxs,
                           bool normalizeOutputs,
                           size_t inputSize,
                           size_t outputSize);

/**
 * @brief Apply selected normalization strategy to training and testing datasets.
 *
 * Supports multiple normalization types: PerVariable (custom row-wise scaling),
 * MLpackMinMax (built-in 0–1 scaling), ZScore (mean/std standardization), and None.
 *
 * @param mode              Normalization type (enum)
 * @param train             Training dataset (modified in-place)
 * @param test              Testing dataset (modified in-place)
 * @param mins              Output: minima or means (for inverse transformation)
 * @param maxs              Output: maxima or std deviations (for inverse transformation)
 * @param normalizeOutputs  Whether to normalize outputs
 * @param inputSize         Number of input variables
 */
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

/**
 * @brief Print debug information about dataset and cube dimensions.
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
 * @param X       Output cube for model inputs
 * @param Y       Output cube for model targets
 * @param rho     Sequence length (number of time steps)
 * @param inputSize Number of input features
 * @param outputSize Number of output features
 * @param IO      If true, outputs are part of input block layout
 */
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
 *
 * @param filename          Output CSV file path
 * @param predictions       Predicted outputs (outputSize × nCols × rho)
 * @param mins              Per-variable minima from normalization
 * @param maxs              Per-variable maxima from normalization
 * @param IOData            Input cube (inputSize × nCols × rho)
 * @param inputSize         Number of input variables
 * @param outputSize        Number of output variables
 * @param IO                Whether IO layout is active
 * @param normalizeOutputs  Whether outputs were normalized
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
 *                Configuration Logging & Validation
 * ============================================================ */

/**
 * @brief Convert integer K-Fold mode code to readable string.
 */
std::string ModeName(int kfoldMode);

/**
 * @brief Validate K-Fold and ratio configurations, correcting invalid values.
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
